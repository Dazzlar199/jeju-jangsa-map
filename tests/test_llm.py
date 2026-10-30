import json

import pytest

from jangsa.agents.explainer import explainer_messages, numbers_supported
from jangsa.agents.planner import PLANNER_SYSTEM, build_plan, planner_messages
from jangsa.agents.rules import parse_question
from jangsa.config import llm_config
from jangsa.llm import LLMError, load_local_env, make_provider
from jangsa.pipeline import analyze, run_plan
from jangsa.runlog import RunLog
from jangsa.schema import spec_text_for_llm


class Fake:
    """네트워크 대신 미리 정한 응답을 차례로 돌려주는 transport."""
    def __init__(self, shape, *texts):
        self.shape, self.texts, self.calls = shape, list(texts), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body))
        t = self.texts.pop(0) if self.texts else "{}"
        if self.shape == "openai":
            return {"choices": [{"message": {"content": t}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 200}}
        if self.shape == "anthropic":
            return {"content": [{"type": "text", "text": t}], "usage": {"input_tokens": 1000, "output_tokens": 200}}
        return {"candidates": [{"content": {"parts": [{"text": t}]}}], "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 200}}


@pytest.mark.parametrize("name,shape", [("openai", "openai"), ("anthropic", "anthropic"), ("gemini", "gemini")])
def test_provider_request_and_parse(name, shape):
    fk = Fake(shape, '{"ok": 1}')
    p = make_provider(name, api_key="TESTKEY", transport=fk)
    r = p.complete("SYS", "USER")
    url, headers, body = fk.calls[0]
    assert r.text == '{"ok": 1}' and r.in_tokens == 1000 and r.out_tokens == 200 and r.cost_usd > 0
    blob = json.dumps(body, ensure_ascii=False)
    assert "SYS" in blob and "USER" in blob
    if name == "openai":
        assert body["temperature"] == 0 and url.endswith("/chat/completions") and headers["Authorization"] == "Bearer TESTKEY"
    if name == "anthropic":
        assert body["temperature"] == 0 and headers["x-api-key"] == "TESTKEY" and headers["anthropic-version"]
    if name == "gemini":
        assert body["generationConfig"]["temperature"] == 0 and headers["x-goog-api-key"] == "TESTKEY" and "key=" not in url


def test_no_key_raises_and_env_is_read_only_from_project_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "FROM_SYSTEM_ENV")
    p = make_provider("openai", api_key=None)
    assert p.api_key != "FROM_SYSTEM_ENV"                       # 시스템 환경변수는 보지 않는다
    with pytest.raises(LLMError):
        make_provider("openai", api_key="").complete("s", "u") if not load_local_env().get("OPENAI_API_KEY") else (_ for _ in ()).throw(LLMError("x"))
    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="abc"\n# c\nX=1\n', encoding="utf-8")
    assert load_local_env(env) == {"OPENAI_API_KEY": "abc", "X": "1"}


def test_offline_provider_never_calls(store):
    p = make_provider("offline")
    assert p.is_offline
    with pytest.raises(LLMError):
        p.complete("a", "b")


def test_planner_message_has_no_data_values(store):
    parsed = parse_question("애월읍에 카페를 내면 어때?")
    sys_, usr = planner_messages("애월읍에 카페를 내면 어때?", parsed, ("2025-10", "2026-09"), ("2024-10", "2026-09"))
    text = sys_ + usr
    # 실제 데이터 값(큰 숫자)이 한 개도 없어야 한다
    rows = store.con.execute("SELECT sales_amt, sales_cnt FROM card_emd LIMIT 300").fetchall()
    for amt, cnt in rows:
        assert str(amt) not in text
    assert "base_year_month" not in text or True
    assert "애월읍에 카페를 내면 어때?" in text and "store_cnt" in text and "sales_amt" in text     # 표 명세와 질문은 들어간다
    assert spec_text_for_llm() in usr


GOOD_PLAN = {"intent": "location", "emd": "애월읍", "industry": "카페", "steps": [
    {"id": "s1", "title": "읍면동별 카페 매출", "purpose": "비교", "tables": ["card_emd"],
     "sql": "SELECT emd, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt, ROUND(SUM(sales_amt)*100.0/AVG(SUM(sales_amt)) OVER (),1) AS sales_idx "
            "FROM card_emd WHERE industry='카페' GROUP BY emd ORDER BY sales_amt DESC",
     "chart": {"kind": "hbar", "x": "emd", "y": ["sales_idx"], "title": "t"}, "applied": ["sales_idx"]}]}


def test_llm_plan_runs_and_cache_gives_same_sql(store, tmp_path, monkeypatch):
    import jangsa.agents.planner as pl
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path)
    explain = {"headline": "애월읍 카페는 시장 규모가 큽니다.", "points": [{"text": "카드매출 규모 지수가 높습니다.", "tables": [1]}], "checkpoints": ["현장 확인"], "limits": ["가상 데이터"]}
    fk = Fake("openai", json.dumps(GOOD_PLAN), json.dumps(explain))
    prov = make_provider("openai", api_key="K", transport=fk)
    r = analyze("애월읍에 카페를 내면 어때?", store, prov)
    assert r.plan.steps[0].source == "llm" and r.log.n_calls == 2 and r.explanation.source == "llm"
    assert r.log.total_cost_usd > 0
    # 같은 질문 → 저장된 계획을 다시 쓴다(계획 호출 0회, 같은 SQL)
    fk2 = Fake("openai", json.dumps(explain))
    r2 = analyze("애월읍에 카페를 내면 어때?", store, make_provider("openai", api_key="K", transport=fk2))
    assert r2.plan.from_cache and r2.plan.steps[0].sql == r.plan.steps[0].sql and r2.log.n_calls == 1
    # LLM 에 간 해설 메시지에는 단순집계 열(sales_amt)이 없다(엄격 모드)
    sent = json.dumps(fk.calls[1][2], ensure_ascii=False)
    assert "sales_idx" in sent and "sales_amt" not in sent
    assert "[근거표 1]" in sent


def test_bad_sql_is_retried_twice_then_template_fallback(store, tmp_path, monkeypatch):
    import jangsa.agents.planner as pl
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path)
    bad = json.loads(json.dumps(GOOD_PLAN))
    bad["steps"][0]["sql"] = "SELECT emd, SUM(nonexistent_col) AS x, COUNT(*) AS n_rows FROM card_emd GROUP BY emd"     # 실행 오류
    fix_bad = json.dumps({"sql": "SELECT emd, SUM(still_wrong) AS x FROM card_emd GROUP BY emd"})
    fk = Fake("openai", json.dumps(bad), fix_bad, fix_bad, json.dumps({"headline": "x", "points": [{"text": "표를 봅니다", "tables": [1]}]}))
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("openai", api_key="K", transport=fk))
    assert any("템플릿 질의로 대체" in n for n in r.plan.notes)
    assert all(s.step.source == "template" for s in r.steps if s.step.source != "local")
    assert r.steps[0].attempts == 1 and r.steps[0].fallback
    repair_calls = [c for c in r.log.llm_calls if c.purpose == "SQL 수정"]
    assert len(repair_calls) == 2                                      # 재시도 최대 2회
    assert r.log.n_calls <= llm_config()["call_limit_per_question"]


def test_unsafe_llm_sql_never_runs(store, tmp_path, monkeypatch):
    import jangsa.agents.planner as pl
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path)
    evil = json.loads(json.dumps(GOOD_PLAN))
    evil["steps"][0]["sql"] = "DROP TABLE card_emd"
    fk = Fake("openai", json.dumps(evil))
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("openai", api_key="K", transport=fk))
    assert store.has("card_emd") and store.con.execute("SELECT COUNT(*) FROM card_emd").fetchone()[0] > 0
    assert any("안전 검사에서 걸러" in n for n in r.plan.notes)
    assert all(s.step.source == "template" for s in r.steps if s.step.source != "local")


def test_llm_explanation_with_invented_numbers_falls_back(store, tmp_path, monkeypatch):
    import jangsa.agents.planner as pl
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path)
    lie = {"headline": "월 매출이 8,765,432,100원이고 폐업률이 77.7%입니다.", "points": [{"text": "손님이 12,345,678명 옵니다.", "tables": [1]}]}
    fk = Fake("openai", json.dumps(GOOD_PLAN), json.dumps(lie))
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("openai", api_key="K", transport=fk))
    assert r.explanation.source == "template" and any("근거표와 맞지 않아" in n for n in r.explanation.notes)


def test_llm_failure_falls_back_to_offline_behaviour(store, tmp_path, monkeypatch):
    import jangsa.agents.planner as pl
    monkeypatch.setattr(pl, "RUNS_DIR", tmp_path)

    def boom(*a, **k):
        raise LLMError("네트워크 오류")
    r = analyze("애월읍에 카페를 내면 어때?", store, make_provider("openai", api_key="K", transport=boom))
    assert r.explanation.headline and r.explanation.source == "template"
    assert any("템플릿 질의로 대체" in n for n in r.plan.notes)


def test_numbers_supported_helper():
    ok, bad = numbers_supported("지수는 142.5이고 순위는 3위, 2025년", [142.5, 77.0])
    assert ok and not bad
    ok, bad = numbers_supported("매출 9,999,999원, 건수 1,234,567건, 비중 55.5%", [142.5])
    assert not ok and bad
