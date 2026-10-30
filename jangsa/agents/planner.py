"""계획 에이전트: 질문 → 의도·지역·업종 · 필요한 표 · "센터에서 받아 올 CSV" · 분석 단계(SQL).

LLM 에는 표 명세(열 이름·형식·코드값)와 질문만 보낸다. 데이터 값은 한 줄도 보내지 않는다.
오프라인 모드(키 없음)는 규칙으로 해석하고 템플릿 질의를 쓴다. LLM 모드에서도 의도·읍면동·업종은
허용 목록으로 검증하고, SQL 은 sql_run 의 안전 검사를 통과해야만 실행된다.
"""
from __future__ import annotations

import json
import re

from ..config import RUNS_DIR
from ..llm import LLMError, Provider, estimate_tokens
from ..models import ChecklistItem, Plan, StepPlan
from ..runlog import LLMCall, RunLog, plan_cache_key, plan_cache_path
from ..schema import EMD_LIST, INDUSTRIES, INDUSTRY_CODES, TABLES, spec_text_for_llm
from ..store import DataStore
from ..tools.sql_run import SqlSafetyError, validate_sql
from .rules import INTENTS, Parsed, missing_inputs, parse_question
from .templates import Ctx, missing_real_notes, tables_for, template_steps

PLANNER_SYSTEM = """당신은 제주도 동네 장사 진단 서비스의 '계획 에이전트'입니다.
이용자의 질문을 읽고 분석 계획을 JSON 하나로만 답하세요(설명 문장 금지).

[할 일]
1. intent: location(입지 진단) | competition(경쟁) | time_pattern(시간대·요일) | season_weather(계절·날씨) | benchmark(내 가게 비교) 중 하나
2. emd(읍면동 이름, 없으면 null), industry(업종, 없으면 null) — 아래 코드값 목록에 있는 값만
3. steps: 분석 단계 2~5개(센터 표 위주; [실데이터] 단계는 앱이 따로 붙이니 만들지 않는다). 각 단계는 {"id","title","purpose","tables":[표 이름],"sql","chart":{"kind":"bar|hbar|line","x":열,"y":[열],"title":"..","y_title":"..","highlight":읍면동 또는 null},"applied":[비율·지수 열 이름]}

[SQL 규칙 — 어기면 실행되지 않습니다]
- DuckDB 문법, SELECT 또는 WITH 한 문장만. 위에 적힌 표와 열만 사용. 파일·시스템 함수 금지.
- 반드시 GROUP BY 또는 집계 함수(SUM·AVG·COUNT 등)를 쓰고 개별 행을 꺼내지 않는다. 마지막에 LIMIT.
- 열 이름 규칙(반출 기준 검사기가 이름으로 종류를 판단): 근거 건수는 …_cnt 로 끝낸다(예: sales_cnt, avg_store_cnt).
  비율·지수·순위는 …_pct, …_idx, …_per_… 처럼 짓고 applied 에 적는다. 합계·평균 금액은 그대로 둔다.
- 모든 결과에는 근거 건수 열(…_cnt)을 하나 이상 포함한다. EXTRACT(... FROM ...) 같은 FROM 이 들어간 함수는 쓰지 않는다(SUBSTR 사용).
- 기간은 주어진 'YYYY-MM' 문자열로 ym BETWEEN 'a' AND 'b' 로 거른다.
- 지정된 읍면동과 다른 동네를 비교하려면 emd 조건을 걸지 않고 GROUP BY emd 한다.
"""


def planner_messages(question: str, parsed: Parsed, period: tuple[str, str] | None, period_all: tuple[str, str] | None,
                     table_ids: list[str] | None = None) -> tuple[str, str]:
    """LLM 에 실제로 보내는 두 덩어리(시스템, 사용자). 화면 '미리보기'와 같은 함수를 쓴다."""
    p = f"최근 12개월: {period[0]} ~ {period[1]}\n전체 기간: {period_all[0]} ~ {period_all[1]}" if period else "(기간 정보 없음)"
    user = (f"[질문]\n{question}\n\n[기간]\n{p}\n\n[규칙 기반 1차 해석(참고)]\n의도={parsed.intent}, 읍면동={parsed.emd}, 업종={parsed.industry}\n\n"
            f"{spec_text_for_llm(table_ids)}")
    return PLANNER_SYSTEM, user


def build_checklist(table_ids: list[str], store: DataStore, emd: str | None, industry: str | None) -> list[ChecklistItem]:
    items = []
    for tid in table_ids:
        spec = TABLES[tid]
        biz = "·".join(INDUSTRY_CODES.get(industry or "", [])) or "전체"
        cond = (spec.filter_hint.format(emd=emd or "전체", industry=industry or "전체", biz_code=biz) if spec.filter_hint else "")
        info = store.tables.get(tid)
        kind = ("public" if spec.public_glob else "user" if spec.export_level == "user"
                else "optional" if tid in ("store_emd", "closure") else "center")
        items.append(ChecklistItem(table_id=tid, center_no=spec.center_no, center_name=spec.center_name, condition=cond,
                                  status="불러옴" if info else "없음", origin=info.origin if info else "", kind=kind))
    return items


def _ctx(store: DataStore, emd, industry, city=None) -> tuple[Ctx | None, tuple | None, tuple | None]:
    mb = store.months_back(12)
    if not mb:
        return None, None, None
    first = store.con.execute('SELECT MIN(ym) FROM "card_emd"').fetchone()[0]
    return Ctx(emd, industry or "카페", mb[0], mb[1], first, city), mb, (first, mb[1])


def _parse_llm_json(text: str) -> dict:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _step_from_llm(d: dict, allowed: list[str]) -> StepPlan:
    sql = str(d["sql"])
    _, used = validate_sql(sql, allowed)        # 계획 단계에서 한 번 거른다(실행 때 다시 검사)
    chart = d.get("chart") if isinstance(d.get("chart"), dict) else None
    if chart:
        chart = {k: v for k, v in chart.items() if k in ("kind", "x", "y", "title", "y_title", "x_title", "highlight", "scale", "top_n", "sort_desc", "ascending")}
        if chart.get("kind") not in ("bar", "hbar", "line") or not isinstance(chart.get("y"), list):
            chart = None
    return StepPlan(id=str(d.get("id") or "step"), title=str(d.get("title") or "분석"), purpose=str(d.get("purpose") or ""),
                    sql=sql, tables=used, chart=chart, applied=[str(a) for a in d.get("applied", []) if isinstance(a, str)], source="llm")


def build_plan(question: str, store: DataStore, provider: Provider, log: RunLog, confirmed: dict | None = None,
               use_cache: bool = True, call_limit: int = 4) -> Plan:
    """질문 하나로 계획을 만든다. confirmed 로 읍면동·업종(·의도)을 이용자가 확정한 값을 덮어쓸 수 있다."""
    parsed = parse_question(question)
    intent = (confirmed or {}).get("intent") or parsed.intent
    emd = (confirmed or {}).get("emd", parsed.emd)
    industry = (confirmed or {}).get("industry", parsed.industry)
    city = None if emd else (confirmed or {}).get("city", parsed.city)
    ctx, period, period_all = _ctx(store, emd, industry, city)
    notes: list[str] = []
    plan = Plan(question=question, intent=intent, intent_label=INTENTS[intent], emd=emd, industry=industry, city=city,
                needs_confirm=missing_inputs(intent, emd, industry), period=period, period_all=period_all,
                mode=provider.name, model=provider.model)
    log.question, log.mode, log.model = question, provider.name, provider.model
    if industry is None and parsed.unsupported:
        notes.append(f"센터 카드 표의 업종 42개에는 '{parsed.unsupported}'이(가) 따로 없습니다. 가장 가까운 업종(예: 한식, 간이음식)을 골라 주세요.")
    if ctx is None or plan.needs_confirm:
        # 데이터가 없거나 확인이 필요하면 표 목록만 안내한다(분석 단계는 확정 후)
        base = template_steps(intent, Ctx(emd, industry or "카페", "0000-00", "0000-00", "0000-00"))
        plan.tables = tables_for(base)
        plan.checklist = build_checklist(plan.tables, store, emd, industry)
        if ctx is None:
            notes.append("데이터를 아직 불러오지 않았습니다. 가상 데이터를 불러오거나 센터에서 받은 CSV 를 올려 주세요.")
        plan.notes = notes
        log.plan = plan.model_dump(mode="json")
        return plan

    avail = store.available()
    steps: list[StepPlan] | None = None
    if not provider.is_offline:
        key = plan_cache_key(question, provider.name, provider.model, intent, str(emd), str(industry), str(period), spec_text_for_llm())
        cache = plan_cache_path(key, RUNS_DIR)
        if use_cache and cache.exists():
            try:
                steps = [StepPlan(**s) for s in json.loads(cache.read_text(encoding="utf-8"))["steps"]]
                plan.from_cache = True
                notes.append("같은 질문의 저장된 계획을 다시 썼습니다(같은 SQL 재현).")
            except Exception:
                steps = None
        if steps is None:
            system, user = planner_messages(question, parsed, period, period_all, [t for t in avail if t != "my_store"])
            try:
                resp = provider.complete(system, user)
                log.add_call(LLMCall("계획", provider.name, resp.model, resp.in_tokens, resp.out_tokens, resp.cost_usd, resp.ms))
                data = _parse_llm_json(resp.text)
                li, le, lind = data.get("intent"), data.get("emd"), data.get("industry")
                if li in INTENTS and not (confirmed or {}).get("intent"):
                    plan.intent, plan.intent_label = li, INTENTS[li]
                if le in EMD_LIST and "emd" not in (confirmed or {}):
                    plan.emd = le
                if lind in INDUSTRIES and "industry" not in (confirmed or {}):
                    plan.industry = lind
                steps = []
                for d in data.get("steps", [])[:6]:
                    try:
                        steps.append(_step_from_llm(d, avail))
                    except (SqlSafetyError, KeyError, ValueError, TypeError) as e:
                        notes.append(f"LLM 이 만든 단계 '{d.get('id', '?')}' 은 안전 검사에서 걸러졌습니다({e}).")
                if not steps:
                    steps = None
                    notes.append("LLM 계획에서 쓸 수 있는 단계가 없어 템플릿 질의로 대체했습니다.")
                else:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({"question": question, "steps": [s.model_dump() for s in steps]}, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
            except (LLMError, json.JSONDecodeError, KeyError, TypeError) as e:
                log.add_call(LLMCall("계획", provider.name, provider.model, 0, 0, 0.0, 0, ok=False, error=str(e)[:200]))
                notes.append(f"LLM 계획을 받지 못해({str(e)[:80]}) 템플릿 질의로 대체했습니다.")
                steps = None
    if steps is None:
        c2 = Ctx(plan.emd, plan.industry or "카페", ctx.a, ctx.b, ctx.a_all, plan.city)
        steps = [StepPlan(**d) for d in template_steps(plan.intent, c2, avail)]
        if provider.is_offline:
            notes.append("오프라인 모드: 규칙으로 질문을 해석하고 템플릿 질의를 썼습니다(LLM 호출 0회).")
    if all(s.source == "template" for s in steps):
        notes += missing_real_notes(plan.intent, Ctx(plan.emd, plan.industry or "카페", ctx.a, ctx.b, ctx.a_all, plan.city), avail)
    plan.steps = steps
    plan.tables = tables_for([s.model_dump() for s in steps])
    plan.checklist = build_checklist(plan.tables, store, plan.emd, plan.industry)
    plan.notes = notes
    log.plan = plan.model_dump(mode="json")
    log.preview["planner"] = {"tokens_est": estimate_tokens(PLANNER_SYSTEM)}
    return plan


def template_plan_for(plan: Plan, store: DataStore) -> list[StepPlan]:
    """LLM 단계가 끝내 실패했을 때 쓰는 대체 질의."""
    ctx, _, _ = _ctx(store, plan.emd, plan.industry, plan.city)
    return [StepPlan(**d) for d in template_steps(plan.intent, ctx, store.available())]
