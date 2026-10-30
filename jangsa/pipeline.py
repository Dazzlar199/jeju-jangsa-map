"""끝-끝 파이프라인: 질문 → 계획 → (확인) → 도구 실행 → 반출 검사 → 해설 → 기록.

원천 결과(export_gate 이전의 표)는 이 함수 안에서만 쓰고 버린다. 화면·보고서·LLM 으로 나가는 것은
export_gate 를 통과한 표뿐이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from .agents.explainer import explain, evidence_numbering
from .agents.planner import PLANNER_SYSTEM, build_plan, template_plan_for
from .config import ROOT, llm_config
from .llm import LLMError, Provider
from .models import Explanation, Plan, StepPlan, StepResult
from .runlog import LLMCall, RunLog
from .schema import TABLES, spec_text_for_llm
from .store import DataStore
from .tools.benchmark import BenchmarkResult, benchmark
from .tools.export_gate import export_gate
from .tools.sql_run import SqlSafetyError, sql_run


@dataclass
class AnalysisResult:
    plan: Plan
    steps: list[StepResult]
    explanation: Explanation
    log: RunLog
    synthetic: bool
    bench: BenchmarkResult | None = None
    evidence_no: dict[str, int] = field(default_factory=dict)
    files: list[dict] = field(default_factory=list)
    log_path: str = ""


def display_path(p) -> str:
    """화면·기록에 보이는 경로: 저장소 기준 상대 경로(PC 이름·사용자 이름이 드러나지 않게). 저장소 밖이면 파일 이름만."""
    from pathlib import Path
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def _repair_sql(step: StepPlan, error: str, store: DataStore, provider: Provider, log: RunLog) -> str | None:
    user = (f"아래 SQL 이 실행되지 않았습니다. 오류를 보고 고친 SQL 하나만 JSON 으로 답하세요: {{\"sql\": \"...\"}}\n\n"
            f"[기존 SQL]\n{step.sql}\n\n[오류]\n{error}\n\n{spec_text_for_llm([t for t in store.available() if t != 'my_store'])}")
    try:
        resp = provider.complete(PLANNER_SYSTEM, user)
        log.add_call(LLMCall("SQL 수정", provider.name, resp.model, resp.in_tokens, resp.out_tokens, resp.cost_usd, resp.ms))
        import json
        import re
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
        return str(json.loads(t)["sql"])
    except (LLMError, ValueError, KeyError, TypeError) as e:
        log.add_call(LLMCall("SQL 수정", provider.name, provider.model, 0, 0, 0.0, 0, ok=False, error=str(e)[:200]))
        return None


def execute_step(step: StepPlan, store: DataStore, provider: Provider, log: RunLog, call_limit: int) -> StepResult:
    missing = [t for t in step.tables if not store.has(t)]
    if missing:
        pub = [t for t in missing if t in TABLES and TABLES[t].public_glob]
        if pub:
            return StepResult(step=step, skipped="[실데이터] 파일 없음 — 받는 법: `python scripts/fetch_public_data.py` (README '공개 데이터 받기'): " + ", ".join(pub))
        return StepResult(step=step, skipped="필요한 CSV 가 아직 없습니다: " + ", ".join(missing))
    row_counts = {t: i.rows for t, i in store.tables.items()}
    allowed = store.available()
    cur = step
    attempts = 1
    while True:
        try:
            r = sql_run(store.con, cur.sql, allowed, row_counts)
            break
        except (SqlSafetyError, duckdb.Error) as e:
            err = str(e)
            log.event("SQL 실패", step=step.id, error=err[:200], attempt=attempts)
            if step.source == "llm" and not provider.is_offline and attempts <= 2 and log.n_calls < call_limit - 1:
                new_sql = _repair_sql(cur, err, store, provider, log)
                if new_sql:
                    cur = cur.model_copy(update={"sql": new_sql})
                    attempts += 1
                    continue
            return StepResult(step=step, error=err[:300], attempts=attempts, sql_run=cur.sql)
    if step.optional and r.rows == 0:
        return StepResult(step=step, sql_run=r.sql, skipped="이 동네·업종 자료가 없어 건너뛰었습니다.", rows=0, elapsed_ms=r.elapsed_ms)
    gate = export_gate(r.df, r.tables_used, r.source_rows, step.applied)
    log.event("export_gate", step=step.id, passed=gate.passed, summary=gate.summary)
    origins = {store.tables[t].origin for t in step.tables if t in store.tables}
    origin = "synthetic" if "synthetic" in origins else ("public" if origins == {"public"} else ("center" if "center" in origins else "mixed"))
    return StepResult(step=step, sql_run=r.sql, attempts=attempts, rows=r.rows, source_rows=r.source_rows, elapsed_ms=r.elapsed_ms,
                      gate=gate, df=gate.df if gate.passed else None, df_export=gate.df_export if gate.passed else None, origin=origin)


def _peers_from(results: list[StepResult], sid: str) -> pd.DataFrame | None:
    for sr in results:
        if sr.step.id == sid and sr.gate and sr.gate.passed:
            return sr.df
    return None


def _bench_part(plan: Plan, results: list[StepResult], store: DataStore, log: RunLog) -> tuple[BenchmarkResult | None, StepResult | None]:
    if plan.intent != "benchmark":
        return None, None
    if not store.has("my_store"):
        log.event("benchmark 건너뜀: 내 가게 CSV 없음")
        return None, None
    peers = _peers_from(results, "sales_per_store_by_emd")      # 센터 안 화면용 표(df)를 쓴다. 내 자료는 밖으로 나가지 않는다
    trend = _peers_from(results, "peer_trend")
    if trend is None:
        return None, None
    mine = store.con.execute('SELECT ym, sales_amt FROM "my_store" ORDER BY ym').df()
    bench = benchmark(mine, peers, plan.emd or "", plan.industry or "", trend)
    log.event("benchmark", months=bench.months, n_peers=bench.n_peers)
    local = None
    if trend is not None and len(trend):
        t = trend.dropna(subset=["sales_idx"]).sort_values("ym")
        m = mine.merge(t[["ym", "sales_idx"]], on="ym", how="inner")
        if len(m) >= 3:
            base = m["sales_idx"].iloc[0]
            idx = pd.DataFrame({"ym": m["ym"], "내 가게(첫 달=100)": (m["sales_amt"] / m["sales_amt"].iloc[0] * 100).round(1),
                                "동네 평균(첫 달=100)": (m["sales_idx"] / base * 100).round(1)})
            sp = StepPlan(id="my_vs_peer", title="내 가게와 동네 평균의 월별 흐름(첫 달=100)", purpose="내 가게 흐름이 동네와 같은지 본다 [내 자료: 이 화면 안에서만 사용]",
                          sql="(로컬 계산: 올린 내 가게 CSV와 동네 집계표의 지수화)", tables=["my_store", "card_emd", "store_count"],
                          chart=dict(kind="line", x="ym", y=["내 가게(첫 달=100)", "동네 평균(첫 달=100)"], title="내 가게 vs 동네 평균 흐름(첫 달=100)", y_title="지수"),
                          source="local")
            local = StepResult(step=sp, rows=len(idx), df=idx, df_export=None, sql_run=sp.sql, origin="user")
    return bench, local


def run_plan(plan: Plan, store: DataStore, provider: Provider, log: RunLog) -> AnalysisResult:
    if plan.needs_confirm:
        raise ValueError("지역·업종을 먼저 확인해야 합니다: " + ", ".join(plan.needs_confirm))
    call_limit = llm_config()["call_limit_per_question"]
    if not plan.steps:     # 데이터가 없을 때 만든 계획 → 다시 만든다
        plan = build_plan(plan.question, store, provider, log, {"intent": plan.intent, "emd": plan.emd, "industry": plan.industry})
    results = [execute_step(s, store, provider, log, call_limit) for s in plan.steps]
    llm_failed = [r for r in results if r.step.source == "llm" and (r.error or (r.gate and not r.gate.passed and r.gate.blocked_reasons))]
    if llm_failed and not provider.is_offline:
        log.event("LLM 질의 실패 → 템플릿 질의로 대체", steps=[r.step.id for r in llm_failed])
        plan = plan.model_copy(update={"steps": template_plan_for(plan, store),
                                       "notes": plan.notes + ["LLM 이 만든 SQL 이 재시도 후에도 실패해 템플릿 질의로 대체했습니다."]})
        results = [execute_step(s, store, provider, log, call_limit) for s in plan.steps]
        for r in results:
            r.fallback = True
    ran_tables = []
    for r in results:
        for t in r.step.tables:
            if t not in ran_tables and store.has(t):
                ran_tables.append(t)
    synth = store.is_synthetic(ran_tables)     # 가상 표를 하나라도 썼는지(단계별 표시는 StepResult.origin)
    files = [{"table": t, "center_no": TABLES[t].center_no, "file": store.tables[t].file_name, "rows": store.tables[t].rows,
              "origin": store.tables[t].origin, "period": store.tables[t].period} for t in ran_tables]
    bench, local = _bench_part(plan, results, store, log)
    if local is not None:
        results.append(local)
    expl = explain(plan.question, plan, [r for r in results if r.step.source != "local"], bench, synth, provider, log)
    log.steps = [{"id": r.step.id, "title": r.step.title, "source": r.step.source, "sql": r.sql_run or r.step.sql, "tables": r.step.tables,
                  "rows": r.rows, "source_rows": r.source_rows, "attempts": r.attempts, "error": r.error, "skipped": r.skipped,
                  "gate": (r.gate.summary if r.gate else ("로컬 전용(LLM·반출 대상 아님)" if r.step.source == "local" else "")),
                  "suppressed_rows": r.gate.suppressed_rows if r.gate else 0, "ms": r.elapsed_ms} for r in results]
    log.files = files
    log.plan = plan.model_dump(mode="json")
    en = evidence_numbering([r for r in results if r.step.source != "local"])
    res_origins = {r.origin for r in results if r.origin}
    res = AnalysisResult(plan, results, expl, log, synth, bench, en, files)
    try:
        res.log_path = display_path(log.save())
    except OSError:
        pass
    return res


def analyze(question: str, store: DataStore, provider: Provider, confirmed: dict | None = None) -> AnalysisResult:
    """계획부터 해설까지 한 번에(확인이 필요한 항목은 confirmed 로 미리 넘긴다)."""
    log = RunLog()
    plan = build_plan(question, store, provider, log, confirmed)
    return run_plan(plan, store, provider, log)


def display_numbers(res: AnalysisResult) -> dict[str, int]:
    """화면·보고서의 근거표 번호: LLM 이 본 번호(통과한 표)를 그대로 두고 로컬 전용 표는 그 뒤에 붙인다."""
    nums = dict(res.evidence_no)
    n = max(nums.values(), default=0)
    for r in res.steps:
        if r.step.source == "local" and r.df is not None:
            n += 1
            nums[r.step.id] = n
    return nums
