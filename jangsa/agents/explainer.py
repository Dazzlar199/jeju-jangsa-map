"""해설 에이전트: export_gate 를 통과한 집계표만 받아 쉬운 말 요약·근거 표 번호·한계를 쓴다.

LLM 모드: 통과한 표(소수 셀이 가려진 상태)와 질문만 보낸다. 응답의 숫자가 표에 있는 값인지 확인하고,
어긋나면 템플릿 해설로 대체한다. 오프라인 모드와 export_gate 를 통과하지 못한 표는 로컬 템플릿으로 설명한다.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from ..config import SYNTHETIC_NOTICE, export_rules
from ..llm import LLMError, Provider
from ..models import Explanation, Plan, StepResult
from ..runlog import LLMCall, RunLog
from ..schema import EMD_CITY
from ..tools.benchmark import BenchmarkResult
from .narrate import josa, narrate_step

EXPLAINER_SYSTEM = """당신은 제주 도민에게 동네 장사 분석 결과를 쉬운 말로 풀어 주는 '해설 에이전트'입니다.
아래 [근거표]에 있는 숫자만 사용해 JSON 하나로만 답하세요.

규칙
- 표에 없는 숫자·사실을 만들지 않는다. '비공개'로 표시된 값은 언급하지 않는다.
- 문장마다 근거표 번호를 tables 에 적는다(예: [1] 또는 [1,2]).
- 중학생도 이해할 쉬운 말, 한 문장 60자 안팎. 어려운 용어는 풀어 쓴다.
- '반드시 성공한다' 같은 단정, 투자·대출 권유를 하지 않는다. 참고 의견으로 쓴다.
- 데이터가 가상이라고 표시되어 있으면 limits 에 그 사실을 넣는다.
- 출력 형식: {"headline": "한 문장 결론", "points": [{"text": "...", "tables": [1]}], "checkpoints": ["지금 확인할 것 3개 이내"], "limits": ["한계 2~4개"]}
"""

STATIC_LIMITS = [
    "카드로 결제된 금액만 집계되어 현금·배달 앱·계좌이체 매출은 빠져 있습니다.",
    "과거 기간의 집계입니다. 앞으로의 매출이나 성공 여부를 보장하지 않습니다.",
    "읍면동 단위로 묶은 평균이라 같은 동네 안에서도 골목·건물마다 차이가 큽니다.",
]


def _table_for_llm(sr: StepResult, n: int, highlight: str | None, max_rows: int) -> str:
    df: pd.DataFrame = sr.df_export
    d = df
    if len(d) > max_rows:
        head = d.head(max_rows)
        if highlight is not None and "emd" in d.columns and highlight not in set(head["emd"]):
            head = pd.concat([head, d[d["emd"] == highlight]])
        d = head
    csv = d.astype(object).where(d.notna(), "비공개").to_csv(index=False)
    return f"[근거표 {n}] {sr.step.title}\n{csv}"


def explainer_messages(question: str, plan: Plan, results: list[StepResult], bench: BenchmarkResult | None,
                       data_kind: str, evidence_no: dict[str, int]) -> tuple[str, str]:
    mx = export_rules()["llm_max_rows_per_table"]
    parts = [f"[질문]\n{question}", f"[지역·업종] {plan.emd or '제주 전체'} / {plan.industry}", f"[데이터 종류] {data_kind}"]
    for sr in results:
        if sr.gate and sr.gate.passed and sr.df_export is not None and sr.step.id in evidence_no:
            parts.append(_table_for_llm(sr, evidence_no[sr.step.id], plan.emd, mx))
    if bench:
        parts.append("[내 가게 비교(금액 없이 비율·백분위만)]\n" + json.dumps(bench.llm_safe, ensure_ascii=False))
    return EXPLAINER_SYSTEM, "\n\n".join(parts)


def evidence_numbering(results: list[StepResult]) -> dict[str, int]:
    out, n = {}, 0
    for sr in results:
        if sr.gate and sr.gate.passed and sr.df_export is not None:
            n += 1
            out[sr.step.id] = n
    return out


def _table_numbers(results: list[StepResult]) -> list[float]:
    vals: list[float] = []
    for sr in results:
        if sr.gate and sr.gate.passed and sr.df_export is not None:
            for c in sr.df_export.columns:
                s = pd.to_numeric(sr.df_export[c], errors="coerce").dropna()
                vals.extend(float(v) for v in s)
    return vals


def numbers_supported(text: str, vals: list[float], bench: BenchmarkResult | None = None) -> tuple[bool, list[str]]:
    """문장 속 숫자가 표(또는 단위를 바꾼 값)에 있는지 느슨하게 확인한다. 작은 정수(순위·개수 ≤ 100)는 통과."""
    extra = list(bench.llm_safe.values()) if bench else []
    pool = [abs(v) for v in vals + [x for x in extra if isinstance(x, (int, float))]]
    bad = []
    for m in re.finditer(r"\d[\d,]*\.?\d*", text):
        raw = m.group(0).rstrip(".").replace(",", "")
        if not raw:
            continue
        x = float(raw)
        if x <= 100 and float(x).is_integer():
            continue
        if re.match(r"^(19|20)\d\d$", raw):
            continue
        ok = False
        for v in pool:
            for div in (1, 1e3, 1e4, 1e6, 1e8):
                w = v / div
                if w and (abs(w - x) / abs(w) <= 0.03 or abs(w - x) <= 0.06):
                    ok = True
                    break
            if ok:
                break
        if not ok:
            bad.append(m.group(0))
    return len(bad) <= max(1, len(re.findall(r"\d[\d,]*\.?\d*", text)) // 3), bad


def template_explanation(plan: Plan, results: list[StepResult], bench: BenchmarkResult | None, data_kind_synth: bool,
                         evidence_no: dict[str, int]) -> Explanation:
    points, signals, hidden = [], [], 0
    for sr in results:
        n = evidence_no.get(sr.step.id)
        if sr.skipped:
            continue
        if sr.error:
            points.append({"text": f"'{sr.step.title}' 단계는 실행하지 못했습니다({sr.error[:60]}).", "tables": []})
            continue
        if sr.gate and not sr.gate.passed:
            points.append({"text": f"'{sr.step.title}' 결과는 반출 기준 검사에서 막혀 수치를 보여 드리지 않습니다({'; '.join(sr.gate.blocked_reasons)[:80]}).", "tables": []})
            continue
        if sr.gate:
            hidden += sr.gate.suppressed_rows
        if sr.gate and sr.df_export is None:
            points.append({"text": f"'{sr.step.title}'의 수치는 반출 기준(엄격 모드)상 센터 밖으로 내보낼 수 없어, 센터 화면의 표에서만 확인하세요.", "tables": []})
            continue
        city_label = (EMD_CITY.get(plan.emd) if plan.emd else plan.city) or "제주도"
        for f in narrate_step(sr.step.id, sr.df_export, plan.emd, plan.industry or "", {"scope": plan.city or "제주 전체", "city_label": city_label}):
            points.append({"text": f.text, "tables": [n] if n else []})
            if f.signal is not None:
                signals.append(f)
    ind, who = plan.industry or "", plan.emd or plan.city or "제주 전체"
    if plan.intent == "location" and signals and plan.emd:
        pos = sum(1 for s in signals if s.signal == 1)
        neg = sum(1 for s in signals if s.signal == -1)
        names_pos = ", ".join(s.label for s in signals if s.signal == 1) or "없음"
        names_neg = ", ".join(s.label for s in signals if s.signal == -1) or "없음"
        if pos >= 2 and neg == 0:
            tone = "해볼 만한 동네로 보입니다"
        elif neg >= 2 and pos == 0:
            tone = "신중하게 봐야 할 동네로 보입니다"
        else:
            tone = "좋은 점과 걱정되는 점이 섞여 있습니다"
        headline = f"{who}에서 {ind}{josa(ind, '을/를')} 열면 확인한 신호 {len(signals)}개 중 유리 {pos}개({names_pos}), 불리 {neg}개({names_neg}). {tone}."
    elif bench:
        r = bench.ratio_vs_emd
        pct = bench.percentile_all
        part = f"내 가게 월평균 매출은 {who} {ind} 매장 평균의 {r:.0%}" if r else f"내 가게 월평균 매출은 {who} {ind} 매장과 비교했습니다"
        headline = part + (f"이고 제주 읍면동 평균 분포에서 상위 {100 - round(pct)}% 수준입니다." if pct is not None and pct == pct else ".")
    elif points:
        headline = points[0]["text"]
    else:
        headline = "보여 드릴 수 있는 결과가 없습니다. 아래 한계와 CSV 안내를 확인하세요."
    if bench:
        for t in bench.checkpoints[:2]:
            points.append({"text": t, "tables": []})
    limits = list(STATIC_LIMITS)
    if data_kind_synth:
        limits.insert(0, SYNTHETIC_NOTICE + " [실데이터] 표시가 붙은 단계만 공개 실데이터입니다.")
    if any(sr.origin == "public" for sr in results):
        limits.append("[실데이터] 단계는 공개 자료(2015~2023년)라 지금과 다를 수 있고, 업종 구분이 없거나 서귀포시만 다루는 등 범위가 좁습니다.")
    if hidden:
        limits.append(f"근거 건수가 10 미만인 {hidden}행은 반출 기준에 따라 수치를 가렸습니다. 매장이 적은 동네일수록 비교에서 빠질 수 있습니다.")
    if plan.period:
        limits.append(f"분석 기간: {plan.period[0]} ~ {plan.period[1]}(최근 12개월 기준, 계절 흐름은 가진 전체 기간).")
    cps = []
    if bench:
        cps = list(bench.checkpoints)[:3]
    else:
        cps = ["월세·권리금·보증금을 현장 부동산에서 확인하세요(이 앱에는 임대료 자료가 없습니다).",
               "같은 동네 경쟁 매장을 낮과 저녁에 직접 가 보고 손님 수와 메뉴 가격을 적어 보세요.",
               "초기 투자금과 비수기 몇 달을 버틸 운영자금을 따로 계산해 보세요."]
    return Explanation(headline=headline, points=points, checkpoints=cps[:3], limits=limits, source="template")


def explain(question: str, plan: Plan, results: list[StepResult], bench: BenchmarkResult | None, synth: bool,
            provider: Provider, log: RunLog) -> Explanation:
    en = evidence_numbering(results)
    base = template_explanation(plan, results, bench, synth, en)
    if provider.is_offline or not en:
        if not provider.is_offline:
            base.notes.append("LLM 에 보낼 수 있는 표가 없어 로컬 템플릿으로 설명했습니다.")
        return base
    kind = "가상(시연용) 데이터" if synth else "센터·공개 데이터"
    system, user = explainer_messages(question, plan, results, bench, kind, en)
    try:
        resp = provider.complete(system, user)
        log.add_call(LLMCall("해설", provider.name, resp.model, resp.in_tokens, resp.out_tokens, resp.cost_usd, resp.ms))
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", resp.text.strip())
        data = json.loads(t)
        points = [{"text": str(p["text"]), "tables": [int(x) for x in p.get("tables", []) if isinstance(x, (int, float))]}
                  for p in data["points"] if isinstance(p, dict) and "text" in p]
        headline = str(data["headline"])
        vals = _table_numbers(results)
        ok, bad = numbers_supported(headline + " " + " ".join(p["text"] for p in points), vals, bench)
        if not ok or not points:
            base.notes.append(f"LLM 해설의 숫자({', '.join(bad[:4])})가 근거표와 맞지 않아 로컬 템플릿 해설로 대체했습니다.")
            return base
        limits = [str(x) for x in data.get("limits", [])][:4]
        for fixed in ([SYNTHETIC_NOTICE] if synth else []) + STATIC_LIMITS[:1]:
            if fixed not in limits:
                limits.append(fixed)
        return Explanation(headline=headline, points=points, checkpoints=[str(x) for x in data.get("checkpoints", [])][:3] or base.checkpoints,
                           limits=limits + base.limits[-1:], source="llm")
    except (LLMError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        log.add_call(LLMCall("해설", provider.name, provider.model, 0, 0, 0.0, 0, ok=False, error=str(e)[:200]))
        base.notes.append(f"LLM 해설을 받지 못해({str(e)[:80]}) 로컬 템플릿으로 설명했습니다.")
        return base
