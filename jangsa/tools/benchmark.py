"""도구 5. benchmark: 내 가게 월 매출 vs 같은 업종·동네의 매장당 월 매출 백분위.

입력 두 가지 모두 '이미 모은 값'이다.
 - peers: 읍면동별 매장당 월평균 카드매출(단순집계 후, export_gate 통과한 것만 쓴다)
 - mine : 이용자가 올린 월 매출 CSV(로컬에서만 계산, LLM 에는 비율·백분위만 넘어감)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BenchmarkResult:
    emd: str
    industry: str
    months: int
    my_avg_monthly: float
    peer_emd_avg: float | None
    ratio_vs_emd: float | None            # 내 매출 / 동네 매장당 평균
    percentile_all: float | None          # 제주 전체 읍면동 평균 분포에서의 위치(0~100, 높을수록 상위)
    percentile_emd_rank: int | None
    n_peers: int
    my_trend_pct: float | None            # 최근 3개월 vs 앞 3개월 (%)
    peer_trend_pct: float | None
    checkpoints: list[str] = field(default_factory=list)
    llm_safe: dict = field(default_factory=dict)   # LLM 해설에 넘겨도 되는 값(금액 없음)


def percentile_of(value: float, peers: pd.Series) -> float:
    s = peers.dropna()
    if s.empty:
        return float("nan")
    return float((s < value).mean() * 100 + (s == value).mean() * 50)


def _trend(series: pd.Series) -> float | None:
    s = series.dropna().reset_index(drop=True)
    if len(s) < 6:
        return None
    a, b = s.iloc[-6:-3].mean(), s.iloc[-3:].mean()
    return float((b / a - 1) * 100) if a else None


def benchmark(my_monthly: pd.DataFrame, peers: pd.DataFrame | None, emd: str, industry: str,
              peer_trend: pd.DataFrame | None = None) -> BenchmarkResult:
    """my_monthly: ym, sales_amt / peers: emd, sales_per_store (+ n 열) / peer_trend: ym, sales_idx (내 동네 월별 카드매출 지수)."""
    mine = my_monthly.dropna(subset=["sales_amt"]).sort_values("ym")
    if mine.empty:
        raise ValueError("내 가게 월 매출 값이 없습니다.")
    avg = float(mine["sales_amt"].mean())
    p = peers.dropna(subset=["sales_per_store"]) if peers is not None and "sales_per_store" in peers else pd.DataFrame(columns=["emd", "sales_per_store"])
    row = p[p["emd"] == emd]
    emd_avg = float(row["sales_per_store"].iloc[0]) if not row.empty else None
    ratio = avg / emd_avg if emd_avg else None
    pct = percentile_of(avg, p["sales_per_store"]) if len(p) else None
    rank = int((p["sales_per_store"] > avg).sum()) + 1 if len(p) else None
    my_tr = _trend(mine["sales_amt"])
    pe_tr = _trend(peer_trend.sort_values("ym")["sales_idx"]) if peer_trend is not None and len(peer_trend) else None

    cps = []
    if ratio is not None:
        if ratio >= 1.1:
            cps.append(f"내 월평균 매출은 동네 같은 업종 매장 평균의 {ratio:.0%}입니다. 평균보다 높으니 이 수준을 지킬 방법(단골·시간대)을 먼저 확인하세요.")
        elif ratio <= 0.9:
            cps.append(f"내 월평균 매출은 동네 같은 업종 매장 평균의 {ratio:.0%}입니다. 평균보다 낮으니 시간대별 손님 수와 객단가부터 살펴보세요.")
        else:
            cps.append(f"내 월평균 매출은 동네 같은 업종 매장 평균과 비슷합니다({ratio:.0%}).")
    if my_tr is not None and pe_tr is not None:
        gap = my_tr - pe_tr
        if gap <= -5:
            cps.append(f"최근 3개월 흐름은 동네보다 약합니다(내 {my_tr:+.0f}% / 동네 {pe_tr:+.0f}%). 계절 요인인지 내 가게 요인인지 나눠 보세요.")
        elif gap >= 5:
            cps.append(f"최근 3개월 흐름은 동네보다 좋습니다(내 {my_tr:+.0f}% / 동네 {pe_tr:+.0f}%).")
        else:
            cps.append(f"최근 3개월 흐름은 동네와 비슷합니다(내 {my_tr:+.0f}% / 동네 {pe_tr:+.0f}%).")
    if ratio is None:
        cps.append("동네 매장당 매출 비교에는 읍면동별 매장 수 CSV 가 필요한데 없어서, 월별 흐름만 견줬습니다.")
    cps.append("카드 매출만 비교한 것입니다. 현금·배달 앱·예약 매출은 빠져 있으니 내 장부와 함께 보세요.")
    return BenchmarkResult(
        emd, industry, len(mine), avg, emd_avg, ratio, pct, rank, len(p), my_tr, pe_tr, cps[:3],
        llm_safe={"내 매출/동네 평균": round(ratio, 2) if ratio else None,
                  "제주 읍면동 중 백분위(상위일수록 높음)": round(pct) if pct == pct and pct is not None else None,
                  "내 최근3개월 증감%": round(my_tr, 1) if my_tr is not None else None,
                  "동네 최근3개월 증감%": round(pe_tr, 1) if pe_tr is not None else None})
