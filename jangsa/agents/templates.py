"""템플릿 질의: 의도별 분석 단계(SQL + 그래프). 오프라인 모드의 기본이자 LLM 이 실패했을 때의 대체 질의.

읍면동·업종은 허용 목록에 있는 값만 SQL 에 넣는다(직접 이어 붙이지만 목록 검증을 거치므로 주입 불가).
열 이름 규칙(export_gate 가 이름을 보고 종류를 판정한다):
  근거 건수 …_cnt / 응용집계(수치가 반출 가능) …_idx, …_rank, …_per_…, …_pct / 그 밖의 숫자는 단순집계(합계·평균)
그래프는 항상 응용집계 열로 그린다(엄격 모드에서 반출용 표에 그 열이 남도록).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..schema import CODES, DOWS, EMD_CITY, EMD_LIST, INDUSTRIES, TIME_BANDS, region_of

INTENT_STEPS = {
    "location": ["sales_by_emd", "sales_per_store_by_emd", "density_by_emd", "closure_by_emd", "time_band",
                 "real_tour_time", "real_resident_hourly", "real_cafe_density", "real_bus_stops"],
    "competition": ["store_trend", "density_by_emd", "closure_by_emd", "sales_per_store_by_emd", "real_cafe_density"],
    "time_pattern": ["time_band", "dow", "monthly_sales", "real_tour_time", "real_tour_dow", "real_resident_hourly"],
    "season_weather": ["monthly_sales", "rain_effect"],
    "benchmark": ["sales_by_emd", "sales_per_store_by_emd", "peer_trend"],
}
STEP_TABLES = {
    "sales_by_emd": ["card_emd"],
    "sales_per_store_by_emd": ["card_emd", "store_emd"],
    "density_by_emd": ["store_emd", "pop_flow"],
    "closure_by_emd": ["closure", "store_emd"],
    "time_band": ["card_time"],
    "dow": ["card_dow"],
    "monthly_sales": ["card_emd"],
    "rain_effect": ["card_emd", "weather_daily"],
    "store_trend": ["store_count"],
    "peer_trend": ["card_emd"],
    "real_tour_time": ["bc_tour_time"],
    "real_tour_dow": ["bc_tour_dow"],
    "real_resident_hourly": ["resident_hourly"],
    "real_cafe_density": ["cafe_license"],
    "real_bus_stops": ["bus_stops"],
}
# 표가 없으면 조용히 건너뛰는 보조 단계(센터 표에 없어 직접 만들어야 하는 집계, 또는 공개 실데이터)
OPTIONAL_STEPS = {"sales_per_store_by_emd", "density_by_emd", "closure_by_emd", "real_tour_time", "real_tour_dow",
                  "real_resident_hourly", "real_cafe_density", "real_bus_stops"}
REAL_STEPS = {"real_tour_time", "real_tour_dow", "real_resident_hourly", "real_cafe_density", "real_bus_stops"}
CAFE_LICENSE_INDUSTRIES = ("카페", "베이커리", "간이음식")


@dataclass
class Ctx:
    emd: str | None
    industry: str
    a: str            # 최근 12개월 시작 'YYYY-MM'
    b: str            # 끝
    a_all: str        # 가진 기간 전체 시작(계절 비교용)
    city: str | None = None   # 읍면동이 없을 때의 범위(제주시|서귀포시)


def _q(v: str, allowed: list[str]) -> str:
    if v not in allowed:
        raise ValueError(f"허용되지 않은 값: {v}")
    return "'" + v + "'"


def _order_case(col: str, values: list[str]) -> str:
    return "CASE " + col + " " + " ".join(f"WHEN '{v}' THEN {i}" for i, v in enumerate(values)) + " END"


def stem_of(emd: str) -> str | None:
    s = emd[:-1] if emd[-1] in "읍면동" else emd
    s = s.rstrip("0123456789")
    return s if len(s) >= 2 else None


def applicable(step_id: str, c: Ctx) -> bool:
    if step_id == "real_cafe_density":
        if c.industry not in CAFE_LICENSE_INDUSTRIES:
            return False
        if c.emd:
            return EMD_CITY.get(c.emd) == "서귀포시"
        return c.city in (None, "서귀포시")
    if step_id == "real_bus_stops":
        return bool(c.emd and stem_of(c.emd))
    if step_id in ("real_resident_hourly",):
        return bool(c.emd)
    return True


def build_step(step_id: str, c: Ctx) -> dict:
    I = _q(c.industry, INDUSTRIES)
    E = _q(c.emd, EMD_LIST) if c.emd else None
    a, b, a_all = c.a, c.b, c.a_all
    ind = c.industry
    cityset = ", ".join("'" + e + "'" for e in EMD_LIST if EMD_CITY[e] == c.city) if (c.city and not E) else ""
    who = f"{c.emd}의 " if c.emd else (f"{c.city}의 " if c.city else "제주 전체 ")
    hl = c.emd

    def ef(col: str) -> str:
        if E:
            return f" AND {col} = {E}"
        return f" AND {col} IN ({cityset})" if cityset else ""

    def cf(col: str) -> str:
        return f" AND {col} IN ({cityset})" if cityset else ""

    if step_id == "sales_by_emd":
        sql = (f"WITH t AS (SELECT emd, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt FROM card_emd "
               f"WHERE industry = {I} AND ym BETWEEN '{a}' AND '{b}'{cf('emd')} GROUP BY emd) "
               f"SELECT emd, sales_amt, sales_cnt, ROUND(sales_amt * 100.0 / AVG(sales_amt) OVER (), 1) AS sales_idx, "
               f"RANK() OVER (ORDER BY sales_amt DESC) AS sales_rank, ROUND(sales_amt * 1.0 / sales_cnt) AS ticket_per_use "
               f"FROM t ORDER BY sales_amt DESC, emd")
        return dict(id=step_id, title=f"읍면동별 {ind} 카드매출 규모(읍면동 평균=100)",
                    purpose="돈이 도는 큰 동네인지 본다(동네 크기가 반영된 규모 비교)",
                    sql=sql, applied=["sales_idx", "sales_rank", "ticket_per_use"],
                    chart=dict(kind="hbar", x="emd", y=["sales_idx"], title=f"{ind} 카드매출 규모 지수(평균=100)", y_title="지수",
                               top_n=12, sort_desc=True, highlight=hl))
    if step_id == "sales_per_store_by_emd":
        sql = (f"SELECT c.emd, ROUND(SUM(c.sales_amt) * 1.0 / (MAX(s.store_cnt) * COUNT(DISTINCT c.ym))) AS sales_per_store, "
               f"MAX(s.store_cnt) AS store_cnt, SUM(c.sales_cnt) AS sales_cnt, "
               f"RANK() OVER (ORDER BY SUM(c.sales_amt) * 1.0 / MAX(s.store_cnt) DESC) AS store_sales_rank "
               f"FROM card_emd c JOIN store_emd s ON c.emd = s.emd AND c.industry = s.industry "
               f"WHERE c.industry = {I} AND c.ym BETWEEN '{a}' AND '{b}'{cf('c.emd')} GROUP BY c.emd HAVING MAX(s.store_cnt) > 0 "
               f"ORDER BY sales_per_store DESC, c.emd")
        return dict(id=step_id, title=f"{ind} 매장 한 곳당 월 카드매출, 읍면동 비교", purpose="어느 동네가 매장당 매출이 높은지 본다",
                    sql=sql, applied=["sales_per_store", "store_sales_rank"],
                    chart=dict(kind="hbar", x="emd", y=["sales_per_store"], title=f"{ind} 매장 한 곳당 월 카드매출", y_title="백만 원",
                               scale=1e6, top_n=12, sort_desc=True, highlight=hl))
    if step_id == "density_by_emd":
        sql = (f"WITH p AS (SELECT emd, AVG(pop_cnt) AS pop FROM pop_flow WHERE ym BETWEEN '{a}' AND '{b}'{cf('emd')} GROUP BY emd) "
               f"SELECT s.emd, s.store_cnt, ROUND(p.pop) AS avg_daily_pop, ROUND(s.store_cnt * 10000.0 / p.pop, 2) AS stores_per_10k_pop, "
               f"RANK() OVER (ORDER BY s.store_cnt * 1.0 / p.pop) AS density_rank "
               f"FROM store_emd s JOIN p ON s.emd = p.emd WHERE s.industry = {I}{cf('s.emd')} ORDER BY stores_per_10k_pop, s.emd")
        return dict(id=step_id, title=f"유동인구 1만 명당 {ind} 매장 수(경쟁 밀도)", purpose="매장이 사람 수에 비해 많은지 적은지 본다",
                    sql=sql, applied=["stores_per_10k_pop", "density_rank"],
                    chart=dict(kind="hbar", x="emd", y=["stores_per_10k_pop"], title=f"유동인구 1만 명당 {ind} 매장 수(적은 순)", y_title="곳",
                               top_n=12, sort_desc=True, ascending=True, highlight=hl))
    if step_id == "closure_by_emd":
        sql = (f"SELECT z.emd, SUM(z.open_cnt) AS open_cnt, SUM(z.close_cnt) AS close_cnt, MAX(s.store_cnt) AS store_cnt, "
               f"ROUND(SUM(z.close_cnt) * 100.0 / MAX(s.store_cnt), 1) AS close_rate_pct, "
               f"RANK() OVER (ORDER BY SUM(z.close_cnt) * 1.0 / MAX(s.store_cnt)) AS close_rank "
               f"FROM closure z JOIN store_emd s ON z.emd = s.emd AND z.industry = s.industry "
               f"WHERE z.industry = {I} AND z.ym BETWEEN '{a}' AND '{b}'{cf('z.emd')} GROUP BY z.emd HAVING MAX(s.store_cnt) > 0 ORDER BY close_rate_pct, z.emd")
        return dict(id=step_id, title=f"최근 12개월 {ind} 폐업 비율(폐업 수 ÷ 매장 수)", purpose="문을 닫는 가게가 얼마나 많은지 본다",
                    sql=sql, applied=["close_rate_pct", "close_rank"],
                    chart=dict(kind="hbar", x="emd", y=["close_rate_pct"], title=f"{ind} 폐업 비율(낮은 순)", y_title="%",
                               top_n=12, sort_desc=True, ascending=True, highlight=hl))
    if step_id == "time_band":
        sql = (f"SELECT time_band, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt, "
               f"ROUND(SUM(sales_amt) * 100.0 / SUM(SUM(sales_amt)) OVER (), 1) AS sales_share_pct "
               f"FROM card_time WHERE industry = {I} AND ym BETWEEN '{a}' AND '{b}'{ef('emd')} GROUP BY time_band "
               f"ORDER BY {_order_case('time_band', TIME_BANDS)}")
        return dict(id=step_id, title=f"{who}{ind} 시간대별 카드매출 비중", purpose="몇 시에 손님이 몰리는지 본다",
                    sql=sql, applied=["sales_share_pct"],
                    chart=dict(kind="bar", x="time_band", y=["sales_share_pct"], title=f"{who}{ind} 시간대별 매출 비중", y_title="%",
                               note="오전 07-11시·점심 11-14시·오후 14-18시·저녁 18-22시·밤 22-07시"))
    if step_id == "dow":
        sql = (f"SELECT dow, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt, "
               f"ROUND(SUM(sales_amt) * 100.0 / SUM(SUM(sales_amt)) OVER (), 1) AS sales_share_pct "
               f"FROM card_dow WHERE industry = {I} AND ym BETWEEN '{a}' AND '{b}'{ef('emd')} GROUP BY dow ORDER BY {_order_case('dow', DOWS)}")
        return dict(id=step_id, title=f"{who}{ind} 요일별 카드매출 비중", purpose="어느 요일에 손님이 많은지 본다(공휴일은 따로 집계)",
                    sql=sql, applied=["sales_share_pct"],
                    chart=dict(kind="bar", x="dow", y=["sales_share_pct"], title=f"{who}{ind} 요일별 매출 비중", y_title="%"))
    if step_id == "monthly_sales":
        sql = (f"SELECT ym, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt, "
               f"ROUND(SUM(sales_amt) * 100.0 / AVG(SUM(sales_amt)) OVER (), 1) AS sales_idx FROM card_emd "
               f"WHERE industry = {I} AND ym BETWEEN '{a_all}' AND '{b}'{ef('emd')} GROUP BY ym ORDER BY ym")
        return dict(id=step_id, title=f"{who}{ind} 월별 카드매출 흐름(월평균=100)", purpose="계절에 따라 얼마나 출렁이는지 본다",
                    sql=sql, applied=["sales_idx"],
                    chart=dict(kind="line", x="ym", y=["sales_idx"], title=f"{who}{ind} 월별 카드매출 지수(월평균=100)", y_title="지수"))
    if step_id == "rain_effect":
        sql = (f"WITH m AS (SELECT ym, SUBSTR(ym, 6, 2) AS mo, SUM(sales_amt) AS sales, SUM(sales_cnt) AS cnt FROM card_emd "
               f"WHERE industry = {I}{ef('emd')} GROUP BY ym), "
               f"w AS (SELECT SUBSTR(\"date\", 1, 7) AS ym, SUM(CASE WHEN rain_mm >= 1 THEN 1 ELSE 0 END) AS rain_days FROM weather_daily GROUP BY 1), "
               f"j AS (SELECT m.ym, m.mo, m.sales, m.cnt, w.rain_days FROM m JOIN w ON m.ym = w.ym), "
               f"k AS (SELECT ym, cnt, sales / AVG(sales) OVER (PARTITION BY mo) AS sales_idx, "
               f"rain_days - AVG(rain_days) OVER (PARTITION BY mo) AS rain_diff FROM j), "
               f"g AS (SELECT cnt, sales_idx, rain_diff, CASE WHEN rain_diff > 0.5 THEN '비가 평소보다 많은 달' "
               f"WHEN rain_diff < -0.5 THEN '비가 평소보다 적은 달' ELSE '비슷한 달' END AS rain_group FROM k) "
               f"SELECT rain_group, SUM(cnt) AS sales_cnt, ROUND(AVG(rain_diff), 1) AS avg_rain_diff, "
               f"ROUND(AVG(sales_idx) * 100, 1) AS avg_sales_idx FROM g GROUP BY rain_group ORDER BY avg_rain_diff")
        return dict(id=step_id, title=f"{who}{ind} 비 온 날이 많은 달 vs 적은 달 매출(같은 달끼리 비교한 지수)",
                    purpose="날씨가 매출에 영향을 주는지 계절 효과를 빼고 본다", sql=sql, applied=["avg_rain_diff", "avg_sales_idx"],
                    chart=dict(kind="bar", x="rain_group", y=["avg_sales_idx"], title=f"{who}{ind} 비 온 날 수에 따른 매출 지수(같은 달 평균=100)", y_title="지수"))
    if step_id == "store_trend":
        city = EMD_CITY.get(c.emd) if c.emd else c.city
        cw = f" AND city = '{city}'" if city in ("제주시", "서귀포시") else ""
        sql = (f"SELECT ym, SUM(store_cnt) AS store_cnt, ROUND(SUM(store_cnt) * 100.0 / FIRST_VALUE(SUM(store_cnt)) OVER (ORDER BY ym), 1) AS store_idx "
               f"FROM store_count WHERE industry = {I} AND ym BETWEEN '{a_all}' AND '{b}'{cw} GROUP BY ym ORDER BY ym")
        return dict(id=step_id, title=f"{city or '제주도'} {ind} 매장 수 추이(시 단위, 첫 달=100)",
                    purpose="경쟁자가 늘고 있는지 줄고 있는지 본다(센터 표는 읍면동이 아닌 시 단위)", sql=sql, applied=["store_idx"],
                    chart=dict(kind="line", x="ym", y=["store_idx"], title=f"{city or '제주도'} {ind} 매장 수 지수(첫 달=100)", y_title="지수"))
    if step_id == "peer_trend":
        sql = (f"SELECT ym, SUM(sales_amt) AS sales_amt, SUM(sales_cnt) AS sales_cnt, "
               f"ROUND(SUM(sales_amt) * 100.0 / FIRST_VALUE(SUM(sales_amt)) OVER (ORDER BY ym), 1) AS sales_idx "
               f"FROM card_emd WHERE industry = {I} AND ym BETWEEN '{a}' AND '{b}'{ef('emd')} GROUP BY ym ORDER BY ym")
        return dict(id=step_id, title=f"{who}{ind} 월별 카드매출 흐름(첫 달=100, 동네 기준선)", purpose="내 가게 흐름과 견줄 동네 흐름", sql=sql,
                    applied=["sales_idx"],
                    chart=dict(kind="line", x="ym", y=["sales_idx"], title=f"{who}{ind} 월별 카드매출 지수(첫 달=100)", y_title="지수"))
    # ── 공개 실데이터 단계 ─────────────────────────────────────────────
    if step_id == "real_tour_time":
        reg = region_of(c.emd) if c.emd else None
        rw = f" WHERE region = '{reg}'" if reg else ""
        sql = (f"SELECT time_band, SUM(sales_amt) AS sales_amt, ROUND(SUM(sales_amt) * 100.0 / SUM(SUM(sales_amt)) OVER (), 1) AS sales_share_pct "
               f"FROM bc_tour_time{rw} GROUP BY time_band ORDER BY {_order_case('time_band', TIME_BANDS)}")
        return dict(id=step_id, title=f"[실데이터] {reg + ' 권역' if reg else '제주 전체'} 관광객 시간대별 카드 이용 비중(전 업종, 2015-09~2016-08)",
                    purpose="관광객이 몇 시에 쓰는지(공개 실데이터, 업종 구분 없음)", sql=sql, applied=["sales_share_pct"],
                    chart=dict(kind="bar", x="time_band", y=["sales_share_pct"], title="관광객 시간대별 카드 이용 비중(실데이터, 전 업종)", y_title="%"))
    if step_id == "real_tour_dow":
        reg = region_of(c.emd) if c.emd else None
        rw = f" WHERE region = '{reg}'" if reg else ""
        sql = (f"SELECT dow, SUM(sales_amt) AS sales_amt, ROUND(SUM(sales_amt) * 100.0 / SUM(SUM(sales_amt)) OVER (), 1) AS sales_share_pct "
               f"FROM bc_tour_dow{rw} GROUP BY dow ORDER BY {_order_case('dow', DOWS)}")
        return dict(id=step_id, title=f"[실데이터] {reg + ' 권역' if reg else '제주 전체'} 내국인 관광객 요일별 카드 이용 비중(전 업종, 2015-09~2016-08)",
                    purpose="관광객이 어느 요일에 쓰는지(공개 실데이터, 업종 구분 없음)", sql=sql, applied=["sales_share_pct"],
                    chart=dict(kind="bar", x="dow", y=["sales_share_pct"], title="내국인 관광객 요일별 카드 이용 비중(실데이터)", y_title="%"))
    if step_id == "real_resident_hourly":
        sql = (f"SELECT time_band, MAX(lot_cnt) AS lot_cnt, ROUND(SUM(pop_avg), 4) AS pop_avg, "
               f"ROUND(SUM(pop_avg) * 100.0 / SUM(SUM(pop_avg)) OVER (), 1) AS pop_share_pct FROM resident_hourly WHERE emd = {E} "
               f"GROUP BY time_band ORDER BY {_order_case('time_band', TIME_BANDS)}")
        return dict(id=step_id, title=f"[실데이터] {c.emd} 도민 유동인구 시간대 구성(지번 표본, 2015-11)",
                    purpose="동네 주민이 언제 움직이는지(제주도민 기준, 2015년 자료)", sql=sql, applied=["pop_share_pct"],
                    chart=dict(kind="bar", x="time_band", y=["pop_share_pct"], title=f"{c.emd} 도민 유동인구 시간대 구성(실데이터)", y_title="%"))
    if step_id == "real_cafe_density":
        sql = (f"SELECT emd, SUM(store_cnt) AS store_cnt, RANK() OVER (ORDER BY SUM(store_cnt) DESC) AS store_rank, "
               f"ROUND(SUM(store_cnt) * 100.0 / SUM(SUM(store_cnt)) OVER (), 1) AS store_share_pct "
               f"FROM cafe_license WHERE industry = {I} GROUP BY emd ORDER BY store_cnt DESC, emd")
        return dict(id=step_id, title=f"[실데이터] 서귀포시 {ind}(휴게음식점 신고 업소) 수, 읍면동별(2023-03)",
                    purpose="이미 얼마나 많은 가게가 있는지(서귀포시만, 업소명으로 분류)", sql=sql, applied=["store_rank", "store_share_pct"],
                    chart=dict(kind="hbar", x="emd", y=["store_cnt"], title=f"서귀포시 {ind} 업소 수(실데이터)", y_title="곳", top_n=12, sort_desc=True, highlight=hl))
    if step_id == "real_bus_stops":
        st = stem_of(c.emd)
        sql = (f"SELECT COUNT(*) AS stop_cnt, (SELECT COUNT(*) FROM bus_stops) AS total_stop_cnt FROM bus_stops "
               f"WHERE stop_name LIKE '%{st}%' OR place_desc LIKE '%{st}%'")
        return dict(id=step_id, title=f"[실데이터] 정류소 이름·주변설명에 '{st}'이 들어간 버스 정류소 수(2022-11, 참고)",
                    purpose="대중교통으로 닿는 동네인지 대략 본다(정류소 이름에 동네 이름이 든 곳만)", sql=sql, applied=[], chart=None)
    raise KeyError(step_id)


def template_steps(intent: str, c: Ctx, available: list[str] | None = None) -> list[dict]:
    """의도에 맞는 단계 목록. available 을 주면 표가 없는 보조 단계는 뺀다."""
    out = []
    for sid in INTENT_STEPS[intent]:
        if not applicable(sid, c):
            continue
        if available is not None and sid in OPTIONAL_STEPS and any(t not in available for t in STEP_TABLES[sid]):
            continue
        d = build_step(sid, c)
        d["tables"] = STEP_TABLES[sid]
        d["source"] = "template"
        d["optional"] = sid in OPTIONAL_STEPS
        out.append(d)
    return out


REAL_STEP_FILE = {
    "real_tour_time": "BC카드 관광객 시간대별 이용금액(공공데이터포털 15046093)",
    "real_tour_dow": "BC카드 관광객 요일별 이용금액(공공데이터포털 15046093)",
    "real_resident_hourly": "제주도민 시간대별 유동인구(공공데이터포털 15069890)",
    "real_cafe_density": "서귀포시 휴게음식점 현황(제주데이터허브 1306)",
    "real_bus_stops": "제주 버스정류소 현황(공공데이터포털 15010850)",
}
REAL_STEP_FETCH_KEY = {"real_tour_time": "bc", "real_tour_dow": "bc", "real_cafe_density": "cafe"}   # scripts/fetch_public_data.py 의 JOBS 키
FETCH_HINT = "받는 법: 터미널에서 `python scripts/fetch_public_data.py` 를 실행하거나 README 의 '공개 데이터 받기'를 보세요"


def missing_real_notes(intent: str, c: Ctx, available: list[str]) -> list[str]:
    """이번 질문에 쓰일 공개 실데이터 단계 중 파일이 없어 빠진 것을 안내 문장으로 돌려준다(조용히 사라지지 않게).

    건너뛴 단계에는 "[실데이터]" 표시를 붙이지 않는다(그 표시는 실제로 돌아간 단계의 해설에만 쓴다).
    Cloud 데모가 자동으로 받다가 실패했으면 그 사유를 함께 알린다.
    """
    from .. import fetch_public
    out = []
    for sid in INTENT_STEPS[intent]:
        if sid in REAL_STEPS and applicable(sid, c) and any(t not in available for t in STEP_TABLES[sid]):
            why = fetch_public.failure_reason(REAL_STEP_FETCH_KEY.get(sid, ""))
            if why is not None:
                out.append(f"공개 실데이터 '{REAL_STEP_FILE[sid]}' 단계는 건너뛰었습니다: 공개 주소에서 자동으로 받지 못했습니다({why}). "
                           "잠시 뒤 새로 고치면 다시 받아 봅니다. 가상 데이터 단계만 보여 드립니다.")
            else:
                out.append(f"공개 실데이터 '{REAL_STEP_FILE[sid]}' 단계는 건너뛰었습니다: 파일 없음 — {FETCH_HINT}.")
    return out


def tables_for(steps: list[dict]) -> list[str]:
    seen = []
    for s in steps:
        for t in s["tables"]:
            if t not in seen:
                seen.append(t)
    return seen
