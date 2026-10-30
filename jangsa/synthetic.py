"""가상 데이터 생성기. 고정 시드라 언제 돌려도 같은 값이 나온다.

주의: 여기서 만드는 값은 모두 지어낸 것이다. 제주의 실제 통계가 아니다.
- 파일은 '가상_' 접두사와 '데이터구분=가상' 열을 갖고, 열 이름·값 형식은 센터 명세(docs/data-spec.md)를 따른다
  (base_year_month=YYYYMM, bjd=법정동, biz_cd/biz_nm=업종코드·이름, user_type, time=오전~밤, week=월요일~공휴일 …).
- 분포는 그럴듯하게만 맞췄다: 관광지 읍면은 여름·주말이 높고, 도심은 평일 점심·저녁이 높고,
  작은 읍면(추자·우도 등)은 업종별 매장이 적어 '10 미만' 억제가 실제로 걸린다.
- 용량 때문에 시간대·요일 표는 최근 12개월, 유동인구는 달마다 4일×4시각 표본만 만든다.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_SYNTHETIC
from .schema import (DOWS, DSTRCT_SE, EMD_CITY, EMD_INFO, INDUSTRIES, INDUSTRY_CODES, LEGAL_TO_ADMIN, TABLES, TIME_BANDS)

SEED = 20261030
START_YM = "2024-10"
N_MONTHS = 24
LABEL_COL = "데이터구분"
LABEL_VALUE = "가상"

BASE = {
    "카페":     dict(stores=18, sales=18_000_000, ticket=7_500),
    "한식":     dict(stores=26, sales=22_000_000, ticket=14_000),
    "일식":     dict(stores=6,  sales=25_000_000, ticket=22_000),
    "중식":     dict(stores=6,  sales=17_000_000, ticket=11_000),
    "양식":     dict(stores=7,  sales=24_000_000, ticket=25_000),
    "간이음식": dict(stores=8,  sales=13_000_000, ticket=9_000),
    "주점":     dict(stores=10, sales=26_000_000, ticket=30_000),
    "편의점":   dict(stores=8,  sales=55_000_000, ticket=6_500),
    "베이커리": dict(stores=5,  sales=16_000_000, ticket=9_000),
}
# 센터 업종명(1-08 [참고2]). 주점은 두 코드로 나뉜다(읽을 때 합산).
BIZ = {
    "카페":     [("I56220", "비알콜 음료점업", 1.0)],
    "한식":     [("I56111", "한식 음식점업", 1.0)],
    "중식":     [("I56112", "중식 음식점업", 1.0)],
    "일식":     [("I56113", "일식 음식점업", 1.0)],
    "양식":     [("I56114", "서양식 음식점업", 1.0)],
    "간이음식": [("I56192", "피자, 햄버거, 샌드위치 및 유사 음식점업", 1.0)],
    "주점":     [("I56211", "일반유흥 주점업", 0.7), ("I56219", "기타 주점업", 0.3)],
    "편의점":   [("G47122", "체인화 편의점", 1.0)],
    "베이커리": [("G47215", "빵 및 과자류 소매업", 1.0)],
}
TYPE_STORE_MULT = {
    "tour":  {"카페": 1.6, "일식": 1.2, "한식": 1.1, "주점": 0.9},
    "urban": {"주점": 1.5, "간이음식": 1.2, "편의점": 1.4, "양식": 1.2},
    "rural": {k: 0.55 for k in INDUSTRIES},
}
TYPE_SALES_MULT = {"tour": 1.15, "urban": 1.0, "rural": 0.75}
TOURIST_FACTOR = {"tour": 1.0, "urban": 0.25, "rural": 0.1}
SEASON = {1: .80, 2: .85, 3: 1.0, 4: 1.1, 5: 1.15, 6: 1.05, 7: 1.35, 8: 1.45, 9: 1.05, 10: 1.2, 11: .95, 12: .85}
TIME_SHARE = {   # 오전 점심 오후 저녁 밤
    "카페":     [.12, .22, .36, .24, .06],
    "한식":     [.05, .38, .12, .38, .07],
    "일식":     [.02, .32, .12, .44, .10],
    "중식":     [.03, .38, .14, .37, .08],
    "양식":     [.02, .30, .14, .46, .08],
    "간이음식": [.04, .30, .22, .32, .12],
    "주점":     [.00, .03, .07, .30, .60],
    "편의점":   [.20, .22, .22, .22, .14],
    "베이커리": [.28, .22, .30, .17, .03],
}
DOW_SHARE_TOUR = [.12, .12, .12, .13, .16, .18, .17]
DOW_SHARE_URBAN = [.14, .14, .14, .15, .17, .14, .12]
DOW_ADJ = {"주점": [.8, .8, .9, 1.0, 1.5, 1.5, 0.9], "간이음식": [.9, .9, .9, 1.0, 1.3, 1.3, 1.2]}
HOLIDAY_SHARE = 0.03
ADMIN_TO_LEGAL: dict[str, list[str]] = {}
for _l, _a in LEGAL_TO_ADMIN.items():
    ADMIN_TO_LEGAL.setdefault(_a, []).append(_l)


def month_list() -> list[str]:
    y, m = map(int, START_YM.split("-"))
    out = []
    for _ in range(N_MONTHS):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _season(month: int, tf: float) -> float:
    return 1 + tf * (SEASON[month] - 1)


def _weather(rng: np.random.Generator) -> pd.DataFrame:
    first = pd.Timestamp(f"{START_YM}-01")
    last = pd.Timestamp(month_list()[-1] + "-01") + pd.offsets.MonthEnd(0)
    p_rain = {1: .30, 2: .28, 3: .30, 4: .30, 5: .33, 6: .42, 7: .47, 8: .40, 9: .38, 10: .25, 11: .28, 12: .30}
    rows = []
    for d in pd.date_range(first, last, freq="D"):
        temp = 16.2 + 9.5 * math.sin((d.day_of_year - 110) / 365 * 2 * math.pi) + rng.normal(0, 1.8)
        rain = round(float(rng.exponential(13)), 1) if rng.random() < p_rain[d.month] else 0.0
        rows.append((d.strftime("%Y-%m-%d"), round(temp, 1), rain))
    return pd.DataFrame(rows, columns=["date", "avg_temp", "rain_mm"])


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """표 이름 → 정리된(분석용) DataFrame. 같은 seed 면 항상 같다."""
    rng = np.random.default_rng(seed)
    months = month_list()
    weather = _weather(rng)
    rain_days = (weather.assign(ym=weather["date"].str[:7], r=weather["rain_mm"] >= 1.0)
                 .groupby("ym")["r"].sum().to_dict())

    card, time_rows, dow_rows, stores, closure, pop = [], [], [], [], [], []
    for emd, info in EMD_INFO.items():
        t, size = info["type"], info["size"]
        tf = TOURIST_FACTOR[t]
        for ind in INDUSTRIES:
            b = BASE[ind]
            mult = TYPE_STORE_MULT[t].get(ind, 1.0)
            base_store = b["stores"] * size * mult * float(rng.lognormal(0, 0.12))
            emd_effect = float(rng.lognormal(0, 0.12))
            per_store = b["sales"] * TYPE_SALES_MULT[t] * emd_effect
            for k, ym in enumerate(months):
                mo = int(ym[5:])
                n_store = max(0, int(round(base_store * (1 + 0.004 * k) * float(rng.lognormal(0, 0.03)))))
                season = _season(mo, tf if ind != "편의점" else tf * 0.5)
                rain_eff = 1.0
                rd = rain_days.get(ym, 10)
                if ind == "카페" and t == "tour":
                    rain_eff = 1 - 0.006 * (rd - 10)
                elif ind == "간이음식":
                    rain_eff = 1 + 0.005 * (rd - 10)
                sales = n_store * per_store * season * rain_eff * (1.003 ** k) * float(rng.lognormal(0, 0.05))
                cnt = sales / (b["ticket"] * float(rng.uniform(0.95, 1.05)))
                card.append((ym, emd, ind, int(sales), int(cnt), int(cnt * 0.62)))
                stores.append((ym, emd, ind, n_store))
                opened = int(rng.poisson(n_store * 0.020))
                closed = int(rng.poisson(n_store * (0.017 + (0.006 if t == "tour" else 0.0))))
                closure.append((ym, emd, ind, opened, closed))
                if k < N_MONTHS - 12:          # 시간대·요일 표는 최근 12개월만
                    continue
                share = np.array(TIME_SHARE[ind], dtype=float)
                if t == "tour":
                    share = share * np.array([0.9, 1.1, 1.15, 0.95, 0.85])
                share = share * rng.lognormal(0, 0.05, 5)
                share = share / share.sum()
                for tb, sh in zip(TIME_BANDS, share):
                    time_rows.append((ym, emd, ind, tb, int(sales * sh), int(cnt * sh)))
                dsh = np.array(DOW_SHARE_TOUR if t == "tour" else DOW_SHARE_URBAN, dtype=float)
                dsh = dsh * np.array(DOW_ADJ.get(ind, [1] * 7)) * rng.lognormal(0, 0.04, 7)
                dsh = dsh / dsh.sum() * (1 - HOLIDAY_SHARE)
                for dw, sh in zip(DOWS, list(dsh) + [HOLIDAY_SHARE]):
                    dow_rows.append((ym, emd, ind, dw, int(sales * sh), int(cnt * sh)))
        for k, ym in enumerate(months):
            s = _season(int(ym[5:]), tf)
            for dt in ("평일", "주말"):
                f = (1.2 if t == "tour" else 0.9) if dt == "주말" else 1.0
                pop.append((ym, emd, dt, round(size * 5200 * s * f * float(rng.lognormal(0, 0.04)), 1)))

    card_df = pd.DataFrame(card, columns=["ym", "emd", "industry", "sales_amt", "sales_cnt", "user_cnt"])
    st = pd.DataFrame(stores, columns=["ym", "emd", "industry", "store_cnt"])
    j = card_df.merge(st, on=["ym", "emd", "industry"])
    j = j[(j.emd == "노형동") & (j.industry == "한식")].sort_values("ym").tail(12)
    mine = ((j.sales_amt / j.store_cnt).to_numpy() * 0.92 * rng.lognormal(0, 0.06, len(j))).astype(int)
    my_store = pd.DataFrame({"ym": j.ym.to_numpy(), "sales_amt": (mine // 1000) * 1000})
    st["city"] = st["emd"].map(EMD_CITY)
    store_city = st.groupby(["ym", "city", "industry"], as_index=False)["store_cnt"].sum()
    latest = st[st.ym == months[-1]][["emd", "industry", "store_cnt"]].reset_index(drop=True)
    return {
        "card_emd": card_df,
        "card_time": pd.DataFrame(time_rows, columns=["ym", "emd", "industry", "time_band", "sales_amt", "sales_cnt"]),
        "card_dow": pd.DataFrame(dow_rows, columns=["ym", "emd", "industry", "dow", "sales_amt", "sales_cnt"]),
        "store_count": store_city,
        "store_emd": latest,
        "pop_flow": pd.DataFrame(pop, columns=["ym", "emd", "day_type", "pop_cnt"]),
        "closure": pd.DataFrame(closure, columns=["ym", "emd", "industry", "open_cnt", "close_cnt"]),
        "weather_daily": weather,
        "my_store": my_store,
    }


# ── 센터 CSV 형식으로 펼치기 ─────────────────────────────────────────────
def _dstrct(emd: str) -> str:
    return next((k for k, v in DSTRCT_SE.items() if emd in v), "제주시 중부(동)")


def _ymd(ym: str, d: int) -> str:
    return f"{ym[:4]}{ym[5:7]}{d:02d}"


def _split_legal(rng: np.random.Generator, emd: str) -> list[tuple[str, float]]:
    legal = ADMIN_TO_LEGAL.get(emd)
    if not legal:
        return [(emd, 1.0)]
    w = rng.dirichlet(np.ones(len(legal)) * 3)
    return list(zip(legal, w))


def to_center_format(frames: dict[str, pd.DataFrame], seed: int = SEED) -> dict[str, pd.DataFrame]:
    """분석용 표 → 센터 명세의 열 이름·값 형식을 따르는 CSV 용 표(여러 이용자구분·법정동으로 쪼갬)."""
    rng = np.random.default_rng(seed + 1)
    out = {}

    def common(ym):  # base_year_month, pt_year, pt_month
        return {"base_year_month": ym.replace("-", ""), "pt_year": ym[:4], "pt_month": str(int(ym[5:]))}

    def expand_biz(df, extra_cols):
        rows = []
        for r in df.itertuples(index=False):
            for code, nm, w in BIZ[r.industry]:
                rows.append((r, code, nm, w))
        return rows

    # 카드(읍면동·시간대·요일): 법정동·업종코드·이용자구분(도민/내국인)으로 쪼갬
    def card_like(df: pd.DataFrame, extra_col: str | None, extra_map) -> pd.DataFrame:
        recs = []
        legal_cache: dict[str, list] = {}
        for r in df.itertuples(index=False):
            emd = r.emd
            legal = legal_cache.setdefault(emd, _split_legal(rng, emd))
            info = EMD_INFO[emd]
            dosh = 0.35 if info["type"] == "tour" else 0.65          # 제주도민 비중
            for code, nm, w in BIZ[r.industry]:
                for lg, lw in legal:
                    for ut, tt, us in (("제주도민", "도민", dosh), ("내국인", "개별", 1 - dosh)):
                        f = w * lw * us
                        rec = {**common(r.ym), "dstrct_se": _dstrct(emd)}
                        if extra_col is None:
                            rec.update({"city": EMD_CITY[emd], "area_type": "도시" if info["type"] == "urban" else "읍면"})
                        rec.update({"bjd": lg, "biz_cd": code, "biz_nm": nm, "user_type": ut, "tour_type": tt})
                        if extra_col:
                            rec[extra_col] = extra_map(r)
                        rec.update({"age": "NA", "sex": "알수없음", "user_cnt": int(getattr(r, "user_cnt", r.sales_cnt * 0.62) * f),
                                    "use_cnt": int(r.sales_cnt * f), "use_amt": int(r.sales_amt * f)})
                        recs.append(rec)
        return pd.DataFrame(recs)

    c = frames["card_emd"]
    out["card_emd"] = card_like(c, None, None)
    t = frames["card_time"].copy()
    t["user_cnt"] = (t.sales_cnt * 0.62).astype(int)
    out["card_time"] = card_like(t, "time", lambda r: r.time_band)
    d = frames["card_dow"].copy()
    d["user_cnt"] = (d.sales_cnt * 0.62).astype(int)
    out["card_dow"] = card_like(d, "week", lambda r: r.dow if r.dow == "공휴일" else r.dow + "요일")
    sc = frames["store_count"]
    recs = []
    for r in sc.itertuples(index=False):
        for code, nm, w in BIZ[r.industry]:
            recs.append({**common(r.ym), "city": r.city, "owner_gb": "내국인 개인", "biz_cd": code, "biz_nm": nm,
                         "store_cnt": int(round(r.store_cnt * w))})
    out["store_count"] = pd.DataFrame(recs)
    out["store_emd"] = frames["store_emd"].rename(columns={"emd": "읍면동", "industry": "업종", "store_cnt": "매장수"})
    # 유동인구: 달마다 4일(평일 2·주말 2) × 4시각 표본
    hours = [7, 12, 18, 22]
    hf = {7: 0.8, 12: 1.15, 18: 1.2, 22: 0.85}
    recs = []
    for r in frames["pop_flow"].itertuples(index=False):
        days = [(2, "평일"), (9, "평일")] if r.day_type == "평일" else [(7, "주말"), (14, "주말")]
        # 2일·9일·7일·14일은 달마다 요일이 다르므로 달력으로 평일/주말을 다시 맞춘다
        wd = [dd for dd in range(1, 29) if (pd.Timestamp(f"{r.ym}-{dd:02d}").day_of_week >= 5) == (r.day_type == "주말")]
        pick = [wd[1], wd[len(wd) // 2]]
        for dd in pick:
            for h in hours:
                tot = r.pop_cnt * hf[h]
                recs.append({"base_date": _ymd(r.ym, dd), "pt_year": r.ym[:4], "pt_month": str(int(r.ym[5:])), "time": h,
                             "dstrct_se": _dstrct(r.emd), "emd_nm": f"{EMD_CITY[r.emd]} {r.emd}", "sex": "알수없음", "age": "NA",
                             "resd_pop_cnt": round(tot * 0.55, 2), "work_pop_cnt": round(tot * 0.15, 2),
                             "visit_pop_cnt": round(tot * 0.30, 2), "resd": "제주"})
    out["pop_flow"] = pd.DataFrame(recs)
    cl = frames["closure"].copy()
    cl["ym"] = cl["ym"].str.replace("-", "", regex=False)
    out["closure"] = cl.rename(columns={"ym": "base_year_month", "emd": "읍면동", "industry": "업종", "open_cnt": "개업수", "close_cnt": "폐업수"})
    w = frames["weather_daily"]
    recs = []
    for r in w.itertuples(index=False):
        for code, dt, dr in (("184", 0.0, 1.0), ("189", 0.7, 0.9)):
            recs.append({"base_date": r.date.replace("-", ""), "pt_year": r.date[:4], "pt_month": str(int(r.date[5:7])),
                         "observatory_code": code, "average_temperature": round(r.avg_temp + dt, 1),
                         "daily_rainfall": round(r.rain_mm * dr, 1)})
    out["weather_daily"] = pd.DataFrame(recs)
    out["my_store"] = frames["my_store"].rename(columns={"ym": "기준년월", "sales_amt": "매출"})
    return out


def write_all(out_dir: Path | str = DATA_SYNTHETIC, seed: int = SEED) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = generate(seed)
    paths = []
    center = to_center_format(frames, seed)
    for tid, df in center.items():
        df = df.copy()
        df[LABEL_COL] = LABEL_VALUE
        p = out_dir / TABLES[tid].synthetic_file
        df.to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    mine = center["my_store"].copy()
    mine[LABEL_COL] = LABEL_VALUE
    p = out_dir / "가상_내가게_월매출_cp949.csv"
    mine.to_csv(p, index=False, encoding="cp949")
    paths.append(p)
    return paths
