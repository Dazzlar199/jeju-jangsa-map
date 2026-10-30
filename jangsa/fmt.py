"""표·열 이름을 도민이 읽기 쉬운 모양으로 바꾼다(화면·보고서 공통)."""
from __future__ import annotations

import pandas as pd

COL_KO = {
    "emd": "읍면동", "ym": "연월", "time_band": "시간대", "dow": "요일", "rain_group": "비 온 날 구분",
    "sales_per_store": "매장 한 곳당 월 매출(원)", "avg_store_cnt": "평균 매장 수(곳)", "sales_cnt": "이용 건수",
    "sales_amt": "이용 금액(원)", "avg_daily_pop": "하루 평균 유동인구(명)", "stores_per_10k_pop": "유동인구 1만 명당 매장 수",
    "open_cnt": "개업 수", "close_cnt": "폐업 수", "close_rate_pct": "폐업 비율(%)", "sales_share_pct": "매출 비중(%)",
    "avg_monthly_riders": "월 평균 버스 승차(명)", "store_cnt": "매장 수(곳)", "avg_rain_diff": "평소 대비 비 온 날 수(일)",
    "avg_sales_idx": "매출 지수(같은 달 평균=100)",
}


def pretty_df(df: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    d = df.head(max_rows) if max_rows else df
    out = pd.DataFrame(index=d.index)
    for c in d.columns:
        col = d[c]
        name = COL_KO.get(c, c)
        if pd.api.types.is_numeric_dtype(col) or col.map(lambda v: isinstance(v, (int, float))).any():
            def f(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return "비공개"
                if isinstance(v, str):
                    return v
                if float(v).is_integer():
                    return f"{int(v):,}"
                return f"{v:,.2f}".rstrip("0").rstrip(".")
            out[name] = col.map(f)
        else:
            out[name] = col.map(lambda v: "비공개" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
    return out


ORIGIN_KO = {"synthetic": "가상 데이터", "center": "센터 CSV", "public": "공개 실데이터", "user": "내 자료"}
INTENT_KO = {"location": "입지 진단", "competition": "업종 경쟁", "time_pattern": "시간대·요일", "season_weather": "계절·날씨",
             "benchmark": "내 가게 비교"}
