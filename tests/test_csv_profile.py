import io

import pandas as pd
import pytest

from jangsa.tools.csv_profile import profile_csv, to_ym


def _csv(df, enc="utf-8-sig"):
    return df.to_csv(index=False).encode(enc)


def test_ym_formats():
    assert [to_ym(x) for x in ("201701", "2017-1", "2017.01", "2017-01-15", 202601)] == ["2017-01", "2017-01", "2017-01", "2017-01", "2026-01"]
    assert to_ym("abc") is None and to_ym("201713") is None


def test_cp949_and_korean_headers_auto_detected():
    df = pd.DataFrame({"기준년월": ["202509", "202510"], "읍면동": ["애월읍", "제주시 노형동"], "업종": ["카페", "한식"],
                       "이용금액": ["1,000,000", "2,000,000"], "이용건수": [100, 200]})
    r = profile_csv(_csv(df, "cp949"), "a.csv")
    assert r.encoding == "cp949" and r.table_id == "card_emd" and r.ok
    assert set(r.df["emd"]) == {"애월읍", "노형동"} and r.df["sales_amt"].tolist() == [1000000, 2000000]


def test_center_english_columns_are_aggregated_and_mapped():
    rows = []
    for ut in ("제주도민", "내국인"):
        for bjd, biz in (("일도일동", "한식 음식점업"), ("일도이동", "한식 음식점업"), ("노형동", "비알콜 음료점업")):
            rows.append(dict(base_year_month="202510", pt_year="2025", pt_month="10", dstrct_se="x", city="제주시", area_type="도시",
                             bjd=bjd, biz_cd="I1", biz_nm=biz, user_type=ut, tour_type="개별", age="NA", sex="알수없음",
                             user_cnt=3, use_cnt=4, use_amt=1000))
    r = profile_csv(_csv(pd.DataFrame(rows)), "c.csv")
    assert r.ok and r.table_id == "card_emd"
    d = r.df.set_index(["emd", "industry"])
    assert d.loc[("일도1동", "한식"), "sales_amt"] == 2000          # 이용자구분 합산
    assert d.loc[("일도2동", "한식"), "sales_cnt"] == 8
    assert d.loc[("노형동", "카페"), "sales_amt"] == 2000          # 업종명 → 앱 업종
    assert r.mapping["sales_amt"] == "use_amt" and r.mapping["emd"] == "bjd"


def test_unknown_industries_dropped_with_warning_and_unknown_emd_error():
    df = pd.DataFrame({"base_year_month": ["202510"] * 3, "bjd": ["노형동", "노형동", "없는동"], "biz_nm": ["한식 음식점업", "호텔업", "한식 음식점업"],
                       "use_amt": [1, 2, 3], "use_cnt": [1, 1, 1]})
    r = profile_csv(_csv(df), "c.csv", table_id="card_emd")
    assert r.ok and len(r.df) == 1 and any("호텔업" in w for w in r.warnings) and any("없는동" in w for w in r.warnings)
    df2 = df.assign(bjd=["가", "나", "다"])
    r2 = profile_csv(_csv(df2), "c.csv", table_id="card_emd")
    assert not r2.ok and r2.errors


def test_raw_company_rows_are_refused():
    df = pd.DataFrame({"base_date": ["20210507"], "kedcd": ["1"], "ent_nm": ["광명종합건축"], "biz_num": ["6162688529"], "closed_biz": ["여"]})
    r = profile_csv(_csv(df), "closure.csv")
    assert not r.ok and "원본 행 표" in r.errors[0]
    r = profile_csv(_csv(df), "closure.csv", table_id="closure")
    assert not r.ok


def test_pop_flow_collapses_dates_hours_sex_age():
    rows = []
    for date in ("20251006", "20251007", "20251011"):          # 월·화·토
        for hour in (7, 12):
            for sex in ("남성", "여성"):
                rows.append(dict(base_date=date, time=hour, emd_nm="제주시 한림읍", sex=sex, age="20", resd_pop_cnt=10, work_pop_cnt=5, visit_pop_cnt=5))
    r = profile_csv(_csv(pd.DataFrame(rows)), "p.csv")
    assert r.ok and r.table_id == "pop_flow"
    d = r.df.set_index("day_type")
    assert d.loc["평일", "pop_cnt"] == 40 and d.loc["주말", "pop_cnt"] == 40       # 성별 합 20+20, 시간 평균
    assert r.df["ym"].iloc[0] == "2025-10"


def test_missing_required_and_manual_mapping():
    df = pd.DataFrame({"월": ["202510"], "무슨동": ["노형동"], "업태": ["한식"], "금액": [1], "건수": [1]})
    r = profile_csv(_csv(df), "x.csv", table_id="card_emd")
    assert not r.ok and "emd" in r.missing_required or "sales_amt" in r.missing_required
    r2 = profile_csv(_csv(df), "x.csv", table_id="card_emd", manual_mapping={"emd": "무슨동", "industry": "업태", "sales_amt": "금액", "sales_cnt": "건수", "ym": "월"})
    assert r2.ok


def test_empty_and_garbage_files():
    assert not profile_csv(b"", "e.csv").ok
    assert not profile_csv(b"\xff\xfe\x00\x01garbage\x80\x81", "g.csv").ok
