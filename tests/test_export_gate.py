import copy

import pandas as pd

from jangsa.config import export_rules
from jangsa.tools.export_gate import export_gate


def _df():
    return pd.DataFrame({"emd": ["a", "b", "c"], "sales_amt": [1e9, 5e8, 2e8], "store_cnt": [50, 8, 2.0],
                         "sales_idx": [150, 90, 50.0]})


def test_small_cells_deleted_below_ten():
    r = export_gate(_df(), ["card_emd"], source_rows=1000)
    assert r.passed
    assert r.df.loc[0, "store_cnt"] == 50
    assert pd.isna(r.df.loc[1, "store_cnt"]) and pd.isna(r.df.loc[1, "sales_amt"])      # 10 미만 → 단순집계 삭제
    assert r.df.loc[1, "sales_idx"] == 90                                                # 응용집계는 수치 유지(근거 3 이상)
    assert pd.isna(r.df.loc[2, "sales_idx"])                                             # 근거 3 미만 → 자체 강화로 가림
    assert r.suppressed_rows == 2


def test_category_and_mask_methods():
    rules = copy.deepcopy(export_rules())
    rules["small_cell_method"] = "category"
    r = export_gate(_df(), ["card_emd"], 1000, rules=rules)
    assert r.df.loc[1, "store_cnt"] == "10 미만"
    rules["small_cell_method"] = "mask"
    r = export_gate(_df(), ["card_emd"], 1000, rules=rules)
    assert r.df.loc[1, "store_cnt"] == rules["mask_value"]


def test_threshold_is_configurable():
    rules = copy.deepcopy(export_rules())
    rules["min_count"] = 5
    r = export_gate(_df(), ["card_emd"], 1000, rules=rules)
    assert r.df.loc[1, "store_cnt"] == 8          # 8 은 5 이상이라 보임


def test_strict_mode_drops_simple_aggregates_for_export():
    r = export_gate(_df(), ["card_emd"], 1000)
    assert list(r.df_export.columns) == ["emd", "sales_idx"]
    assert set(r.dropped_for_export) == {"sales_amt", "store_cnt"}
    rules = copy.deepcopy(export_rules())
    rules["strict_mode"] = False
    r2 = export_gate(_df(), ["card_emd"], 1000, rules=rules)
    assert "sales_amt" in r2.df_export.columns and pd.isna(r2.df_export.loc[1, "sales_amt"])


def test_simple_aggregate_without_count_column_is_blocked():
    r = export_gate(pd.DataFrame({"emd": ["a", "b"], "sales_amt": [1.0, 2.0]}), ["card_emd"], 100)
    assert not r.passed and "근거 건수" in r.blocked_reasons[0]


def test_raw_rows_blocked():
    df = pd.DataFrame({"emd": [f"e{i}" for i in range(90)], "sales_cnt": [100] * 90, "sales_idx": [100.0] * 90})
    r = export_gate(df, ["card_emd"], source_rows=100)
    assert not r.passed and "원본 일부 추출" in r.blocked_reasons[0]
    big = pd.DataFrame({"emd": range(300), "sales_cnt": 50, "sales_idx": 1.0})
    assert not export_gate(big, ["card_emd"], source_rows=10**7).passed      # 최대 행 수 초과


def test_identifier_columns_blocked():
    for col in ("store_id", "card_no", "격자id", "lat"):
        df = pd.DataFrame({col: ["x"], "sales_cnt": [100], "sales_idx": [1.0]})
        assert not export_gate(df, ["card_emd"], 1000).passed, col
    ok = pd.DataFrame({"store_idx": [100.0], "sales_cnt": [100]})                  # store_idx 는 식별 열이 아님
    assert export_gate(ok, ["card_emd"], 1000).passed


def test_open_tables_not_suppressed_and_csv_allowed():
    df = pd.DataFrame({"time_band": ["오전"], "sales_amt": [5.0], "sales_share_pct": [10.0]})
    r = export_gate(df, ["bc_tour_time"], 100)
    assert r.passed and r.df_export.equals(r.df) and "csv" in r.allowed_formats
    r2 = export_gate(df.assign(sales_cnt=3), ["card_emd"], 1000)
    assert "csv" not in r2.allowed_formats and set(r2.allowed_formats) == {"png", "report"}


def test_all_hidden_is_blocked():
    df = pd.DataFrame({"emd": ["a"], "sales_amt": [5.0], "store_cnt": [1]})
    assert not export_gate(df, ["card_emd"], 1000).passed
