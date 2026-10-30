import pytest

from jangsa.tools.sql_run import MAX_ROWS, SqlSafetyError, sql_run, validate_sql

ALLOWED = ["card_emd", "store_count"]


@pytest.mark.parametrize("sql", [
    "DROP TABLE card_emd",
    "DELETE FROM card_emd",
    "INSERT INTO card_emd VALUES (1)",
    "UPDATE card_emd SET sales_amt = 0",
    "CREATE TABLE x AS SELECT SUM(sales_amt) FROM card_emd",
    "COPY (SELECT SUM(sales_amt) FROM card_emd) TO 'x.csv'",
    "ATTACH 'x.db'",
    "PRAGMA database_list",
    "SET enable_external_access=true",
    "SELECT SUM(sales_amt) FROM card_emd; DROP TABLE card_emd",
    "SELECT * FROM read_csv('a.csv')",
    "SELECT COUNT(*) FROM read_parquet('a.parquet')",
    "SELECT COUNT(*) FROM information_schema.tables",
    "SELECT COUNT(*) FROM duckdb_tables()",
    "SELECT COUNT(*) FROM secret_table",
    "SELECT emd, sales_amt FROM card_emd",                # 집계 없음 = 원천 행
    "SELECT * FROM card_emd",
    "",
    "EXPLAIN SELECT COUNT(*) FROM card_emd",
])
def test_rejected(sql):
    with pytest.raises(SqlSafetyError):
        validate_sql(sql, ALLOWED)


def test_keywords_inside_string_literals_are_ok():
    clean, used = validate_sql("SELECT COUNT(*) FROM card_emd WHERE industry = 'DROP;DELETE'", ALLOWED)
    assert used == ["card_emd"]


def test_limit_forced_and_capped():
    clean, _ = validate_sql("SELECT industry, SUM(sales_amt) FROM card_emd GROUP BY industry", ALLOWED)
    assert clean.endswith(f"LIMIT {MAX_ROWS}")
    clean, _ = validate_sql("SELECT industry, SUM(sales_amt) FROM card_emd GROUP BY industry LIMIT 99999", ALLOWED)
    assert clean.endswith(f"LIMIT {MAX_ROWS}")
    clean, _ = validate_sql("SELECT industry, SUM(sales_amt) FROM card_emd GROUP BY industry LIMIT 5", ALLOWED)
    assert clean.endswith("LIMIT 5")


def test_cte_allowed_and_comments_stripped():
    sql = "WITH t AS (SELECT emd, SUM(sales_amt) AS s FROM card_emd GROUP BY emd) -- 주석\nSELECT emd, s FROM t"
    clean, used = validate_sql(sql, ALLOWED)
    assert used == ["card_emd"] and "--" not in clean


def test_runs_and_returns_aggregates(store):
    rc = {t: i.rows for t, i in store.tables.items()}
    r = sql_run(store.con, "SELECT industry, SUM(sales_amt) AS s, COUNT(*) AS n_rows FROM card_emd GROUP BY industry", store.available(), rc)
    assert r.rows == 9 and r.source_rows == rc["card_emd"]


def test_connection_cannot_read_files(store, syn_dir):
    import duckdb
    with pytest.raises(duckdb.Error):
        store.con.execute(f"SELECT * FROM read_csv('{syn_dir}/가상_일일날씨.csv')")


def test_unknown_table_and_dml_do_not_change_data(store):
    n = store.con.execute("SELECT COUNT(*) FROM card_emd").fetchone()[0]
    with pytest.raises(SqlSafetyError):
        sql_run(store.con, "DELETE FROM card_emd", store.available())
    assert store.con.execute("SELECT COUNT(*) FROM card_emd").fetchone()[0] == n
