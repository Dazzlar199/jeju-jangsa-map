"""도구 2. sql_run: 계획이 만든 SQL 을 DuckDB 에서 안전하게 실행한다.

안전장치(겹겹이):
 1) 한 문장만, SELECT 또는 WITH 로 시작
 2) 쓰기·파일·확장·설정 명령과 파일 읽기 함수 차단(문자열 리터럴 안은 검사에서 제외)
 3) DuckDB 자체 파서(json_serialize_sql)로 SELECT 문인지 한 번 더 확인
 4) 허용된 표(읽어 둔 표)만 사용
 5) 집계 강제: GROUP BY 또는 집계 함수가 없으면 거절(원천 행 조회 차단)
 6) LIMIT 강제: 없으면 붙이고, 상한보다 크면 낮춘다
 7) 연결 자체가 enable_external_access=false (파일·네트워크 접근 불가)
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd

MAX_ROWS = 500

_FORBIDDEN_WORDS = (
    "INSERT UPDATE DELETE DROP ALTER CREATE ATTACH DETACH COPY EXPORT IMPORT PRAGMA SET RESET INSTALL LOAD CALL "
    "TRUNCATE REPLACE VACUUM CHECKPOINT EXECUTE PREPARE USE FORCE DESCRIBE SHOW SUMMARIZE PIVOT UNPIVOT MERGE GRANT "
    "TRANSACTION BEGIN COMMIT ROLLBACK"
).split()
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_WORDS) + r")\b", re.I)
_FORBIDDEN_FUNC_RE = re.compile(
    r"(\bread_\w+|\bwrite_\w+|\bglob\s*\(|\bduckdb_\w+|\bpragma_\w+|\bquery\s*\(|\bquery_table\s*\(|\bgetenv\s*\(|"
    r"\bcurrent_setting\s*\(|\bsniff_csv|\bparquet_\w+|\bjson_serialize|\bhttp\w*\(|information_schema|\bpg_\w+|"
    r"\bsqlite_\w+|\bfile_\w+|\bcheckpoint\b|\bglob\b)", re.I)
_AGG_RE = re.compile(
    r"(\bGROUP\s+BY\b|\b(COUNT|SUM|AVG|MIN|MAX|MEDIAN|MODE|QUANTILE\w*|STDDEV\w*|VAR_\w+|PERCENT_RANK|"
    r"STRING_AGG|ARG_MAX|ARG_MIN|ANY_VALUE|LIST|ARRAY_AGG|CORR|REGR_\w+|FIRST|LAST|BOOL_OR|BOOL_AND)\s*\()", re.I)
_STR_RE = re.compile(r"'(?:[^']|'')*'")
_COMMENT_RE = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.S)
_FROM_RE = re.compile(r"\b(?:FROM|JOIN)\s+(\"?[A-Za-z_][\w]*\"?)", re.I)
_CTE_RE = re.compile(r"(?:\bWITH\b|,)\s*(\"?[A-Za-z_]\w*\"?)\s+AS\s*\(", re.I)
_LIMIT_END_RE = re.compile(r"\bLIMIT\s+(\d+)\s*$", re.I)


class SqlSafetyError(Exception):
    """안전 규칙 위반(재시도해도 같은 문장이면 소용없다)."""


@dataclass
class SqlResult:
    sql: str
    df: pd.DataFrame
    tables_used: list[str]
    rows: int
    source_rows: int
    elapsed_ms: int
    truncated: bool = False


def _blank_strings(sql: str) -> str:
    return _STR_RE.sub("''", sql)


def validate_sql(sql: str, allowed_tables: list[str], max_rows: int = MAX_ROWS) -> tuple[str, list[str]]:
    """안전 검사를 하고 (정리된 SQL, 사용한 표 목록)을 돌려준다. 위반이면 SqlSafetyError."""
    s = _COMMENT_RE.sub(" ", sql or "").strip()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(";").strip()
    if not s:
        raise SqlSafetyError("빈 SQL 입니다.")
    probe = _blank_strings(s)
    if ";" in probe:
        raise SqlSafetyError("한 번에 한 문장만 실행할 수 있습니다.")
    if not re.match(r"^(SELECT|WITH)\b", probe, re.I):
        raise SqlSafetyError("SELECT(조회) 문만 실행할 수 있습니다.")
    m = _FORBIDDEN_RE.search(probe)
    if m:
        raise SqlSafetyError(f"허용되지 않는 명령입니다: {m.group(1).upper()}")
    m = _FORBIDDEN_FUNC_RE.search(probe)
    if m:
        raise SqlSafetyError(f"허용되지 않는 함수·대상입니다: {m.group(1)}")
    ctes = {c.strip('"').lower() for c in _CTE_RE.findall(probe)}
    used = []
    for t in _FROM_RE.findall(probe):
        t = t.strip('"')
        if t.lower() in ctes:
            continue
        if t not in allowed_tables:
            raise SqlSafetyError(f"사용할 수 없는 표입니다: {t} (허용: {', '.join(allowed_tables)})")
        if t not in used:
            used.append(t)
    if not used:
        raise SqlSafetyError("읽어 둔 표를 FROM 으로 지정해야 합니다.")
    if not _AGG_RE.search(probe):
        raise SqlSafetyError("집계(GROUP BY 또는 SUM·COUNT·AVG 등)가 없는 조회는 원천 행을 그대로 꺼내는 것이라 막습니다.")
    m = _LIMIT_END_RE.search(s)
    if m:
        if int(m.group(1)) > max_rows:
            s = _LIMIT_END_RE.sub(f"LIMIT {max_rows}", s)
    else:
        s = f"{s} LIMIT {max_rows}"
    return s, used


def sql_run(con: duckdb.DuckDBPyConnection, sql: str, allowed_tables: list[str], row_counts: dict[str, int] | None = None,
            max_rows: int = MAX_ROWS) -> SqlResult:
    clean, used = validate_sql(sql, allowed_tables, max_rows)
    try:  # 2중 확인: DuckDB 파서가 SELECT 로 인식하는지
        j = json.loads(con.execute("SELECT json_serialize_sql(?)", [clean]).fetchone()[0])
    except duckdb.Error as e:
        raise SqlSafetyError(f"SQL 을 확인하지 못했습니다: {e}") from e
    if j.get("error"):
        raise SqlSafetyError(str(j.get("error_message", "SELECT 문이 아닙니다.")))
    t0 = time.time()
    df = con.execute(clean).df()
    src = max([(row_counts or {}).get(t, 0) for t in used] or [0])
    return SqlResult(clean, df, used, len(df), src, int((time.time() - t0) * 1000), truncated=len(df) >= max_rows)
