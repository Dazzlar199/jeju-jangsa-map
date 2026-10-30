"""도구 3. export_gate: 반출 기준 검사기.

센터 밖(외부 LLM API 포함)으로 나갈 수 있는 값은 이 검사를 통과한 집계표뿐이다.
기준은 config/export_rules.yaml (제주빅데이터센터 데이터 반출 정책 2026.09.01 개정 기준 초안).

규칙
 R1 원천 행 차단 : 결과 행이 원천 표 행 수의 일정 비율을 넘거나 최대 행 수를 넘으면 차단
 R2 식별 열 차단 : 격자·좌표·카드번호·가맹점명 같은 열이 결과에 있으면 차단
 R3 근거 건수    : 단순집계(합계·평균 등) 값이 있는데 근거 건수(…_cnt 등)가 없으면 확인 불가로 차단
 R4 소수 셀 억제 : 근거 건수가 10 미만인 행의 단순집계는 삭제·범주·비식별 처리(기본 삭제)
 R5 응용집계      : 비율·순위·지수는 수치 포함 가능, 단 근거가 applied_min_support 미만이면 가림(자체 강화)
 R6 반출 형식     : 표 종류(restricted/open/user)에 따라 허용 형식(png, report, csv)을 알려 준다
 R7 엄격 모드     : strict_mode(기본 켬)이면 원본 불가 표의 단순집계 열은 df_export(LLM·보고서·PNG)에서 뺀다
 (open = 버스·날씨 같은 반출 가능 표와 공개 실데이터: R3~R5 를 적용하지 않고 df_export == df)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..config import export_rules
from ..schema import TABLES

SUPPRESSED = "비공개"


@dataclass
class GateResult:
    passed: bool
    df: pd.DataFrame | None                      # 센터 안 화면에 보여 줄 수 있는 (소수 셀 처리된) 표
    df_export: pd.DataFrame | None = None        # LLM·보고서·PNG 로 나갈 수 있는 표(엄격 모드면 응용집계만)
    level: str = "restricted"
    dropped_for_export: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    rows_in: int = 0
    rows_out: int = 0
    suppressed_rows: int = 0
    suppressed_cells: int = 0
    allowed_formats: list[str] = field(default_factory=list)
    column_kinds: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.passed:
            return "차단: " + "; ".join(self.blocked_reasons)
        extra = f" (소수 셀 {self.suppressed_rows}행 처리)" if self.suppressed_rows else ""
        return f"통과{extra}"


def classify_columns(df: pd.DataFrame, rules: dict | None = None, applied: list[str] | None = None) -> dict[str, str]:
    """결과 열 → dim(기준·이름) | count(근거 건수) | applied(응용집계) | simple(단순집계)."""
    rules = rules or export_rules()
    kinds = {}
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or c in ("ym",):
            kinds[c] = "dim"
            continue
        if applied and c in applied:
            kinds[c] = "applied"
        elif any(re.search(p, c, re.I) for p in rules["column_kinds"]["applied"]):
            kinds[c] = "applied"
        elif any(re.search(p, c, re.I) for p in rules["column_kinds"]["count"]):
            kinds[c] = "count"
        else:
            kinds[c] = "simple"
    return kinds


def _export_level(tables_used: list[str]) -> str:
    levels = {TABLES[t].export_level for t in tables_used if t in TABLES}
    if "restricted" in levels:
        return "restricted"
    if "user" in levels and levels == {"user"}:
        return "user"
    return "open" if levels else "restricted"


def export_gate(df: pd.DataFrame, tables_used: list[str], source_rows: int = 0, applied: list[str] | None = None,
                rules: dict | None = None, extra_dims: list[str] | None = None) -> GateResult:
    """집계 결과 한 장을 검사한다. rules 를 안 주면 config/export_rules.yaml 을 쓴다."""
    rules = rules or export_rules()
    level = _export_level(tables_used)
    res = GateResult(passed=False, df=None, rows_in=len(df), allowed_formats=list(rules["allowed_formats"][level]), level=level)
    kinds = classify_columns(df, rules, applied)
    for d in extra_dims or []:
        kinds[d] = "dim"
    res.column_kinds = kinds

    # R2 식별 열
    for c in df.columns:
        if any(re.search(p, str(c), re.I) for p in rules["blocked_column_patterns"]):
            res.blocked_reasons.append(f"식별·세부 위치 열 '{c}' 은(는) 반출할 수 없습니다.")
    # R1 원천 행
    if len(df) > rules["max_result_rows"]:
        res.blocked_reasons.append(f"결과가 {len(df)}행으로 한 번에 반출 가능한 {rules['max_result_rows']}행을 넘습니다.")
    if source_rows and len(df) > 1 and len(df) / source_rows > rules["raw_row_ratio_limit"]:
        res.blocked_reasons.append("결과 행 수가 원천 표와 거의 같아 '원본 일부 추출'로 볼 수 있어 차단합니다.")
    if df.empty:
        res.blocked_reasons.append("결과가 비어 있습니다.")

    count_cols = [c for c, k in kinds.items() if k == "count"]
    simple_cols = [c for c, k in kinds.items() if k == "simple"]
    applied_cols = [c for c, k in kinds.items() if k == "applied"]
    enforce = level in rules["suppress_levels"]
    # R3 근거 건수
    if enforce and simple_cols and not count_cols:
        res.blocked_reasons.append("합계·평균 같은 단순집계가 있는데 근거 건수(…_cnt, n_rows) 열이 없어 10 미만 여부를 확인할 수 없습니다.")
    if enforce and applied_cols and not count_cols and not simple_cols:
        res.notes.append("응용집계(비율·순위)만 있고 근거 건수 열이 없어 소수 셀 확인을 하지 못했습니다.")

    if res.blocked_reasons:
        return res

    out = df.copy()
    if not enforce:
        res.df = res.df_export = out
        res.rows_out = len(out)
        res.passed = True
        res.notes.append("반출 가능 표·공개 실데이터라 10 미만 억제를 적용하지 않았습니다.")
        return res
    min_n = rules["min_count"]
    method = rules["small_cell_method"]
    gs_names = set(rules["group_size_columns"])
    group_cols = [c for c in count_cols if c in gs_names] or count_cols
    event_cols = [c for c in count_cols if c not in group_cols]
    if group_cols:
        support = out[group_cols].min(axis=1)
    else:
        support = pd.Series([float("inf")] * len(out), index=out.index)
    small = support < min_n                       # 근거 집단이 10 미만인 행
    app_small = support < rules["applied_min_support"]
    to_obj = simple_cols + count_cols + applied_cols
    if to_obj:
        out = out.astype({c: "object" for c in to_obj})

    def _hide(mask: pd.Series, cols: list[str]) -> int:
        n = 0
        for c in cols:
            if method == "delete":
                val = None
            elif method == "category":
                val = f"{min_n} 미만"
            else:
                val = rules["mask_value"]
            n += int((mask & out[c].notna()).sum())
            out.loc[mask, c] = val
        return n

    # R4 단순집계 억제: 근거 집단이 작은 행은 단순집계·개수 열 전체, 그 밖에는 10 미만인 개수 칸만
    res.suppressed_cells += _hide(small, simple_cols + group_cols)
    for c in event_cols:
        v = pd.to_numeric(out[c], errors="coerce")
        res.suppressed_cells += _hide(v < min_n, [c])
        small = small | (v < min_n)
    # R5 응용집계 자체 강화
    if applied_cols:
        res.suppressed_cells += _hide(app_small, applied_cols)
    res.suppressed_rows = int((small | app_small).sum())
    if res.suppressed_rows:
        res.issues.append(f"근거 건수 {min_n} 미만이라 {res.suppressed_rows}행의 수치를 '{SUPPRESSED}'로 처리했습니다"
                          f"({ {'delete': '삭제처리', 'category': '범주처리', 'mask': '비식별처리'}[method] }).")
    # 모든 수치가 가려졌으면 쓸모없으므로 알림
    num_cols = simple_cols + count_cols + applied_cols
    if num_cols and out[num_cols].isna().all().all():
        res.blocked_reasons.append("모든 수치가 소수 셀 기준으로 가려져 보여 줄 값이 없습니다.")
        return res
    res.df = out
    res.rows_out = len(out)
    if rules.get("strict_mode", True) and level == "restricted":
        drop = simple_cols + count_cols
        res.df_export = out.drop(columns=drop)
        res.dropped_for_export = drop
        if not applied_cols or res.df_export[applied_cols].isna().all().all():
            res.notes.append("엄격 모드: 원본 불가 표의 단순집계는 반출용 표에서 빠지고 응용집계(지수·비율·순위)가 없어 반출용 표가 비었습니다.")
            res.df_export = None
    else:
        res.df_export = out
    res.passed = True
    return res
