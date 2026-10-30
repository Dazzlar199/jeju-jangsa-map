"""도구 1. csv_profile: 올린 CSV 를 표 명세와 맞춘다.

- 인코딩 자동 판별(UTF-8 / UTF-8 BOM / CP949), 구분자(, ; 탭) 자동 판별
- 열 자동 매핑: 이름 정규화 → 별칭 일치 → 포함 → 철자 유사도. 못 찾은 열은 사람이 고를 수 있게 돌려준다.
- 값 정리: 연월(202410, 2024.10, 2024-10-15 → 2024-10), 숫자(쉼표 제거), 읍면동(제주시 애월읍 → 애월읍), 업종 동의어
- 검증: 필수 열, 숫자 변환 실패 비율, 모르는 읍면동·업종 비율, 기간
"""
from __future__ import annotations

import csv
import difflib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..schema import (CODE_TO_INDUSTRY, DAY_TYPES, DOWS, EMD_CITY, INDUSTRIES, INDUSTRY_WORDS, TABLES, TIME_BANDS, TableSpec,
                      normalize_emd, to_admin)

LABEL_COL = "데이터구분"
# 개별 기업·사업자를 알아볼 수 있는 원본 행 표의 열. 이 열이 있으면 올리지 않는다(반출 불가 원본·식별 정보).
BLOCKED_RAW_COLUMNS = {"biz_num", "kedcd", "ent_nm", "사업자번호", "기업명", "한국기업데이터코드", "card_no", "카드번호"}
ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


class CsvError(Exception):
    pass


@dataclass
class ProfileResult:
    table_id: str | None
    file_name: str
    encoding: str
    rows: int
    mapping: dict[str, str] = field(default_factory=dict)      # 내부 열 → 원본 열
    scores: dict[str, float] = field(default_factory=dict)
    missing_required: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    unmapped_source_columns: list[str] = field(default_factory=list)
    rows_in: int = 0
    period: tuple[str, str] | None = None
    data_kind: str = "center"                                   # center | synthetic
    df: pd.DataFrame | None = None                              # 정리된 데이터(내부 열 이름)

    @property
    def ok(self) -> bool:
        return self.table_id is not None and not self.missing_required and not self.errors and self.df is not None


# ── 읽기 ────────────────────────────────────────────────────────────────
def decode_bytes(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise CsvError("글자 깨짐: UTF-8 도 CP949 도 아닙니다. 엑셀에서 'CSV UTF-8'로 다시 저장해 주세요.")


def read_csv_auto(source: bytes | str | Path) -> tuple[pd.DataFrame, str]:
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    else:
        raw = bytes(source)
    if not raw.strip():
        raise CsvError("빈 파일입니다.")
    text, enc = decode_bytes(raw)
    head = text[:4096]
    try:
        sep = csv.Sniffer().sniff(head.split("\n", 1)[0] + "\n", delimiters=",;\t").delimiter
    except csv.Error:
        sep = ","
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    return df, enc


# ── 열 이름 맞추기 ──────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-\(\)\[\]/·.]", "", str(s)).lower()


def _score(src: str, aliases: list[str]) -> float:
    n = _norm(src)
    best = 0.0
    for a in aliases:
        an = _norm(a)
        if n == an:
            return 1.0
        if len(an) >= 2 and len(n) >= 2 and (an in n or n in an):
            best = max(best, 0.85)
        else:
            best = max(best, difflib.SequenceMatcher(None, n, an).ratio() * 0.9)
    return best


def match_columns(src_columns: list[str], spec: TableSpec, threshold: float = 0.75
                  ) -> tuple[dict[str, str], dict[str, float]]:
    pairs = []
    for c in spec.read_columns:
        names = [c.name, c.desc.split("(")[0]] + list(c.aliases)
        for s in src_columns:
            sc = _score(s, names)
            if sc >= threshold:
                pairs.append((sc, c.name, s))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    mapping, scores, used = {}, {}, set()
    for sc, cname, s in pairs:
        if cname in mapping or s in used:
            continue
        mapping[cname] = s
        scores[cname] = round(sc, 2)
        used.add(s)
    return mapping, scores


def detect_table(src_columns: list[str], candidates: list[str] | None = None) -> tuple[str | None, float]:
    best, best_key = None, (-1.0, -1)
    for tid in candidates or [t for t, sp in TABLES.items() if sp.synthetic_file]:
        spec = TABLES[tid]
        mapping, _ = match_columns(src_columns, spec)
        req = [c.name for c in spec.read_columns if c.required]
        frac = sum(1 for r in req if r in mapping) / len(req)
        key = (frac, len(mapping))
        if key > best_key:
            best, best_key = tid, key
    if best is None or best_key[0] < 0.99:
        return (best if best_key[0] >= 0.6 else None), max(best_key[0], 0.0)
    return best, best_key[0]


# ── 값 정리 ─────────────────────────────────────────────────────────────
def to_ym(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    m = re.search(r"((?:19|20)\d{2})\D{0,2}(\d{1,2})", str(v).strip())
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return f"{y:04d}-{mo:02d}" if 1 <= mo <= 12 else None


def to_date(v) -> str | None:
    s = str(v).strip()
    m = re.match(r"^((?:19|20)\d{2})\D?(\d{1,2})\D?(\d{1,2})", s)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return pd.Timestamp(year=y, month=mo, day=d).strftime("%Y-%m-%d")
    except ValueError:
        return None


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[,\s원명건㎜mm℃]", "", regex=True), errors="coerce")


_IND_LOOKUP = {}
for _k, _words in INDUSTRY_WORDS.items():
    _IND_LOOKUP[_norm(_k)] = _k
    for _w in _words:
        _IND_LOOKUP[_norm(_w)] = _k
_DOW_LOOKUP = {**{d: d for d in DOWS}, **{d + "요일": d for d in DOWS if d != "공휴일"},
               **dict(zip(["mon", "tue", "wed", "thu", "fri", "sat", "sun"], DOWS))}
_TB_LOOKUP = {tb: tb for tb in TIME_BANDS}


def norm_industry(v) -> str:
    """업종 코드(I56220)나 업종명(비알콜 음료점업)을 이 앱의 업종 이름으로. 모르는 업종은 그대로 둔다."""
    s = str(v).strip()
    if s.upper() in CODE_TO_INDUSTRY:
        return CODE_TO_INDUSTRY[s.upper()]
    n = _norm(s)
    if n in _IND_LOOKUP:
        return _IND_LOOKUP[n]
    for key, canon in _IND_LOOKUP.items():
        if len(key) >= 2 and key in n:
            return canon
    return s


def norm_time_band(v) -> str:
    s = str(v).strip()
    if s in _TB_LOOKUP:
        return s
    for tb in TIME_BANDS:
        if tb in s:
            return tb
    return s


def norm_dow(v) -> str:
    return _DOW_LOOKUP.get(str(v).strip().lower(), str(v).strip())


def norm_day_type(v) -> str:
    s = str(v).strip()
    if s in ("주중", "평일", "weekday"):
        return "평일"
    if s in ("주말", "휴일", "공휴일", "weekend", "주말·공휴일"):
        return "주말"
    return s


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str], spec: TableSpec, warnings: list[str],
                  errors: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in spec.read_columns:
        if c.name not in mapping:
            continue
        raw = df[mapping[c.name]]
        if c.type == "ym":
            out[c.name] = raw.map(to_ym)
        elif c.type == "date":
            out[c.name] = raw.map(to_date)
        elif c.type == "number":
            out[c.name] = to_number(raw)
        elif c.type == "emd":
            out[c.name] = raw.map(lambda v: to_admin(v) if pd.notna(v) else None)
        else:
            if c.name == "industry":
                out[c.name] = raw.map(norm_industry)
            elif c.name == "time_band":
                out[c.name] = raw.map(norm_time_band)
            elif c.name == "dow":
                out[c.name] = raw.map(norm_dow)
            elif c.name == "day_type":
                out[c.name] = raw.map(norm_day_type)
            else:
                out[c.name] = raw.astype(str).str.strip()
        bad = out[c.name].isna().mean() if len(out) else 0
        if bad > 0:
            msg = f"'{mapping[c.name]}' 열에서 {bad:.0%}가 {c.type} 형식으로 읽히지 않아 빈 값 처리했습니다."
            (errors if (c.required and bad > 0.5) else warnings).append(msg)
    return out


def validate(out: pd.DataFrame, spec: TableSpec, warnings: list[str], errors: list[str]) -> pd.DataFrame:
    """값 검증. 모르는 읍면동·업종 행은 빼고 알린다(전부 모르면 오류)."""
    if "emd" in out.columns:
        ok = out["emd"].isin(EMD_CITY)
        if not ok.all():
            unknown = sorted(out.loc[~ok, "emd"].dropna().unique())
            if ok.mean() < 0.5:
                errors.append(f"제주 읍면동으로 알아볼 수 있는 행이 {ok.mean():.0%}뿐입니다(예: {', '.join(map(str, unknown[:5]))}). "
                              "법정동→행정동 매핑(config/tables.yaml legal_to_admin)을 확인하세요.")
            else:
                warnings.append(f"읍면동으로 매핑하지 못한 {len(unknown)}곳({', '.join(map(str, unknown[:6]))}{'…' if len(unknown) > 6 else ''}) "
                                f"{int((~ok).sum()):,}행은 분석에서 뺐습니다.")
            out = out[ok]
    if "industry" in out.columns:
        ok = out["industry"].isin(INDUSTRIES)
        if not ok.all():
            unknown = sorted(out.loc[~ok, "industry"].dropna().unique())
            warnings.append(f"이 앱이 다루는 업종 {len(INDUSTRIES)}개 밖의 업종 {len(unknown)}개({', '.join(map(str, unknown[:4]))}…) "
                            f"{int((~ok).sum()):,}행은 뺐습니다.")
            out = out[ok]
    for col, allowed in (("time_band", TIME_BANDS), ("dow", DOWS), ("day_type", DAY_TYPES)):
        if col in out.columns:
            bad = ~out[col].isin(allowed)
            if bad.any():
                warnings.append(f"'{col}' 값 {list(out.loc[bad, col].dropna().unique()[:4])} 은(는) 예상 값({', '.join(allowed)})이 아니라 뺐습니다.")
                out = out[~bad]
    if "ym" in out.columns and out["ym"].notna().any() and out["ym"].nunique() < 2 and spec.id != "my_store":
        warnings.append("기간이 한 달뿐입니다. 12개월 이상을 받아 오면 계절 흐름까지 볼 수 있습니다.")
    return out


def _postprocess(spec: TableSpec, out: pd.DataFrame) -> pd.DataFrame:
    """센터 원본 형식 → 분석용 형식. 유동인구는 날짜·시간·성별·연령 행을 월·평일/주말 평균으로 줄인다."""
    if spec.id == "pop_flow":
        d = out.copy()
        for c in ("resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"):
            if c not in d.columns:
                d[c] = 0.0
            d[c] = d[c].fillna(0.0)
        d["pop"] = d["resd_pop_cnt"] + d["work_pop_cnt"] + d["visit_pop_cnt"]
        keys = ["date", "emd"] + (["hour"] if "hour" in d.columns else [])
        d = d.groupby(keys, as_index=False)["pop"].sum()                      # 성별·연령대 합
        d = d.groupby(["date", "emd"], as_index=False)["pop"].mean()          # 시간대 평균
        d["ym"] = d["date"].str[:7]
        d["day_type"] = pd.to_datetime(d["date"]).dt.day_of_week.map(lambda x: "주말" if x >= 5 else "평일")
        d = d.groupby(["ym", "emd", "day_type"], as_index=False)["pop"].mean().rename(columns={"pop": "pop_cnt"})
        d["pop_cnt"] = d["pop_cnt"].round(1)
        return d
    keys = [c.name for c in spec.columns if c.type != "number"]
    nums = [c.name for c in spec.columns if c.type == "number" and c.name in out.columns]
    keys = [k for k in keys if k in out.columns]
    if keys and nums and out.duplicated(keys).any():
        out = out.groupby(keys, as_index=False)[nums].agg(spec.agg)
    return out


def profile_csv(source: bytes | str | Path | pd.DataFrame, file_name: str = "", table_id: str | None = None,
                manual_mapping: dict[str, str] | None = None, candidates: list[str] | None = None) -> ProfileResult:
    """CSV 한 개를 읽고 맞춘다. table_id 를 안 주면 열 이름으로 표를 알아낸다."""
    if isinstance(source, pd.DataFrame):
        df, enc = source, "memory"
    else:
        try:
            df, enc = read_csv_auto(source)
        except CsvError as e:
            return ProfileResult(None, file_name, "?", 0, errors=[str(e)])
        except Exception as e:  # pandas 파싱 오류 등
            return ProfileResult(None, file_name, "?", 0, errors=[f"CSV 를 읽지 못했습니다: {e}"])
    res = ProfileResult(table_id, file_name, enc, len(df))
    cols = [c for c in df.columns if c != LABEL_COL]
    blocked = sorted(c for c in cols if _norm(c) in {_norm(b) for b in BLOCKED_RAW_COLUMNS})
    if blocked and table_id in (None, "closure"):
        res.errors.append(f"개별 기업·사업자를 알아볼 수 있는 열({', '.join(blocked)})이 있는 원본 행 표입니다. 올릴 수 없습니다. "
                          "센터 안에서 읍면동·업종·월별 개업·폐업 수로 집계한 CSV 만 올려 주세요.")
        return res
    if table_id is None:
        table_id, conf = detect_table(cols, candidates)
        res.table_id = table_id
        if table_id is None:
            res.errors.append("어느 표인지 알 수 없습니다. 열 이름이 센터 표 명세와 다릅니다. 표 종류를 직접 고르세요.")
            res.unmapped_source_columns = cols
            return res
    spec = TABLES[table_id]
    mapping, scores = match_columns(cols, spec)
    if manual_mapping:
        for k, v in manual_mapping.items():
            if v in cols:
                mapping[k], scores[k] = v, 1.0
    res.mapping, res.scores = mapping, scores
    res.missing_required = [c.name for c in spec.read_columns if c.required and c.name not in mapping]
    res.unmapped_source_columns = [c for c in cols if c not in mapping.values()]
    if LABEL_COL in df.columns and df[LABEL_COL].astype(str).str.contains("가상").any():
        res.data_kind = "synthetic"
    if res.missing_required:
        res.errors.append("필수 열을 찾지 못했습니다: " + ", ".join(
            f"{n}({next(c for c in spec.read_columns if c.name == n).desc})" for n in res.missing_required))
        return res
    out = apply_mapping(df, mapping, spec, res.warnings, res.errors)
    req = [c.name for c in spec.read_columns if c.required]
    before = len(out)
    out = out.dropna(subset=req).reset_index(drop=True)
    if len(out) < before:
        res.warnings.append(f"필수 값이 비어 있는 {before - len(out):,}행을 뺐습니다.")
    out = validate(out, spec, res.warnings, res.errors)
    if res.errors or out.empty:
        if not res.errors:
            res.errors.append("쓸 수 있는 행이 없습니다.")
        return res
    out = _postprocess(spec, out).reset_index(drop=True)
    res.rows_in = before
    if "ym" in out.columns:
        res.period = (out["ym"].min(), out["ym"].max())
    elif "date" in out.columns:
        res.period = (out["date"].min()[:7], out["date"].max()[:7])
    res.df = out
    return res
