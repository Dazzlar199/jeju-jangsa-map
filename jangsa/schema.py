"""표 명세(config/tables.yaml)를 코드에서 쓰기 좋게 감싼다. LLM 에 보내는 '표 명세 문장'도 여기서 만든다."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import tables_config


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type: str
    required: bool
    desc: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class TableSpec:
    id: str
    center_no: str
    center_name: str
    provider: str
    export_level: str
    title: str
    grain: str
    synthetic_file: str
    filter_hint: str
    public_glob: str | None
    columns: tuple[ColumnSpec, ...]
    raw_columns: tuple[ColumnSpec, ...] = ()      # CSV 를 읽을 때 쓰는 열(없으면 columns 와 같다)
    agg: str = "sum"                               # 같은 키가 겹칠 때 합치는 방법(sum|mean)

    @property
    def read_columns(self) -> tuple[ColumnSpec, ...]:
        return self.raw_columns or self.columns

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column(self, name: str) -> ColumnSpec:
        return next(c for c in self.columns if c.name == name)


def _build() -> dict[str, TableSpec]:
    out = {}
    for tid, t in tables_config()["tables"].items():
        def mk(d):
            return tuple(ColumnSpec(n, c["type"], c.get("required", True), c["desc"], tuple(c.get("aliases", [])))
                         for n, c in d.items())
        out[tid] = TableSpec(tid, t["center_no"], t["center_name"], t["provider"], t["export_level"], t["title"],
                             t["grain"], t["synthetic_file"], t.get("filter_hint", ""), t.get("public_glob"), mk(t["columns"]),
                             mk(t.get("raw_columns", {})), t.get("agg", "sum"))
    return out


TABLES: dict[str, TableSpec] = _build()
CODES: dict = tables_config()["codes"]
INDUSTRIES: list[str] = list(CODES["industries"].keys())
INDUSTRY_CODES: dict[str, list[str]] = {k: v["codes"] for k, v in CODES["industries"].items()}
INDUSTRY_WORDS: dict[str, list[str]] = {k: v["words"] for k, v in CODES["industries"].items()}
CODE_TO_INDUSTRY: dict[str, str] = {c: k for k, cs in INDUSTRY_CODES.items() for c in cs}
UNSUPPORTED_INDUSTRY_WORDS: list[str] = CODES["unsupported_industry_words"]
LEGAL_TO_ADMIN: dict[str, str] = CODES["legal_to_admin"]
BC_REGIONS: dict[str, list[str]] = CODES["bc_regions"]
DSTRCT_SE: dict[str, list[str]] = CODES["dstrct_se"]
TIME_BANDS: list[str] = CODES["time_bands"]
DOWS: list[str] = CODES["dow"]
DAY_TYPES: list[str] = CODES["day_types"]
EMD_CITY: dict[str, str] = {e: city for city, d in CODES["emd"].items() for e in d}
EMD_INFO: dict[str, dict] = {e: v for d in CODES["emd"].values() for e, v in d.items()}
EMD_LIST: list[str] = list(EMD_CITY)


def to_admin(name: str) -> str:
    """센터 카드 표의 법정동 이름 → 분석 단위 행정동(읍면동). 매핑이 없으면 정리만 해서 돌려준다."""
    n = normalize_emd(name)
    return LEGAL_TO_ADMIN.get(n, n)


def region_of(emd: str) -> str | None:
    """BC카드 권역(북부·남부·동부·서부). 동 지역은 시 기준."""
    for r, lst in BC_REGIONS.items():
        if emd in lst:
            return r
    city = EMD_CITY.get(emd)
    if emd.endswith("동") and city:
        return "북부" if city == "제주시" else "남부"
    return None


def table_nos() -> dict[str, int]:
    """근거표 번호가 아니라 표 이름 → 순번(명세 목록용)."""
    return {tid: i + 1 for i, tid in enumerate(TABLES)}


def spec_text_for_llm(table_ids: list[str] | None = None) -> str:
    """LLM 에 보내는 '표 명세' 문장. 열 이름·형식·코드값만 들어가고 실제 데이터 값은 한 줄도 없다."""
    ids = table_ids or [t for t in TABLES if t != "my_store"]
    ids = [t for t in ids if t in TABLES]
    lines = ["[사용할 수 있는 표(DuckDB 보기). 아래 이름과 열만 쓸 수 있다]"]
    for tid in ids:
        t = TABLES[tid]
        lines.append(f"- {tid}: {t.title} ({t.grain})")
        for c in t.columns:
            lines.append(f"    {c.name} ({c.type}) {c.desc}")
    lines.append("[코드값]")
    lines.append("ym 는 'YYYY-MM' 문자열. 업종: " + ", ".join(f"{k}(코드 {'·'.join(v)})" for k, v in INDUSTRY_CODES.items()))
    lines.append("시간대: " + ", ".join(f"{t}({CODES['time_band_hours'][t]})" for t in TIME_BANDS) + " / 요일: " + ", ".join(DOWS) + " / 평일·주말: " + ", ".join(DAY_TYPES))
    lines.append("읍면동(" + str(len(EMD_LIST)) + "곳): " + ", ".join(EMD_LIST))
    return "\n".join(lines)


def normalize_emd(name: str) -> str:
    """'제주시 애월읍' → '애월읍', 공백 제거. 목록에 없으면 그대로 돌려준다."""
    s = re.sub(r"\s+", "", str(name))
    s = re.sub(r"^(제주특별자치도)?(제주시|서귀포시)", "", s)
    return s
