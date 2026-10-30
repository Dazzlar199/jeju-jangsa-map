"""데이터 저장소: 읽어 들인 표를 DuckDB(메모리)에 올려 두는 곳. 디스크에 개인 자료를 쓰지 않는다."""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from .config import DATA_PUBLIC, DATA_SYNTHETIC
from .schema import TABLES
from .tools.csv_profile import ProfileResult, profile_csv


@dataclass
class TableInfo:
    table_id: str
    file_name: str
    rows: int
    origin: str            # synthetic(가상) | center(센터 CSV) | public(공개 실데이터) | user(내 가게)
    period: tuple[str, str] | None
    encoding: str = ""
    notes: list[str] = field(default_factory=list)


def ensure_synthetic(folder: Path | str = DATA_SYNTHETIC) -> Path:
    """가상 데이터 파일이 없으면(저장소에 넣지 않은 것) 고정 시드로 만들고, 실제로 쓴 폴더를 돌려준다.
    저장소 폴더에 쓸 수 없는 환경(읽기 전용 배포 등)이면 임시 폴더에 만든다."""
    import tempfile
    from .synthetic import write_all
    folder = Path(folder)
    need = lambda d: not all((d / sp.synthetic_file).exists() for sp in TABLES.values() if sp.synthetic_file)
    if need(folder):
        try:
            write_all(folder)
        except OSError:
            folder = Path(tempfile.gettempdir()) / "jangsa_synthetic"
            if need(folder):
                write_all(folder)
    return folder


def read_synthetic(folder: Path | str = DATA_SYNTHETIC, include_my_store: bool = False) -> list[ProfileResult]:
    folder = ensure_synthetic(folder)
    out = []
    for tid, spec in TABLES.items():
        if not spec.synthetic_file or (tid == "my_store" and not include_my_store):
            continue
        p = Path(folder) / spec.synthetic_file
        res = profile_csv(p, p.name, table_id=tid)
        if not res.ok:
            raise RuntimeError(f"가상 데이터 {p.name} 를 읽지 못했습니다: {res.errors}")
        out.append(res)
    return out


def read_public(folder: Path | str = DATA_PUBLIC) -> list:
    from .public_data import load_all
    return load_all(folder)


class DataStore:
    def __init__(self) -> None:
        self.con = self._connect()
        self.tables: dict[str, TableInfo] = {}

    @staticmethod
    def _connect() -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        con.execute("SET enable_external_access=false")   # 파일·네트워크 접근 차단
        return con

    # ── 등록 ────────────────────────────────────────────────────────
    def register(self, res: ProfileResult, origin: str | None = None) -> TableInfo:
        if not res.ok:
            raise ValueError("검증을 통과하지 못한 CSV 는 올릴 수 없습니다: " + "; ".join(res.errors))
        tid = res.table_id
        if origin is None:
            origin = "synthetic" if res.data_kind == "synthetic" else ("user" if tid == "my_store" else "center")
        self.con.register("_incoming", res.df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{tid}" AS SELECT * FROM _incoming')
        self.con.unregister("_incoming")
        info = TableInfo(tid, res.file_name, len(res.df), origin, res.period, res.encoding)
        self.tables[tid] = info
        return info

    def load_synthetic(self, folder: Path | str = DATA_SYNTHETIC, include_my_store: bool = False) -> list[TableInfo]:
        return [self.register(r, origin="synthetic") for r in read_synthetic(folder, include_my_store)]

    def load_public(self, folder: Path | str = DATA_PUBLIC) -> list[TableInfo]:
        """data/public/ 의 공개 실데이터를 읽어 '실데이터(public)' 표로 올린다. 파일이 없으면 그 표만 빠진다."""
        return [self.register_public(pl) for pl in read_public(folder)]

    def register_public(self, pl) -> TableInfo:
        self.con.register("_incoming", pl.df)
        self.con.execute(f'CREATE OR REPLACE TABLE "{pl.table_id}" AS SELECT * FROM _incoming')
        self.con.unregister("_incoming")
        info = TableInfo(pl.table_id, pl.file_name, len(pl.df), "public", pl.period, "", notes=pl.notes)
        self.tables[pl.table_id] = info
        return info

    # ── 조회 ────────────────────────────────────────────────────────
    def has(self, tid: str) -> bool:
        return tid in self.tables

    def available(self) -> list[str]:
        return list(self.tables)

    def latest_ym(self, tid: str = "card_emd") -> str | None:
        if tid not in self.tables:
            return None
        return self.con.execute(f'SELECT MAX(ym) FROM "{tid}"').fetchone()[0]

    def months_back(self, n: int = 12, tid: str = "card_emd") -> tuple[str, str] | None:
        last = self.latest_ym(tid)
        if not last:
            return None
        y, m = int(last[:4]), int(last[5:7])
        total = y * 12 + (m - 1) - (n - 1)
        return f"{total // 12:04d}-{total % 12 + 1:02d}", last

    def origins(self, table_ids: list[str]) -> dict[str, str]:
        return {t: self.tables[t].origin for t in table_ids if t in self.tables}

    def is_synthetic(self, table_ids: list[str]) -> bool:
        return any(i.origin == "synthetic" for t, i in self.tables.items() if t in table_ids)

    def distinct_values(self, tid: str, col: str) -> list[str]:
        if tid not in self.tables:
            return []
        return [r[0] for r in self.con.execute(f'SELECT DISTINCT "{col}" FROM "{tid}" WHERE "{col}" IS NOT NULL ORDER BY 1').fetchall()]

    def drop(self, tid: str) -> None:
        if tid in self.tables:
            self.con.execute(f'DROP TABLE IF EXISTS "{tid}"')
            del self.tables[tid]

    def clear(self, only_origin: str | None = None) -> None:
        for tid in [t for t, i in self.tables.items() if only_origin is None or i.origin == only_origin]:
            self.drop(tid)

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass
