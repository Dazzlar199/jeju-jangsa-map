"""분석 기록(RunLog): 실행한 SQL, 사용 파일, 행 수, export_gate 결과, LLM 호출 수·토큰·예상 비용.

기록에는 데이터 값을 넣지 않는다(SQL·행 수·판정만). 그래서 runs/ 에 남겨도 반출 위험이 없고,
같은 질문을 다시 돌렸을 때 같은 SQL·같은 결과인지 대조할 수 있다.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import RUNS_DIR


@dataclass
class LLMCall:
    purpose: str
    provider: str
    model: str
    in_tokens: int
    out_tokens: int
    cost_usd: float
    ms: int
    ok: bool = True
    error: str = ""


@dataclass
class RunLog:
    question: str = ""
    mode: str = "offline"
    model: str = ""
    started: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    events: list[dict] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    preview: dict = field(default_factory=dict)
    _t0: float = field(default_factory=time.time, repr=False)

    def event(self, msg: str, **kw) -> None:
        self.events.append({"t": round(time.time() - self._t0, 2), "msg": msg, **kw})

    def add_call(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    @property
    def n_calls(self) -> int:
        return len(self.llm_calls)

    @property
    def total_in(self) -> int:
        return sum(c.in_tokens for c in self.llm_calls)

    @property
    def total_out(self) -> int:
        return sum(c.out_tokens for c in self.llm_calls)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_t0", None)
        d["totals"] = {"llm_calls": self.n_calls, "in_tokens": self.total_in, "out_tokens": self.total_out,
                       "cost_usd": round(self.total_cost_usd, 6)}
        return d

    def save(self, folder: Path | str = RUNS_DIR) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(self.question.encode("utf-8")).hexdigest()[:8]
        p = folder / f"{datetime.now():%Y%m%d-%H%M%S}_{h}.json"
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p


def plan_cache_key(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def plan_cache_path(key: str, folder: Path | str = RUNS_DIR) -> Path:
    return Path(folder) / "plans" / f"{key}.json"
