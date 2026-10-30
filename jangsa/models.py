"""계획·결과 데이터 모양(pydantic)."""
from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .tools.export_gate import GateResult


class StepPlan(BaseModel):
    id: str
    title: str
    purpose: str = ""
    sql: str
    tables: list[str] = Field(default_factory=list)
    chart: dict[str, Any] | None = None
    applied: list[str] = Field(default_factory=list)       # 응용집계(비율·지수) 열 이름
    source: str = "template"                                 # template | llm | local
    optional: bool = False                                   # 표가 없거나 결과가 비면 조용히 건너뜀


class ChecklistItem(BaseModel):
    table_id: str
    center_no: str
    center_name: str
    condition: str
    status: str                  # 불러옴 | 없음
    origin: str = ""             # synthetic | center | public | user
    kind: str = "center"         # center(센터에서 받아 올 CSV) | optional(직접 만든 집계) | public(앱에 들어 있는 공개 실데이터) | user


class Plan(BaseModel):
    question: str
    intent: str
    intent_label: str
    emd: str | None = None
    industry: str | None = None
    city: str | None = None            # 읍면동을 안 정했을 때 '제주시'·'서귀포시' 범위
    needs_confirm: list[str] = Field(default_factory=list)
    period: tuple[str, str] | None = None                    # 분석 기간(최근 12개월) 'YYYY-MM'
    period_all: tuple[str, str] | None = None
    tables: list[str] = Field(default_factory=list)
    steps: list[StepPlan] = Field(default_factory=list)
    checklist: list[ChecklistItem] = Field(default_factory=list)
    mode: str = "offline"
    model: str = ""
    notes: list[str] = Field(default_factory=list)
    from_cache: bool = False


class StepResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    step: StepPlan
    sql_run: str = ""                    # 실제 실행된 SQL(LIMIT 보정 후)
    attempts: int = 1
    error: str = ""
    skipped: str = ""
    rows: int = 0
    source_rows: int = 0
    elapsed_ms: int = 0
    gate: GateResult | None = None
    df: Any = None                       # 센터 안 화면용: export_gate 를 통과한(10 미만 처리된) 표. 원천 결과는 버린다.
    df_export: Any = None                # LLM·보고서·PNG 로 나갈 수 있는 표(엄격 모드면 응용집계 열만)
    origin: str = ""                     # synthetic(가상) | public(공개 실데이터) | center | mixed
    findings: list[str] = Field(default_factory=list)
    fallback: bool = False


class Explanation(BaseModel):
    headline: str
    points: list[dict[str, Any]] = Field(default_factory=list)   # {"text": str, "tables": [int]}
    checkpoints: list[str] = Field(default_factory=list)
    limits: list[str] = Field(default_factory=list)
    source: str = "template"             # template | llm
    notes: list[str] = Field(default_factory=list)
