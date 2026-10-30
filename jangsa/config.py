"""설정 파일(config/*.yaml) 읽기와 경로. 설정 파일 위치는 이 저장소 안으로 고정한다."""
from __future__ import annotations

import functools
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_SYNTHETIC = ROOT / "data" / "synthetic"
DATA_PUBLIC = ROOT / "data" / "public"
RUNS_DIR = ROOT / "runs"
REPORTS_DIR = ROOT / "reports"

# 공개 데모(Streamlit Community Cloud 등 메모리 1GB 안): JANGSA_DEMO=1 이면 가상 데이터를 작게 만들고 안내 문구를 더 붙인다.
# Streamlit Community Cloud 는 저장소를 /mount/src 아래에 두므로 거기서 돌면 환경변수 없이도 데모로 본다.
DEMO = os.environ.get("JANGSA_DEMO", "").strip() in ("1", "true", "True", "yes") or Path(__file__).resolve().parts[:3] == ("/", "mount", "src")

ATTRIBUTION = ("본 자료는 제주빅데이터센터에서 제공하는 자료를 활용한 것이며, "
               "연구결과는 제주특별자치도 및 제주테크노파크와 관련이 없음을 밝힙니다")
SYNTHETIC_NOTICE = "가상 데이터(시연용)입니다. 실제 제주의 통계가 아닙니다."


@functools.lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tables_config() -> dict:
    return load_yaml("tables.yaml")


def export_rules() -> dict:
    return load_yaml("export_rules.yaml")


def llm_config() -> dict:
    return load_yaml("llm.yaml")
