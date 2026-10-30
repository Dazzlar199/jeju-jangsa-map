#!/usr/bin/env bash
# macOS/Linux 용 설치. Python 3.11 권장.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.11 || command -v python3)
echo "[1/3] 가상환경(.venv) 만들기 ($PY)"
"$PY" -m venv .venv
echo "[2/3] 패키지 설치(고정 버전)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "[3/3] 가상 데이터 만들기"
.venv/bin/python scripts/make_synthetic.py
echo "끝. ./run.sh 로 실행하세요."
