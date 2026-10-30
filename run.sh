#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "먼저 ./install.sh 를 실행하세요."; exit 1; }
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 200 --browser.gatherUsageStats false
