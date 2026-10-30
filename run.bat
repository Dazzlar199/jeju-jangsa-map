@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 먼저 install.bat 을 실행하세요.
  pause
  exit /b 1
)
echo 우리 동네 장사 지도를 시작합니다. 잠시 뒤 브라우저가 열립니다. (끝내려면 이 창을 닫으세요)
echo.| ".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.maxUploadSize 200 --browser.gatherUsageStats false
pause
