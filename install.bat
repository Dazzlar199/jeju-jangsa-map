@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo ================================================
echo  우리 동네 장사 지도 - 설치 (처음 한 번만)
echo ================================================
echo [1/3] 가상환경(.venv)을 만듭니다. Python 3.11 이 필요합니다.
where py >nul 2>nul
if %errorlevel%==0 ( py -3.11 -m venv .venv ) else ( python -m venv .venv )
if errorlevel 1 (
  echo Python 3.11 을 찾지 못했습니다. https://www.python.org 에서 설치 후 다시 실행하세요.
  pause
  exit /b 1
)
echo [2/3] 필요한 패키지를 설치합니다(requirements.txt 의 고정 버전).
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo 패키지 설치에 실패했습니다. 인터넷 연결 또는 사전 설치 허용 목록을 확인하세요.
  pause
  exit /b 1
)
echo [3/3] 가상 데이터를 만듭니다(고정 시드, 약 10초).
".venv\Scripts\python.exe" scripts\make_synthetic.py
echo.
echo 설치가 끝났습니다. run.bat 을 더블클릭하면 브라우저에서 열립니다.
pause
