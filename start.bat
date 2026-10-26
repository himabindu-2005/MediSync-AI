@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run setup is needed. See README.md, section "Windows quick start".
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m uvicorn app.main:app --reload
