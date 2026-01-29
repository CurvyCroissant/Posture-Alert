@echo off
cd /d "%~dp0"

IF NOT EXIST .venv (
    echo [ERROR] Virtual environment not found.
    echo Please run "install_requirements.bat" first.
    pause
    exit /b 1
)

echo Starting Posture Alert...
.venv\Scripts\python -m streamlit run app.py

pause
