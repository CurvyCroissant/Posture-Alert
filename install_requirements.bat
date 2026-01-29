@echo off
cd /d "%~dp0"

echo Installing requirements...

set "PY_CMD="
py -3.11 --version >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD (
    python --version >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python not found. Install Python 3.11+ and try again.
    goto :END
)

if not exist .venv (
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto :END
)

if not exist .venv\Scripts\python.exe (
    echo [ERROR] Virtual environment was not created.
    goto :END
)

.venv\Scripts\python -m pip install --upgrade pip
if errorlevel 1 goto :END

.venv\Scripts\python -m pip install "streamlit>=1.30" "opencv-python>=4.8" "mediapipe==0.10.32" "numpy>=1.24,<2.0" "pandas>=2.0"
if errorlevel 1 goto :END

echo.
echo Done. You can now run "run_app.bat".

:END
pause
