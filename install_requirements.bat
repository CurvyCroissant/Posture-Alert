@echo off
cd /d "%~dp0"
echo ==========================================
echo   Installing Posture Alert Requirements
echo ==========================================

set "PY_VER=3.11"

REM Ensure a compatible Python is available (MediaPipe wheels are not published for every Python version)
py -%PY_VER% -c "import sys; print(sys.version)" >nul 2>&1
if errorlevel 1 (
	echo.
	echo ERROR: Python %PY_VER% was not found via the py launcher.
		echo Install Python %PY_VER% ^(64-bit^), then re-run this script.
	echo You can check installed versions with: py -0p
	echo.
	pause
	exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
	echo Creating virtual environment in .venv using Python %PY_VER%...
	py -%PY_VER% -m venv .venv
	if errorlevel 1 (
		echo.
		echo ERROR: Failed to create virtual environment.
		echo.
		pause
		exit /b 1
	)
) else (
	echo Using existing virtual environment: .venv
)

echo Upgrading pip tooling...
".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
	echo.
	echo ERROR: Failed to upgrade pip tooling.
	echo.
	pause
	exit /b 1
)

echo Installing requirements from requirements.txt...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
	echo.
	echo ERROR: Dependency installation failed.
	echo If this is a network/proxy issue, try: ".venv\Scripts\python.exe" -m pip install -r requirements.txt -v
	echo.
	pause
	exit /b 1
)
echo.
echo ==========================================
echo   Installation Complete!
echo ==========================================
pause