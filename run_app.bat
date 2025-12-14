@echo off
cd /d "%~dp0"
echo Starting App...

if not exist ".venv\Scripts\python.exe" (
	echo.
	echo ERROR: Virtual environment not found.
	echo Run install_requirements.bat first.
	echo.
	pause
	exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run posture_app.py
pause