@echo off
cd /d "%~dp0"
echo ==========================================
echo   Installing Posture Alert Requirements
echo ==========================================
pip install -r requirements.txt
echo.
echo ==========================================
echo   Installation Complete!
echo ==========================================
pause