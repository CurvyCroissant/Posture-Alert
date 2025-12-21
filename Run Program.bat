@echo off
cd /d "%~dp0"
echo Starting Posture Alert...
streamlit run posture_guard.py
pause