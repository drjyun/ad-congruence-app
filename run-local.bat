@echo off
echo ========================================
echo  Ad-Context Congruence - Local Server
echo ========================================
echo.
echo Starting application on http://localhost:7860
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

cd /d "%~dp0"
python app.py

pause

