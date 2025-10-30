@echo off
echo ========================================
echo  Ad-Context Congruence - PUBLIC Server
echo ========================================
echo.

cd /d "%~dp0"

if not exist cloudflared.exe (
    echo ✗ Cloudflared not found!
    echo.
    echo Please run: install-cloudflare-tunnel.bat first
    echo.
    pause
    exit /b 1
)

echo Starting application...
echo.
start "Ad-Congruence App" python app.py

echo Waiting for app to start (10 seconds)...
timeout /t 10 /nobreak > nul

echo.
echo Creating Cloudflare Tunnel...
echo ========================================
echo.
echo Your PUBLIC URL will appear below:
echo.
echo Share this URL with anyone! 🎉
echo.
echo Press Ctrl+C to stop (will close both app and tunnel)
echo ========================================
echo.

cloudflared.exe tunnel --url http://localhost:7860

pause

