@echo off
echo ========================================
echo  Installing Cloudflare Tunnel
echo ========================================
echo.

cd /d "%~dp0"

echo Downloading cloudflared...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"

if exist cloudflared.exe (
    echo.
    echo ✓ Cloudflared downloaded successfully!
    echo.
    echo To expose your app publicly, run: run-public.bat
    echo.
) else (
    echo.
    echo ✗ Download failed. Please check your internet connection.
    echo.
)

pause

