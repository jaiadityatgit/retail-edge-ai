@echo off
title RetailSense OS (HTTPS Mode) // SIH26179
echo ======================================================================
echo   Starting RetailSense OS in Secure HTTPS Mode
echo   (Enables direct Chrome / Safari Mobile Camera Access)
echo.
echo   Dashboard Console:   https://localhost:8000
echo   Mobile Phone Ingest: https://192.168.0.7:8000/mobile_cam
echo   Pitch Deck:          https://localhost:8000/presentation
echo.
echo   * When Chrome shows "Connection is not private", tap:
echo     Advanced -> Proceed to 192.168.0.7 (unsafe)
echo ======================================================================
.\.venv\Scripts\python.exe app.py --ssl
pause
