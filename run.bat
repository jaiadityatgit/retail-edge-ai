@echo off
title RetailSense OS // SIH26179
echo ======================================================================
echo   Starting RetailSense OS // Dual-Camera Vision Platform
echo   Dashboard Console:   http://localhost:8000
echo   Mobile Phone Ingest: http://192.168.0.7:8000/mobile_cam
echo   Pitch Deck:          http://localhost:8000/presentation
echo ======================================================================
.\.venv\Scripts\python.exe app.py
pause
