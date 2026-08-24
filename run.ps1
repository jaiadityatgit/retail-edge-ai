Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Starting RetailSense OS // Dual-Camera Vision Platform" -ForegroundColor Cyan
Write-Host "  Dashboard Console:   http://localhost:8000" -ForegroundColor Green
Write-Host "  Mobile Phone Ingest: http://192.168.0.7:8000/mobile_cam" -ForegroundColor Yellow
Write-Host "  Pitch Deck:          http://localhost:8000/presentation" -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" app.py
