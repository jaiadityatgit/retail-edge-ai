Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Starting RetailSense OS // Dual-Camera Vision Platform (HTTPS Mode)" -ForegroundColor Cyan
Write-Host "  Dashboard Console:   https://localhost:8000" -ForegroundColor Green
Write-Host "  Mobile Phone Ingest: https://192.168.0.7:8000/mobile_cam" -ForegroundColor Yellow
Write-Host "  Pitch Deck:          https://localhost:8000/presentation" -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "* Mobile Chrome: Tap 'Advanced' -> 'Proceed to 192.168.0.7 (unsafe)'" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" app.py --ssl
