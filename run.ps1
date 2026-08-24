Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Starting EdgeRetail AI // SIH26179 On-Device Vision Server" -ForegroundColor Cyan
Write-Host "  Dashboard URL: http://localhost:8000" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\python.exe" app.py
