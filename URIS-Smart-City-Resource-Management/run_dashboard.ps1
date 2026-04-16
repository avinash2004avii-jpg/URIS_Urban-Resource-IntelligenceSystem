#!/usr/bin/env pwsh

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "URIS Dashboard Setup Script" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host ""
Write-Host "[2/3] Verifying installation..." -ForegroundColor Yellow
try {
    python -c "import streamlit, plotly, pandas, numpy; print('✓ All dependencies installed')"
} catch {
    Write-Host "✗ Installation failed" -ForegroundColor Red
    Exit 1
}

Write-Host ""
Write-Host "[3/3] Launching URIS Dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "Dashboard will open at: http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan
Write-Host ""

streamlit run dashboard_app.py
