@echo off
echo ====================================
echo URIS Dashboard Setup Script
echo ====================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Verifying installation...
python -c "import streamlit, plotly, pandas, numpy; print('✓ All dependencies installed')" 2>nul || (
    echo ✗ Installation failed
    pause
    exit /b 1
)

echo [3/3] Launching URIS Dashboard...
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
streamlit run dashboard_app.py

pause
