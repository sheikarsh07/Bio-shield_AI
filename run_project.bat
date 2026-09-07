@echo off
title BioShield AI Starter
echo ============================================================
echo   BioShield AI - Starting Backend (this takes 15-30s)...
echo ============================================================

:: Start backend in a minimized separate window
start "BioShield Backend" /min py -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

:: Wait until the backend health endpoint responds (retry every 3 seconds)
:waitloop
timeout /t 3 >nul
py -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2)" 2>nul
if %errorlevel% neq 0 (
    echo   Still loading AI models, please wait...
    goto waitloop
)

echo   Backend is READY!
echo ============================================================
echo   Starting Streamlit UI...
echo ============================================================
start "" http://localhost:8501
py -m streamlit run app/streamlit_app.py --server.port 8501
pause