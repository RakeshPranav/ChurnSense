@echo off
cd /d "%~dp0"

echo Starting ChurnSense API on port 8000...
start "ChurnSense API" cmd /k "venv\Scripts\activate && python -m uvicorn api:app --port 8000"

echo Starting frontend on port 5500...
start "ChurnSense Frontend" cmd /k "cd frontend && ..\venv\Scripts\python.exe -m http.server 5500"

timeout /t 4 >nul
start http://localhost:5500