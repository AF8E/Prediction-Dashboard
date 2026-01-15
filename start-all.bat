@echo off
echo ========================================
echo Sustainment Prediction Dashboard
echo ========================================
echo.

cd /d %~dp0

echo [1/3] Checking Python dependencies...
py -m pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo Python dependencies OK
echo.

echo [2/3] Checking Node.js dependencies...
if not exist frontend\node_modules (
    echo Installing Node.js dependencies...
    pushd frontend
    call npm install
    if errorlevel 1 (
        echo ERROR: Failed to install Node.js dependencies
        popd
        pause
        exit /b 1
    )
    popd
) else (
    echo Node.js dependencies OK
)
echo.

echo [3/3] Starting servers...
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000 (or 3001 if busy)
echo.

start "Backend Server" cmd /k "cd /d %~dp0backend && py -m uvicorn main:app --reload --port 8000"
timeout /t 3 /nobreak >nul
start "Frontend Server" cmd /k "cd /d %~dp0frontend && npm run dev"

timeout /t 4 /nobreak >nul
start http://localhost:3000

echo.
echo Keep the terminal windows open while using the dashboard.
pause
