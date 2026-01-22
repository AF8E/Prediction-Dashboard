@echo off
REM ============================================================
REM Infrastructure Failure Prediction Dashboard
REM Automated Setup and Run Script
REM ============================================================

echo ============================================================
echo Infrastructure Failure Prediction Dashboard
echo Automated Setup Script
echo ============================================================
echo.

REM Check Python installation (try py first, then python)
py --version >nul 2>&1
if errorlevel 1 (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python is not installed or not in PATH
        echo Please install Python 3.8 or higher from https://www.python.org/
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
) else (
    set PYTHON_CMD=py
)

echo [1/5] Checking Python version...
%PYTHON_CMD% --version
echo.

REM Create virtual environment if it doesn't exist
echo [2/5] Setting up virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated.
echo.

REM Install dependencies
echo [4/5] Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed successfully.
echo.

REM Check if data files exist
echo [5/5] Checking data files...
if not exist "data\generated_facility_data.xlsx" (
    echo WARNING: data\generated_facility_data.xlsx not found
    echo Please ensure your inventory file is in the data\ directory
)
if not exist "data\Simulated_Data.xlsx" (
    echo WARNING: data\Simulated_Data.xlsx not found
    echo Please ensure your work order file is in the data\ directory
)
echo.

echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Run Phase 1: %PYTHON_CMD% src\data_prep.py
echo   2. Run Phase 2: %PYTHON_CMD% src\train_model.py
echo   3. Run Phase 3: %PYTHON_CMD% app.py
echo.
echo Or run all phases automatically? (Y/N)
set /p run_all="> "

if /i "%run_all%"=="Y" (
    echo.
    echo ============================================================
    echo Running Phase 1: Data Preparation
    echo ============================================================
    %PYTHON_CMD% src\data_prep.py
    if errorlevel 1 (
        echo ERROR: Phase 1 failed
        pause
        exit /b 1
    )
    echo.
    
    echo ============================================================
    echo Running Phase 2: Model Training
    echo ============================================================
    %PYTHON_CMD% src\train_model.py
    if errorlevel 1 (
        echo ERROR: Phase 2 failed
        pause
        exit /b 1
    )
    echo.
    
    echo ============================================================
    echo Starting Flask Dashboard (Phase 3)
    echo ============================================================
    echo Dashboard will be available at: http://localhost:5000
    echo Press Ctrl+C to stop the server
    echo.
    %PYTHON_CMD% app.py
) else (
    echo.
    echo Setup complete. Run the phases manually as needed.
    echo.
)

pause
