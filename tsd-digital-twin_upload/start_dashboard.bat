@echo off
REM ============================================================================
REM  start_dashboard.bat - one-click start of the TSD digital twin dashboard (Windows)
REM  First run: creates a private Python environment (.venv), installs the packages,
REM  regenerates the simulated data if it is missing. Later runs: just starts the app.
REM  Needs Python 3.10 or newer from https://www.python.org (tick "Add python to PATH").
REM ============================================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10+ from https://www.python.org and tick "Add python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the Python environment (.venv) - once only...
    python -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.installed" (
    echo Installing packages - once only, 2-5 minutes depending on the connection...
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt || (echo Package installation failed & pause & exit /b 1)
    echo ok> ".venv\.installed"
)

if not exist "data\raw\timeseries.csv" (
    echo Generating the simulated plant year - about 20 seconds...
    python scripts\check_step3.py || (echo Data generation failed & pause & exit /b 1)
)

echo.
echo Starting the dashboard. Your browser will open at http://localhost:8501
echo Close this window (or press Ctrl+C here) to stop it.
echo.
python -m streamlit run dashboard\app.py --browser.gatherUsageStats false
pause
