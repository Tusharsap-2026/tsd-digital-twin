#!/usr/bin/env bash
# ============================================================================
#  start_dashboard.sh - one-click start of the TSD digital twin dashboard (Linux / macOS)
#  First run: creates a private Python environment (.venv), installs the packages,
#  regenerates the simulated data if it is missing. Later runs: just starts the app.
#  Needs Python 3.10 or newer.          Usage:  bash start_dashboard.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
    echo "Python was not found. Install Python 3.10+ and try again."; exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
    echo "Creating the Python environment (.venv) - once only..."
    "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [ ! -f ".venv/.installed" ]; then
    echo "Installing packages - once only, 2-5 minutes depending on the connection..."
    python -m pip install --upgrade pip >/dev/null
    python -m pip install -r requirements.txt
    echo ok > .venv/.installed
fi

if [ ! -f "data/raw/timeseries.csv" ]; then
    echo "Generating the simulated plant year - about 20 seconds..."
    python scripts/check_step3.py
fi

echo
echo "Starting the dashboard at http://localhost:8501  (Ctrl+C here to stop)"
echo
python -m streamlit run dashboard/app.py --browser.gatherUsageStats false
