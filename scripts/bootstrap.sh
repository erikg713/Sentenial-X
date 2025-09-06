#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# Sentenial-X Bootstrap Script
# ----------------------------
# Sets up the Python environment, installs dependencies,
# and verifies that core simulators and examples run correctly.

set -euo pipefail
IFS=$'\n\t'

echo "🔹 Starting Sentenial-X bootstrap..."

# -------------------------------------------------------------------
# 1. Detect Python 3.11+ or fallback
# -------------------------------------------------------------------
PYTHON_BIN=$(which python3 || true)
if [[ -z "$PYTHON_BIN" ]]; then
    echo "❌ Python 3.x not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python detected: $PYTHON_VERSION"

# -------------------------------------------------------------------
# 2. Create virtual environment
# -------------------------------------------------------------------
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "🔹 Creating virtual environment in $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "🔹 Virtual environment already exists."
fi

# Activate the venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -------------------------------------------------------------------
# 3. Upgrade pip and install dependencies
# -------------------------------------------------------------------
echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing dependencies..."
if [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found; installing minimal packages..."
    pip install --upgrade setuptools wheel
fi

# -------------------------------------------------------------------
# 4. Create logs and tmp directories
# -------------------------------------------------------------------
mkdir -p logs tmp
echo "🔹 Logs directory: logs/"
echo "🔹 Temporary files directory: tmp/"

# -------------------------------------------------------------------
# 5. Run smoke test
# -------------------------------------------------------------------
SMOKE_TEST="examples/run_simulation.py"
if [[ -f "$SMOKE_TEST" ]]; then
    echo "🔹 Running smoke test: $SMOKE_TEST"
    python "$SMOKE_TEST"
else
    echo "⚠️ Smoke test script not found: $SMOKE_TEST"
fi

# -------------------------------------------------------------------
# 6. Completion
# -------------------------------------------------------------------
echo "✅ Sentenial-X bootstrap complete!"
