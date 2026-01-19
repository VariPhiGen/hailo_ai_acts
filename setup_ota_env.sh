#!/bin/bash

# Create/refresh a Python virtual environment named "ota" for running edge_ota.py.
# Installs only the dependencies required by edge_ota.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="ota"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"

PY_BIN="$(command -v python3 || command -v python || true)"

if [[ -z "$PY_BIN" ]]; then
    echo "❌ python3/python not found in PATH"
    exit 1
fi

echo "👉 Using Python at: $PY_BIN"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "👉 Creating virtual environment: $VENV_PATH"
    "$PY_BIN" -m venv "$VENV_PATH"
else
    echo "ℹ️  Virtual environment already exists: $VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

echo "👉 Upgrading pip and installing edge_ota dependencies..."
pip install --upgrade pip
pip install websockets requests python-dotenv

deactivate

echo "👉 Making run_ota.sh executable..."
chmod +x "$SCRIPT_DIR/run_ota.sh"

echo "✅ OTA environment ready at $VENV_PATH"
echo "   Activate with: source \"$VENV_PATH/bin/activate\""
echo "   Run OTA with: ./run_ota.sh"
echo "   Note: Scripts will be made executable automatically during OTA operations"

