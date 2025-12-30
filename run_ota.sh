#!/bin/bash

# Activate the "ota" virtual environment and run edge_ota.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/ota"
OTA_SCRIPT="$SCRIPT_DIR/edge_ota.py"

if [[ ! -f "$OTA_SCRIPT" ]]; then
    echo "❌ edge_ota.py not found in $SCRIPT_DIR"
    exit 1
fi

if [[ ! -d "$VENV_PATH" ]]; then
    echo "❌ OTA virtual environment not found at $VENV_PATH"
    echo "   Run setup_ota_env.sh first."
    exit 1
fi

source "$VENV_PATH/bin/activate"

echo "👉 Running edge_ota.py using virtualenv at $VENV_PATH"
exec python "$OTA_SCRIPT"

