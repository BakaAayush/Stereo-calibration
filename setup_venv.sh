#!/usr/bin/env bash
# =============================================================================
# setup_venv.sh - Create universal venv and install all requirements
# =============================================================================
# Usage:  chmod +x setup_venv.sh && ./setup_venv.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating Universal Python venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel

echo "Installing Depth Sensing requirements..."
pip install -r "$SCRIPT_DIR/Depth sensing/requirements.txt"

echo "Installing Edge Pipeline requirements..."
pip install -r "$SCRIPT_DIR/edge_pipeline/requirements.txt"

echo ""
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
