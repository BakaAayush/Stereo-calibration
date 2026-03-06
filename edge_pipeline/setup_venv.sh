#!/usr/bin/env bash
# =============================================================================
# setup_venv.sh — Create venv and install pinned requirements
# =============================================================================
# Lightweight alternative to install_deps.sh (assumes system deps are present).
# Usage:  chmod +x setup_venv.sh && ./setup_venv.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating Python venv at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
