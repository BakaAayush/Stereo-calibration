#!/usr/bin/env bash
# =============================================================================
# install_deps.sh — Idempotent provisioning for Raspberry Pi 5 (64-bit Bookworm)
# =============================================================================
# Run once with internet. After this, runtime needs no network access.
# Usage:  chmod +x install_deps.sh && sudo ./install_deps.sh
# =============================================================================
set -euo pipefail

echo "=== Edge Pipeline — Raspberry Pi 5 Provisioning ==="

# -------------------------------------------------------------------------
# 1. System packages
# -------------------------------------------------------------------------
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-dev python3-pip \
    libopenblas-dev liblapack-dev \
    i2c-tools libi2c-dev \
    libhdf5-dev libatlas-base-dev \
    build-essential cmake pkg-config \
    git curl wget

# -------------------------------------------------------------------------
# 2. Enable I2C (for PCA9685 servo driver)
# -------------------------------------------------------------------------
echo "[2/6] Enabling I2C interface..."
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
    echo "  → I2C enabled in config.txt (reboot required to take effect)"
else
    echo "  → I2C already enabled"
fi

# Ensure i2c-dev module loads at boot
if ! grep -q "^i2c-dev" /etc/modules 2>/dev/null; then
    echo "i2c-dev" >> /etc/modules
fi
modprobe i2c-dev 2>/dev/null || true

# -------------------------------------------------------------------------
# 3. Configure OpenBLAS as default BLAS
# -------------------------------------------------------------------------
echo "[3/6] Configuring OpenBLAS..."
update-alternatives --set libblas.so.3-aarch64-linux-gnu \
    /usr/lib/aarch64-linux-gnu/openblas-pthread/libblas.so.3 2>/dev/null || \
    echo "  → OpenBLAS alternative not available (may already be default)"

# -------------------------------------------------------------------------
# 4. Create Python venv
# -------------------------------------------------------------------------
VENV_DIR="$(dirname "$(readlink -f "$0")")/.venv"
echo "[4/6] Creating Python venv at ${VENV_DIR}..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# -------------------------------------------------------------------------
# 5. Install Python packages
# -------------------------------------------------------------------------
echo "[5/6] Installing Python packages..."
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
pip install -r "$SCRIPT_DIR/requirements.txt"

# Try installing roboticstoolbox (may fail on aarch64)
echo "  → Attempting roboticstoolbox-python install..."
pip install roboticstoolbox-python>=1.1.0 2>/dev/null || \
    echo "  → roboticstoolbox-python not available for aarch64; ikpy fallback will be used"

# -------------------------------------------------------------------------
# 6. Verify OpenBLAS linkage
# -------------------------------------------------------------------------
echo "[6/6] Verifying NumPy OpenBLAS linkage..."
python3 -c "
import numpy as np
config = np.__config__
if hasattr(config, 'show'):
    config.show()
else:
    # NumPy 2.x
    print(np.show_config())
print()
print('NumPy version:', np.__version__)
"

echo ""
echo "=== Provisioning complete ==="
echo "Activate the venv:  source ${VENV_DIR}/bin/activate"
echo "NOTE: Reboot if I2C was just enabled."
