# =============================================================================
# config.py — Shared Configuration for Stereo Depth Sensing Pipeline
# =============================================================================
# Edit the values here; they are imported by all three pipeline scripts.
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Camera Settings
# -----------------------------------------------------------------------------
# Camera indices confirmed by hardware scan:
#   ID 0 = Built-in laptop webcam (not used)
#   ID 1 = HP w200  → LEFT camera  (supports up to 1080p)
#   ID 2 = HP w200  → RIGHT camera (supports up to 1080p)
LEFT_CAM_ID  = 1  # HP w200 (Left)
RIGHT_CAM_ID = 0   # HP w200 (Right)

# Resolution locked to 1280x720 (720p). Both HP w200 cameras support up to 1080p,
# but 720p gives a good balance of speed and accuracy for stereo matching.
FRAME_WIDTH  = 1280   # 720p — matches HP w100 maximum
FRAME_HEIGHT = 720

# -----------------------------------------------------------------------------
# Calibration Board Settings
# -----------------------------------------------------------------------------
# Number of INNER corners (not squares).
# A printed 11x8 square board has (10, 7) inner corners.
CHESSBOARD_SIZE = (7, 10)

# Real-world size of ONE square on your printed checkerboard (in millimetres).
# 2.3 cm = 23.0 mm
SQUARE_SIZE_MM = 23.0

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
CALIBRATION_IMAGES_FOLDER = os.path.join(BASE_DIR, "calibration_images")
CALIBRATION_DATA_FILE     = os.path.join(BASE_DIR, "stereo_calibration.npz")

# -----------------------------------------------------------------------------
# SGBM Disparity Matcher Defaults (used in Stage 3)
# These are good starting values; tune live with the trackbars.
# -----------------------------------------------------------------------------
SGBM_NUM_DISPARITIES   = 128   # Must be a multiple of 16.
SGBM_BLOCK_SIZE        = 11    # Must be an odd number ≥ 5.
SGBM_MIN_DISPARITY     = 0
SGBM_UNIQUENESS_RATIO  = 10
SGBM_SPECKLE_WIN_SIZE  = 100
SGBM_SPECKLE_RANGE     = 32
SGBM_DISP12_MAX_DIFF   = 1

# WLS filter parameters (smooths the disparity map)
WLS_LAMBDA = 8000.0
WLS_SIGMA  = 1.5
