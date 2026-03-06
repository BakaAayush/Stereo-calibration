"""
02_stereo_calibrate.py
=======================
Stage 2 of the Stereo Depth Sensing Pipeline.

PURPOSE
-------
Reads the image pairs captured in Stage 1, computes the stereo camera
calibration (intrinsics + extrinsics), performs stereo rectification, and
saves all the resulting matrices to 'stereo_calibration.npz'.

Run ONCE after collecting at least 20 good calibration image pairs.

OUTPUT
------
  stereo_calibration.npz   ← loaded by Stage 3 (03_depth_map_live.py)

WHAT GOOD RESULTS LOOK LIKE
-----------------------------
  RMS reprojection error < 1.0 px  — acceptable
  RMS reprojection error < 0.5 px  — excellent
  If the error is high, collect more pairs from varied angles.
"""

import cv2
import glob
import os
import sys
import numpy as np
from config import (
    CHESSBOARD_SIZE,
    SQUARE_SIZE_MM,
    CALIBRATION_IMAGES_FOLDER,
    CALIBRATION_DATA_FILE,
)

# ── Sub-pixel corner refinement criteria ────────────────────────────────────
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50, 1e-6
)

# ── Stereo calibration termination criteria ──────────────────────────────────
STEREO_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100, 1e-6
)


def build_object_points(board_size: tuple, square_mm: float) -> np.ndarray:
    """
    Build the 3-D coordinates of the chessboard corners in the board's own
    coordinate frame (Z=0 for all corners).
    """
    pts = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    pts[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    pts *= square_mm
    return pts


def load_image_pairs(folder: str) -> list[tuple]:
    """
    Return sorted (left_path, right_path) pairs from the calibration folder.
    """
    left_files  = sorted(glob.glob(os.path.join(folder, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(folder, "right_*.png")))

    # Match by timestamp (filename is left_NNNN.png / right_NNNN.png)
    left_stems  = {os.path.basename(f).replace("left_",  ""): f for f in left_files}
    right_stems = {os.path.basename(f).replace("right_", ""): f for f in right_files}
    common      = sorted(set(left_stems) & set(right_stems))

    if not common:
        raise FileNotFoundError(
            f"No matching left_*/right_* image pairs found in '{folder}'. "
            "Run Stage 1 first."
        )

    return [(left_stems[k], right_stems[k]) for k in common]


def detect_corners(pair: tuple, board_size: tuple) -> tuple | None:
    """
    Find and sub-pixel refine checkerboard corners in both images of a pair.
    Returns (gray_l, gray_r, corners_l, corners_r) or None if not found.
    """
    path_l, path_r = pair
    img_l = cv2.imread(path_l)
    img_r = cv2.imread(path_r)

    if img_l is None or img_r is None:
        print(f"  [WARN] Could not read: {path_l} or {path_r}")
        return None

    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE)

    ok_l, c_l = cv2.findChessboardCorners(gray_l, board_size, flags)
    ok_r, c_r = cv2.findChessboardCorners(gray_r, board_size, flags)

    if not (ok_l and ok_r):
        return None

    cv2.cornerSubPix(gray_l, c_l, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    cv2.cornerSubPix(gray_r, c_r, (11, 11), (-1, -1), SUBPIX_CRITERIA)

    return gray_l, gray_r, c_l, c_r


def show_rectified_preview(img_l: np.ndarray, img_r: np.ndarray,
                            map1_l, map2_l, map1_r, map2_r,
                            roi1: tuple, roi2: tuple) -> None:
    """
    Warp one example image pair with the calibration maps and display
    side-by-side with horizontal epipolar lines as a visual sanity check.
    """
    rect_l = cv2.remap(img_l, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map1_r, map2_r, cv2.INTER_LINEAR)

    combined = cv2.hconcat([rect_l, rect_r])

    # Draw horizontal epipolar lines every 60 pixels
    h = combined.shape[0]
    for y in range(0, h, 60):
        cv2.line(combined, (0, y), (combined.shape[1], y),
                 (0, 255, 0), 1)

    # Draw ROI rectangles
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    cv2.rectangle(combined, (x1, y1), (x1 + w1, y1 + h1), (255, 80, 0), 2)
    # right image ROI is offset by the width of the left image
    offset = rect_l.shape[1]
    cv2.rectangle(combined, (offset + x2, y2),
                  (offset + x2 + w2, y2 + h2), (255, 80, 0), 2)

    cv2.putText(combined,
                "Rectified preview (green lines should be horizontal). Press any key.",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    # Downscale if necessary
    if combined.shape[1] > 1920:
        scale    = 1920 / combined.shape[1]
        combined = cv2.resize(combined, None, fx=scale, fy=scale)

    cv2.imshow("Rectification Preview (press any key to close)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    # Ensure relative paths resolve from the script's own directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  STEREO CALIBRATION — Stage 2")
    print("=" * 60)

    # ── 1. Load image pairs ──────────────────────────────────────────────
    print(f"\n[1/5] Loading image pairs from '{CALIBRATION_IMAGES_FOLDER}/' …")
    pairs = load_image_pairs(CALIBRATION_IMAGES_FOLDER)
    print(f"      Found {len(pairs)} pairs.")

    # ── 2. Detect corners ────────────────────────────────────────────────
    print(f"\n[2/5] Detecting checkerboard corners ({CHESSBOARD_SIZE}) …")
    obj_pts_template = build_object_points(CHESSBOARD_SIZE, SQUARE_SIZE_MM)

    obj_pts_all  = []   # 3-D world points (same for every pair)
    img_pts_l    = []   # 2-D corners in left images
    img_pts_r    = []   # 2-D corners in right images
    image_size   = None
    good_count   = 0

    for i, pair in enumerate(pairs):
        result = detect_corners(pair, CHESSBOARD_SIZE)
        if result is None:
            print(f"  [SKIP] Pair {i+1:02d}: corners not found in both views.")
            continue

        gray_l, gray_r, c_l, c_r = result
        obj_pts_all.append(obj_pts_template)
        img_pts_l.append(c_l)
        img_pts_r.append(c_r)
        good_count += 1

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

    print(f"      {good_count}/{len(pairs)} pairs usable.")

    if good_count < 10:
        print("\n[ERROR] Need at least 10 usable pairs. "
              "Capture more images and re-run.")
        sys.exit(1)

    # ── 3. Individual camera calibration (good initial guess) ────────────
    print("\n[3/5] Calibrating individual cameras …")
    flags_mono = (cv2.CALIB_RATIONAL_MODEL)

    rms_l, K1, D1, _, _ = cv2.calibrateCamera(
        obj_pts_all, img_pts_l, image_size, None, None,
        flags=flags_mono
    )
    rms_r, K2, D2, _, _ = cv2.calibrateCamera(
        obj_pts_all, img_pts_r, image_size, None, None,
        flags=flags_mono
    )
    print(f"      Left  camera RMS: {rms_l:.4f} px")
    print(f"      Right camera RMS: {rms_r:.4f} px")

    # ── 4. Stereo calibration ────────────────────────────────────────────
    print("\n[4/5] Running stereo calibration …")

    stereo_flags = (
        0  # Allow everything to be optimized together
    )

    rms_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_pts_all, img_pts_l, img_pts_r,
        K1, D1, K2, D2,
        image_size,
        criteria=STEREO_CRITERIA,
        flags=stereo_flags,
    )

    print(f"      Stereo RMS reprojection error: {rms_stereo:.4f} px", end="  ")
    if rms_stereo < 0.5:
        print("← EXCELLENT ✓")
    elif rms_stereo < 1.0:
        print("← Good ✓")
    else:
        print("← High — consider recapturing with more varied angles.")

    # ── 5. Stereo rectification ──────────────────────────────────────────
    print("\n[5/5] Computing rectification maps …")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2,
        image_size, R, T,
        alpha=0,            # alpha=0: crop to only valid pixels (no black borders)
        newImageSize=image_size,
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    # ── Save ─────────────────────────────────────────────────────────────
    np.savez_compressed(
        CALIBRATION_DATA_FILE,
        K1=K1, D1=D1, K2=K2, D2=D2,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        map1_l=map1_l, map2_l=map2_l,
        map1_r=map1_r, map2_r=map2_r,
        image_size=image_size,
        rms=rms_stereo,
    )
    print(f"\n[SAVED] Calibration data → '{CALIBRATION_DATA_FILE}'")

    # ── Baseline info ────────────────────────────────────────────────────
    baseline_mm = np.linalg.norm(T)
    focal_px    = P1[0, 0]          # fx from left projection matrix
    print(f"\n  Baseline   : {baseline_mm:.1f} mm")
    print(f"  Focal len  : {focal_px:.1f} px  (left camera)")
    print(f"  Image size : {image_size[0]} × {image_size[1]} px")

    # ── Rectified preview ────────────────────────────────────────────────
    print("\n[PREVIEW] Showing rectified example pair. "
          "Epipolar lines should be perfectly horizontal.")

    # Use the first usable pair for the preview
    for pair in pairs:
        result = detect_corners(pair, CHESSBOARD_SIZE)
        if result is not None:
            img_l = cv2.imread(pair[0])
            img_r = cv2.imread(pair[1])
            show_rectified_preview(img_l, img_r,
                                   map1_l, map2_l, map1_r, map2_r,
                                   tuple(roi1), tuple(roi2))
            break

    print("\n[DONE] Calibration complete.")
    print("[NEXT] Run:  python 03_depth_map_live.py")


if __name__ == "__main__":
    main()
