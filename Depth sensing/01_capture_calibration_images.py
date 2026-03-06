"""
01_capture_calibration_images.py
=================================
Stage 1 of the Stereo Depth Sensing Pipeline.

PURPOSE
-------
Captures synchronized image PAIRS from both cameras for use in stereo
calibration (Stage 2). Run this script and collect 20-30 good pairs showing
the checkerboard from varied angles, distances, and tilts.

CONTROLS
--------
  S   — Save the current frame pair (works regardless of board detection)
  A   — Auto-save ONLY when the checkerboard is clearly visible in BOTH views
  Q / ESC — Quit

HOW TO USE
----------
1. Print the checkerboard pattern and glue it flat to stiff cardboard.
2. Run this script. Two windows open (left + right camera side by side).
3. Hold the board in view of both cameras. Wait for GREEN corners to appear.
4. Press 'A' (or 'S') to save a pair. Capture 20-30 pairs.
5. Move on to Stage 2: python 02_stereo_calibrate.py
"""

import cv2
import os
import time
import threading
import numpy as np
from config import (
    LEFT_CAM_ID, RIGHT_CAM_ID,
    FRAME_WIDTH, FRAME_HEIGHT,
    CHESSBOARD_SIZE,
    CALIBRATION_IMAGES_FOLDER,
)

# ── Sub-pixel refinement criteria (used when drawing accurate corners) ──────
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30, 0.001
)

# Minimum seconds between auto-saves (prevents burst duplicates)
AUTO_SAVE_COOLDOWN = 1.5


# ── Threaded Stereo Camera Capture ──────────────────────────────────────────
class StereoStream:
    """
    Reads both cameras in a dedicated background thread so the main loop
    never blocks on USB I/O.  Uses MJPG codec + buffer size 1 to ensure
    fresh frames at full framerate.
    """

    def __init__(self, src_left: int, src_right: int,
                 width: int, height: int, fps: int = 30):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

        self.cam_left  = cv2.VideoCapture(src_left,  cv2.CAP_DSHOW)
        self.cam_right = cv2.VideoCapture(src_right, cv2.CAP_DSHOW)

        for cam, label in ((self.cam_left, "LEFT"), (self.cam_right, "RIGHT")):
            if not cam.isOpened():
                raise IOError(f"Cannot open {label} camera. "
                              "Check connection and camera ID in config.py.")
            cam.set(cv2.CAP_PROP_FOURCC, fourcc)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cam.set(cv2.CAP_PROP_FPS, fps)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep only the latest frame

        # Let sensors settle
        time.sleep(0.5)

        self.frame_left  = None
        self.frame_right = None
        self.stopped     = False
        self._lock       = threading.Lock()

    def start(self) -> "StereoStream":
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            # Grab both cameras as close together in time as possible
            self.cam_left.grab()
            self.cam_right.grab()

            ret_l, frame_l = self.cam_left.retrieve()
            ret_r, frame_r = self.cam_right.retrieve()

            if ret_l and ret_r:
                with self._lock:
                    self.frame_left  = frame_l
                    self.frame_right = frame_r

    def read(self):
        """Return the latest (left, right) frame pair (copies)."""
        with self._lock:
            l = self.frame_left.copy()  if self.frame_left  is not None else None
            r = self.frame_right.copy() if self.frame_right is not None else None
        return l, r

    def stop(self):
        self.stopped = True
        self.cam_left.release()
        self.cam_right.release()


def draw_hud(frame: np.ndarray, label: str, found: bool,
             fps: float, pairs_saved: int) -> np.ndarray:
    """
    Overlay HUD information on a copy of the frame:
      • Coloured status badge (GREEN = board found, RED = not found)
      • Camera label
      • FPS counter
      • Number of pairs saved so far
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Status badge
    badge_color = (0, 200, 0) if found else (0, 0, 220)
    cv2.circle(out, (28, 28), 14, badge_color, -1)
    cv2.circle(out, (28, 28), 14, (255, 255, 255), 2)

    # Camera label (top-left)
    cv2.putText(out, label, (50, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # FPS counter (top-right)
    fps_text = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(out, fps_text, (w - tw - 10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Saved pairs counter (bottom-left)
    saved_text = f"Saved: {pairs_saved} pairs"
    cv2.putText(out, saved_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)

    return out


def main() -> None:
    # ── Setup ───────────────────────────────────────────────────────────────
    os.makedirs(CALIBRATION_IMAGES_FOLDER, exist_ok=True)
    print(f"[INFO] Images will be saved to: '{CALIBRATION_IMAGES_FOLDER}/'")

    print("[INFO] Opening cameras … (this may take a few seconds)")
    stream = StereoStream(LEFT_CAM_ID, RIGHT_CAM_ID,
                          FRAME_WIDTH, FRAME_HEIGHT, fps=30).start()
    print("[INFO] Both cameras opened — threaded capture running.")

    # Print actual resolutions
    actual_w_l = int(stream.cam_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h_l = int(stream.cam_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_w_r = int(stream.cam_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h_r = int(stream.cam_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Left  camera actual resolution : {actual_w_l}\u00d7{actual_h_l}")
    print(f"[INFO] Right camera actual resolution : {actual_w_r}\u00d7{actual_h_r}\n")

    print("─" * 52)
    print("  STEREO CALIBRATION — IMAGE CAPTURE")
    print("─" * 52)
    print(f"  Checkerboard inner corners : {CHESSBOARD_SIZE}")
    print(f"  Resolution                 : {FRAME_WIDTH}×{FRAME_HEIGHT}")
    print()
    print("  S        → Save current pair (always)")
    print("  A        → Auto-save ONLY when board detected in BOTH")
    print("  Q / ESC  → Quit")
    print()
    print("  Aim for 20-30 pairs. Vary angle, distance, and tilt!")
    print("─" * 52)

    # ── CLAHE for contrast enhancement (improves detection in uneven lighting) ─
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    pairs_saved     = 0
    last_auto_save  = 0.0
    fps             = 0.0
    prev_tick       = cv2.getTickCount()

    WINDOW_TITLE = "Stereo Capture  |  S=save  A=auto-save  Q=quit"
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    # 2:1 aspect ratio window for side-by-side 1280x720 panels
    cv2.resizeWindow(WINDOW_TITLE, 1280, 640)

    while True:
        # ── Compute FPS ─────────────────────────────────────────────────────
        curr_tick = cv2.getTickCount()
        dt = (curr_tick - prev_tick) / cv2.getTickFrequency()
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        prev_tick = curr_tick

        # ── Read latest frames from background thread ───────────────────────
        frame_l, frame_r = stream.read()
        if frame_l is None or frame_r is None:
            continue

        # Normalize hardware input to configured resolution
        frame_l = cv2.resize(frame_l, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_r = cv2.resize(frame_r, (FRAME_WIDTH, FRAME_HEIGHT))

        # ── Chessboard Detection (full resolution + CLAHE) ──────────────────
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

        # CLAHE equalisation dramatically improves detection under uneven
        # or dim lighting by boosting local contrast on the checkerboard.
        gray_l_eq = clahe.apply(gray_l)
        gray_r_eq = clahe.apply(gray_r)

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        found_l, corners_l = cv2.findChessboardCorners(gray_l_eq, CHESSBOARD_SIZE, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r_eq, CHESSBOARD_SIZE, flags)

        both_found = found_l and found_r

        # ── Build display frames ─────────────────────────────────────────────
        disp_l = draw_hud(frame_l, "LEFT",  found_l, fps, pairs_saved)
        disp_r = draw_hud(frame_r, "RIGHT", found_r, fps, pairs_saved)

        # Sub-pixel refine corners on the original grayscale (already full-res)
        if found_l:
            cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            cv2.drawChessboardCorners(disp_l, CHESSBOARD_SIZE, corners_l, found_l)

        if found_r:
            cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            cv2.drawChessboardCorners(disp_r, CHESSBOARD_SIZE, corners_r, found_r)

        if both_found:
            cv2.putText(disp_l, "BOARD FOUND \u2014 Press A to save",
                        (10, FRAME_HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 255), 2, cv2.LINE_AA)

        # ── Combine and show ─────────────────────────────────────────────────
        combined = cv2.hconcat([disp_l, disp_r])
        cv2.imshow(WINDOW_TITLE, combined)

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        save_now = False

        if key in (ord('q'), 27):   # Q or ESC → quit
            break

        elif key == ord('s'):       # Manual save
            save_now = True

        elif key == ord('a'):       # Auto-save when board detected in both
            now = time.time()
            if both_found and (now - last_auto_save) >= AUTO_SAVE_COOLDOWN:
                save_now      = True
                last_auto_save = now
            elif not both_found:
                print(f"[WARN] Auto-save skipped: Board detected in LEFT={found_l}, RIGHT={found_r}")

        if save_now:
            ts = int(time.time() * 1000)   # millisecond timestamp → unique filenames
            path_l = os.path.join(CALIBRATION_IMAGES_FOLDER, f"left_{ts:015d}.png")
            path_r = os.path.join(CALIBRATION_IMAGES_FOLDER, f"right_{ts:015d}.png")
            cv2.imwrite(path_l, frame_l)   # Save the raw (no overlay) frames
            cv2.imwrite(path_r, frame_r)
            pairs_saved += 1
            status = "✓ board detected" if both_found else "⚠ board NOT detected"
            print(f"[{pairs_saved:02d}] Saved pair  ({status})")

    # ── Clean up ─────────────────────────────────────────────────────────────
    stream.stop()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Captured {pairs_saved} synchronized pairs in '{CALIBRATION_IMAGES_FOLDER}/'.")
    if pairs_saved < 20:
        print("[HINT] Aim for at least 20 pairs for reliable calibration.")
    print("[NEXT] Run:  python 02_stereo_calibrate.py")


if __name__ == "__main__":
    main()
