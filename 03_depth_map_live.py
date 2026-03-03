"""
03_depth_map_live.py
=====================
Stage 3 of the Stereo Depth Sensing Pipeline — Live Depth Sensing.

PURPOSE
-------
Loads the calibration data produced by Stage 2, rectifies live camera frames,
computes a dense disparity map using Semi-Global Block Matching (SGBM), and
reprojects every pixel into real-world 3-D coordinates (X, Y, Z in mm).

WINDOWS
-------
  "Rectified Stereo"  — Left and right images after rectification + epipolar lines
  "Depth Map"         — Colourised disparity (JET: blue=far, red=close)
                        Hover the mouse to see the 3-D coordinate at that pixel.

CONTROLS
--------
  Trackbars  — Tune SGBM parameters live
  Q / ESC    — Quit

PREREQUISITES
-------------
  • stereo_calibration.npz must exist (run Stage 2 first)
  • opencv-contrib-python installed (needed for WLS filter)
"""

import cv2
import sys
import time
import threading
import numpy as np
from config import (
    LEFT_CAM_ID, RIGHT_CAM_ID,
    FRAME_WIDTH, FRAME_HEIGHT,
    CALIBRATION_DATA_FILE,
    SGBM_NUM_DISPARITIES, SGBM_BLOCK_SIZE, SGBM_MIN_DISPARITY,
    SGBM_UNIQUENESS_RATIO, SGBM_SPECKLE_WIN_SIZE, SGBM_SPECKLE_RANGE,
    SGBM_DISP12_MAX_DIFF, WLS_LAMBDA, WLS_SIGMA,
)

# ── Window names ─────────────────────────────────────────────────────────────
WIN_STEREO = "Rectified Stereo  |  Q=quit"
WIN_DEPTH  = "Depth Map  |  Hover for 3-D coords  |  Q=quit"
WIN_TUNER  = "SGBM Tuner"

# ── Global mouse state ────────────────────────────────────────────────────────
_mouse_x, _mouse_y = 0, 0


def _on_mouse(event, x, y, flags, param):
    global _mouse_x, _mouse_y
    _mouse_x, _mouse_y = x, y


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
                print(f"[ERROR] Cannot open {label} camera.")
                sys.exit(1)
            cam.set(cv2.CAP_PROP_FOURCC, fourcc)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cam.set(cv2.CAP_PROP_FPS, fps)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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


def load_calibration(path: str) -> dict:
    """Load the .npz calibration file and return a dict of arrays."""
    if not __import__("os").path.exists(path):
        print(f"[ERROR] Calibration file not found: '{path}'.")
        print("        Run Stage 2 first:  python 02_stereo_calibrate.py")
        sys.exit(1)

    data = np.load(path)
    print(f"[INFO] Loaded calibration from '{path}'")
    print(f"       Stored RMS: {float(data['rms']):.4f} px")
    return data


def create_sgbm(num_disp: int, block: int,
                uniq: int, speckle_win: int,
                speckle_rng: int) -> cv2.StereoSGBM:
    """Construct an SGBM matcher with the given parameters."""
    # P1 / P2 control smoothness; rule-of-thumb: P1 = 8*cn*bs², P2 = 32*cn*bs²
    cn    = 1          # grayscale = 1 channel
    p1    = 8  * cn * block * block
    p2    = 32 * cn * block * block

    matcher = cv2.StereoSGBM.create(
        minDisparity      = SGBM_MIN_DISPARITY,
        numDisparities    = num_disp,
        blockSize         = block,
        P1                = p1,
        P2                = p2,
        disp12MaxDiff     = SGBM_DISP12_MAX_DIFF,
        uniquenessRatio   = uniq,
        speckleWindowSize = speckle_win,
        speckleRange      = speckle_rng,
        preFilterCap      = 63,
        mode              = cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher


def create_trackbar_window() -> None:
    """Create the SGBM tuner window with all trackbars."""
    cv2.namedWindow(WIN_TUNER, cv2.WINDOW_NORMAL)

    def _add(name: str, val: int, max_val: int):
        cv2.createTrackbar(name, WIN_TUNER, val, max_val, lambda v: None)

    # numDisparities must be a multiple of 16; store as multiple → 1=16, 2=32 …
    _add("numDisp x16",    SGBM_NUM_DISPARITIES // 16, 16)    # 16–256
    _add("BlockSize (odd)", (SGBM_BLOCK_SIZE - 5) // 2,  23)  # 5–51 step 2
    _add("UniqRatio",       SGBM_UNIQUENESS_RATIO,       20)
    _add("SpeckleWin",      SGBM_SPECKLE_WIN_SIZE,       200)
    _add("SpeckleRange",    SGBM_SPECKLE_RANGE,          50)

    # WLS filter params
    _add("WLS Lambda",      int(WLS_LAMBDA // 100),      200)  # 0 to 20,000
    _add("WLS Sigma x10",   int(WLS_SIGMA * 10),         50)   # 0.0 to 5.0


def read_trackbars() -> tuple:
    """Read current trackbar values and convert to valid SGBM params."""
    num_disp_mult = max(1, cv2.getTrackbarPos("numDisp x16", WIN_TUNER))
    num_disp      = num_disp_mult * 16

    block_raw     = cv2.getTrackbarPos("BlockSize (odd)", WIN_TUNER)
    block         = block_raw * 2 + 5          # maps 0..23 → 5,7,9…51
    block         = max(5, block)

    uniq          = cv2.getTrackbarPos("UniqRatio",    WIN_TUNER)
    speckle_win   = cv2.getTrackbarPos("SpeckleWin",   WIN_TUNER)
    speckle_rng   = cv2.getTrackbarPos("SpeckleRange", WIN_TUNER)

    wls_lambda    = cv2.getTrackbarPos("WLS Lambda",    WIN_TUNER) * 100
    wls_sigma     = cv2.getTrackbarPos("WLS Sigma x10", WIN_TUNER) / 10.0

    return num_disp, block, uniq, speckle_win, speckle_rng, wls_lambda, wls_sigma


def colorize_disparity(disp_float: np.ndarray,
                        num_disp: int) -> np.ndarray:
    """
    Normalise and apply JET colourmap to a float disparity image.
    Invalid pixels (disp ≤ 0) are painted black.
    """
    valid_mask      = disp_float > 0
    colour          = np.zeros((*disp_float.shape, 3), dtype=np.uint8)

    if valid_mask.any():
        disp_norm   = np.clip(disp_float / num_disp, 0, 1)
        jet_8u      = (disp_norm * 255).astype(np.uint8)
        jet_colour  = cv2.applyColorMap(jet_8u, cv2.COLORMAP_JET)
        colour[valid_mask] = jet_colour[valid_mask]

    return colour


def draw_depth_hud(depth_img: np.ndarray, pts_3d: np.ndarray,
                   mx: int, my: int, fps: float) -> np.ndarray:
    """
    Overlay FPS and the 3-D world coordinate under the mouse cursor onto
    the colourised depth image.
    """
    out = depth_img.copy()
    h, w = out.shape[:2]

    # FPS
    cv2.putText(out, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    # Clamp mouse coords to image bounds
    mx_c = max(0, min(mx, w - 1))
    my_c = max(0, min(my, h - 1))

    # Draw crosshair at mouse position
    cv2.drawMarker(out, (mx_c, my_c), (0, 255, 255),
                   cv2.MARKER_CROSS, 20, 2)

    # Read 3-D point
    if pts_3d is not None and pts_3d.shape[:2] == (h, w):
        X, Y, Z = pts_3d[my_c, mx_c]
        if np.isfinite(Z) and Z > 0:
            coord_text = f"X:{X:+6.0f}  Y:{Y:+6.0f}  Z:{Z:6.0f} mm"
        else:
            coord_text = "No depth at cursor"
        cv2.putText(out, coord_text, (10, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    return out


def draw_epipolar_lines(img: np.ndarray, step: int = 40) -> np.ndarray:
    """Draw thin horizontal lines to visualise epipolar alignment."""
    out = img.copy()
    for y in range(0, img.shape[0], step):
        cv2.line(out, (0, y), (img.shape[1], y), (0, 200, 0), 1)
    return out


def main() -> None:
    print("=" * 60)
    print("  LIVE DEPTH MAP — Stage 3")
    print("=" * 60)

    # ── Load calibration ─────────────────────────────────────────────────
    cal = load_calibration(CALIBRATION_DATA_FILE)
    map1_l   = cal["map1_l"]
    map2_l   = cal["map2_l"]
    map1_r   = cal["map1_r"]
    map2_r   = cal["map2_r"]
    Q        = cal["Q"]

    # ── Try to import WLS filter (requires opencv-contrib) ───────────────
    try:
        wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        use_wls = True
        print("[INFO] WLS disparity filter: ENABLED (opencv-contrib detected)")
    except AttributeError:
        use_wls = False
        print("[WARN] WLS filter unavailable. Install opencv-contrib-python for "
              "smoother depth maps.")

    # ── Open cameras (threaded) ──────────────────────────────────────────
    print("[INFO] Opening cameras …")
    stream = StereoStream(LEFT_CAM_ID, RIGHT_CAM_ID,
                          FRAME_WIDTH, FRAME_HEIGHT, fps=30).start()
    print("[INFO] Cameras ready — threaded capture running.")

    # ── Create windows ───────────────────────────────────────────────────
    cv2.namedWindow(WIN_STEREO, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DEPTH,  cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_DEPTH, _on_mouse)
    create_trackbar_window()

    # ── Initial matcher ───────────────────────────────────────────────────
    prev_params   = None
    matcher       = None

    fps            = 0.0
    prev_tick      = cv2.getTickCount()
    pts_3d         = None    # last computed 3-D points cloud

    print("\n[LIVE] Depth map running. Adjust trackbars to tune.")
    print("       Hover over the Depth Map window to see (X, Y, Z) in mm.")
    print("       Press  Q / ESC  to quit.\n")

    while True:
        # ── FPS ───────────────────────────────────────────────────────────
        curr_tick = cv2.getTickCount()
        dt = (curr_tick - prev_tick) / cv2.getTickFrequency()
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt
        prev_tick = curr_tick

        # ── Read latest frames from background thread ─────────────────────
        frame_l, frame_r = stream.read()
        if frame_l is None or frame_r is None:
            continue

        # ── Rectify ───────────────────────────────────────────────────────
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)

        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        # ── Read trackbars & rebuild matcher if params changed ────────────
        params = read_trackbars()
        if params != prev_params:
            num_disp, block, uniq, speckle_win, speckle_rng, w_lam, w_sig = params
            matcher   = create_sgbm(num_disp, block, uniq,
                                     speckle_win, speckle_rng)
            
            if use_wls:
                wls_filter.setLambda(w_lam)
                wls_filter.setSigmaColor(w_sig)
                
            prev_params = params
        else:
            num_disp = params[0]

        # ── Disparity computation ─────────────────────────────────────────
        disp_raw = matcher.compute(gray_l, gray_r)

        if use_wls:
            # WLS filter needs a right matcher
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            disp_raw_r    = right_matcher.compute(gray_r, gray_l)
            wls_filter.setDepthDiscontinuityRadius(disp_raw.shape[1] // 100)
            disp_filtered = wls_filter.filter(disp_raw, gray_l,
                                               disparity_map_right=disp_raw_r)
            disp_float = disp_filtered.astype(np.float32) / 16.0
        else:
            disp_float = disp_raw.astype(np.float32) / 16.0

        # Clamp negative (invalid) disparities to 0
        disp_float = np.clip(disp_float, 0, None)

        # ── Reproject to 3-D ─────────────────────────────────────────────
        pts_3d = cv2.reprojectImageTo3D(disp_float, Q, handleMissingValues=True)
        # Mask out-of-range / invalid points
        mask_invalid = (disp_float <= 0) | ~np.isfinite(pts_3d[:, :, 2])
        pts_3d[mask_invalid] = [0, 0, np.nan]

        # ── Colourised depth map ──────────────────────────────────────────
        depth_colour = colorize_disparity(disp_float, num_disp)
        depth_disp   = draw_depth_hud(depth_colour, pts_3d,
                                       _mouse_x, _mouse_y, fps)

        # ── Stereo side-by-side with epipolar lines ───────────────────────
        rect_l_lines = draw_epipolar_lines(rect_l)
        rect_r_lines = draw_epipolar_lines(rect_r)
        stereo_disp  = cv2.hconcat([rect_l_lines, rect_r_lines])

        # FPS on stereo window too
        cv2.putText(stereo_disp, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(stereo_disp,
                    f"numDisp={num_disp}  block={params[1]}  "
                    f"uniq={params[2]}  speckle={params[3]}/{params[4]}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Downscale if too wide
        if stereo_disp.shape[1] > 1920:
            s = 1920 / stereo_disp.shape[1]
            stereo_disp = cv2.resize(stereo_disp, None, fx=s, fy=s)

        # ── Show ──────────────────────────────────────────────────────────
        cv2.imshow(WIN_STEREO, stereo_disp)
        cv2.imshow(WIN_DEPTH,  depth_disp)

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    # ── Clean up ──────────────────────────────────────────────────────────
    stream.stop()
    cv2.destroyAllWindows()
    print("\n[DONE] Depth sensing session ended.")


if __name__ == "__main__":
    main()
