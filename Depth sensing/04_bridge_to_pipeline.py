"""
04_bridge_to_pipeline.py
=========================
Stage 4 of the Stereo Depth Sensing Pipeline — Bridge to Edge Pipeline.

PURPOSE
-------
Extends Stage 3 with arm trajectory planning. Shows:
  - Left camera feed (rectified)
  - Depth map with SGBM tuning trackbars + WLS filtering
  - Click on the depth map to select a 3D target
  - Edge Pipeline computes IK + path plan + smooth trajectory
  - Exports trajectory CSV / JSON for MATLAB or servo actuation

WINDOWS
-------
  "Camera + Depth"  — Left rectified image beside colourised depth map
  "SGBM Tuner"      — Live trackbars for SGBM + WLS parameters

CONTROLS
--------
  Left-click on depth map  — Select target, compute trajectory
  R                        — Retract arm to home position
  Q / ESC                  — Quit

MODES
-----
  --mode simulation   (default) Export CSV only
  --mode live         Send PWM to PCA9685 servos

USAGE
-----
  python 04_bridge_to_pipeline.py
  python 04_bridge_to_pipeline.py --mode live
"""

import argparse
import cv2
import json
import logging
import os
import sys
import time
import threading
import numpy as np
from pathlib import Path

# ── Import depth sensing config ─────────────────────────────────────────────
from config import (
    LEFT_CAM_ID, RIGHT_CAM_ID,
    FRAME_WIDTH, FRAME_HEIGHT,
    CALIBRATION_DATA_FILE,
    SGBM_NUM_DISPARITIES, SGBM_BLOCK_SIZE, SGBM_MIN_DISPARITY,
    SGBM_UNIQUENESS_RATIO, SGBM_SPECKLE_WIN_SIZE, SGBM_SPECKLE_RANGE,
    SGBM_DISP12_MAX_DIFF, WLS_LAMBDA, WLS_SIGMA,
)

# ── Add edge pipeline to path ───────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
EDGE_DIR = SCRIPT_DIR.parent / "edge_pipeline"
sys.path.insert(0, str(EDGE_DIR))

from src.kinematics.arm_kinematics import ArmKinematics
from src.collision.checker import CollisionChecker
from src.planning.planner import Planner, PlannerConfig, PlannerMode, PlanningFailure
from src.planning.trajectory import smooth_trajectory, time_parameterize
from src.export.csv_export import write_trajectory_csv
from src.export.json_export import write_trajectory_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Window names + Save file ─────────────────────────────────────────────────
WIN_MAIN  = "Camera + Depth  |  Click depth to set target  |  Q=quit"
WIN_TUNER = "SGBM Tuner"
SGBM_CONF_FILE = SCRIPT_DIR / "sgbm_config.json"

# ── Global mouse + click state ───────────────────────────────────────────────
_mouse_x, _mouse_y = 0, 0
_click_queue = []  # list of (x, y) clicks to process
_click_lock = threading.Lock()
_display_scale = 1.0  # updated each frame; maps display coords -> original coords


def _on_mouse(event, x, y, flags, param):
    global _mouse_x, _mouse_y
    _mouse_x, _mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        with _click_lock:
            _click_queue.append((x, y))


def _pop_click():
    with _click_lock:
        if _click_queue:
            return _click_queue.pop(0)
    return None


# =============================================================================
# Reuse proven Stage 3 components
# =============================================================================
class StereoStream:
    """Threaded stereo camera reader (identical to Stage 3)."""
    def __init__(self, src_left, src_right, width, height, fps=30):
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.cam_left  = cv2.VideoCapture(src_left,  cv2.CAP_DSHOW)
        self.cam_right = cv2.VideoCapture(src_right, cv2.CAP_DSHOW)

        for cam, label in ((self.cam_left, "LEFT"), (self.cam_right, "RIGHT")):
            if not cam.isOpened():
                print(f"[ERROR] Cannot open {label} camera.")
                sys.exit(1)
            cam.set(cv2.CAP_PROP_FOURCC, fourcc)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cam.set(cv2.CAP_PROP_FPS, fps)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        time.sleep(0.5)
        self.frame_left = None
        self.frame_right = None
        self.stopped = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            self.cam_left.grab()
            self.cam_right.grab()
            ret_l, fl = self.cam_left.retrieve()
            ret_r, fr = self.cam_right.retrieve()
            if ret_l and ret_r:
                with self._lock:
                    self.frame_left = fl
                    self.frame_right = fr

    def read(self):
        with self._lock:
            l = self.frame_left.copy()  if self.frame_left  is not None else None
            r = self.frame_right.copy() if self.frame_right is not None else None
        return l, r

    def stop(self):
        self.stopped = True
        self.cam_left.release()
        self.cam_right.release()


def create_sgbm(num_disp, block, uniq, speckle_win, speckle_rng):
    """Construct SGBM matcher."""
    cn = 1
    p1 = 8  * cn * block * block
    p2 = 32 * cn * block * block
    return cv2.StereoSGBM.create(
        minDisparity=SGBM_MIN_DISPARITY,
        numDisparities=num_disp,
        blockSize=block,
        P1=p1, P2=p2,
        disp12MaxDiff=SGBM_DISP12_MAX_DIFF,
        uniquenessRatio=uniq,
        speckleWindowSize=speckle_win,
        speckleRange=speckle_rng,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def load_sgbm_config():
    if SGBM_CONF_FILE.exists():
        try:
            with open(SGBM_CONF_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def save_sgbm_config(params):
    num_disp, block, uniq, sp_win, sp_rng, w_lam, w_sig = params
    config = {
        "num_disp": num_disp,
        "block": block,
        "uniq": uniq,
        "speckle_win": sp_win,
        "speckle_rng": sp_rng,
        "wls_lambda": w_lam,
        "wls_sigma": w_sig
    }
    with open(SGBM_CONF_FILE, 'w') as f:
        json.dump(config, f)


def create_trackbar_window(saved_conf=None):
    """Create SGBM + WLS tuner trackbars with optional saved values."""
    cv2.namedWindow(WIN_TUNER, cv2.WINDOW_NORMAL)
    def _add(name, val, max_val):
        cv2.createTrackbar(name, WIN_TUNER, val, max_val, lambda v: None)
    
    # Defaults
    d_nd = SGBM_NUM_DISPARITIES // 16
    d_bl = (SGBM_BLOCK_SIZE - 5) // 2
    d_uq = SGBM_UNIQUENESS_RATIO
    d_sw = SGBM_SPECKLE_WIN_SIZE
    d_sr = SGBM_SPECKLE_RANGE
    d_wl = int(WLS_LAMBDA // 100)
    d_ws = int(WLS_SIGMA * 10)

    # Load from config if available
    if saved_conf:
        d_nd = saved_conf.get("num_disp", SGBM_NUM_DISPARITIES) // 16
        d_bl = (saved_conf.get("block", SGBM_BLOCK_SIZE) - 5) // 2
        d_uq = saved_conf.get("uniq", SGBM_UNIQUENESS_RATIO)
        d_sw = saved_conf.get("speckle_win", SGBM_SPECKLE_WIN_SIZE)
        d_sr = saved_conf.get("speckle_rng", SGBM_SPECKLE_RANGE)
        d_wl = int(saved_conf.get("wls_lambda", WLS_LAMBDA) // 100)
        d_ws = int(saved_conf.get("wls_sigma", WLS_SIGMA) * 10)

    _add("numDisp x16",     d_nd, 16)
    _add("BlockSize (odd)", d_bl, 23)
    _add("UniqRatio",       d_uq, 20)
    _add("SpeckleWin",      d_sw, 200)
    _add("SpeckleRange",    d_sr, 50)
    _add("WLS Lambda",      d_wl, 200)
    _add("WLS Sigma x10",   d_ws, 50)


def read_trackbars():
    """Read trackbar values."""
    num_disp = max(1, cv2.getTrackbarPos("numDisp x16", WIN_TUNER)) * 16
    block    = cv2.getTrackbarPos("BlockSize (odd)", WIN_TUNER) * 2 + 5
    block    = max(5, block)
    uniq     = cv2.getTrackbarPos("UniqRatio",    WIN_TUNER)
    sp_win   = cv2.getTrackbarPos("SpeckleWin",   WIN_TUNER)
    sp_rng   = cv2.getTrackbarPos("SpeckleRange", WIN_TUNER)
    wls_lam  = cv2.getTrackbarPos("WLS Lambda",   WIN_TUNER) * 100
    wls_sig  = cv2.getTrackbarPos("WLS Sigma x10", WIN_TUNER) / 10.0
    return num_disp, block, uniq, sp_win, sp_rng, wls_lam, wls_sig


def colorize_disparity(disp_float, num_disp):
    """Colorize disparity with proper masking."""
    valid_mask = disp_float > 0
    colour = np.zeros((*disp_float.shape, 3), dtype=np.uint8)
    if valid_mask.any():
        disp_norm = np.clip(disp_float / num_disp, 0, 1)
        jet_8u = (disp_norm * 255).astype(np.uint8)
        jet_colour = cv2.applyColorMap(jet_8u, cv2.COLORMAP_JET)
        colour[valid_mask] = jet_colour[valid_mask]
    return colour


# =============================================================================
# Main
# =============================================================================
def main():
    os.chdir(SCRIPT_DIR)

    parser = argparse.ArgumentParser(description="Stage 4: Bridge Depth Sensing -> Edge Pipeline")
    parser.add_argument("--mode", default="simulation", choices=["simulation", "live"])
    parser.add_argument("--dof", type=int, default=3, choices=[3, 4])
    parser.add_argument("--output-dir", default=str(EDGE_DIR / "output" / "bridge"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  BRIDGE: Depth Sensing -> Edge Pipeline (Stage 4)")
    print(f"  Mode: {args.mode.upper()}  |  DOF: {args.dof}")
    print("=" * 60)

    # ── 1. Load calibration ──────────────────────────────────────────────────
    cal = np.load(CALIBRATION_DATA_FILE)
    map1_l = cal["map1_l"];  map2_l = cal["map2_l"]
    map1_r = cal["map1_r"];  map2_r = cal["map2_r"]
    Q = cal["Q"]
    print(f"[INFO] Loaded calibration (RMS: {float(cal['rms']):.4f} px)")

    # ── 2. WLS filter ────────────────────────────────────────────────────────
    try:
        wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
        wls_filter.setLambda(WLS_LAMBDA)
        wls_filter.setSigmaColor(WLS_SIGMA)
        use_wls = True
        print("[INFO] WLS filter: ENABLED")
    except AttributeError:
        use_wls = False
        print("[WARN] WLS filter unavailable (install opencv-contrib-python)")

    # ── 3. Setup edge pipeline ───────────────────────────────────────────────
    arm = ArmKinematics(dof=args.dof)

    def fk_frames(q):
        frames = [np.eye(4)]
        for i in range(arm.dof):
            q_partial = np.zeros(arm.dof)
            q_partial[:i + 1] = q[:i + 1]
            frames.append(arm.fk(q_partial))
        return frames

    cc = CollisionChecker([0.02] * arm.dof, fk_frames)
    planner = Planner(
        dof=args.dof, joint_limits=arm.joint_limits, collision_checker=cc,
        config=PlannerConfig(mode=PlannerMode.RRT_STAR, max_iterations=3000, max_time_s=2.0),
    )

    # =========================================================================
    # HARDWARE MOUNTING MATRIX (CRITICAL FOR IK SUCCESS)
    # =========================================================================
    # 
    # The previous matrix assumed the camera was pointing STRAIGHT DOWN at the 
    # table. An object 3m deep (Z=3) ended up 3m UNDERGROUND (Base Z = -3.0). 
    # IK failed because the arm cannot reach into the floor!
    #
    # This matrix assumes DESKTOP TESTING orientation:
    # The camera is sitting *behind* the robot arm, pointing FORWARD.
    #  - Camera +Z (forward depth)  => Base +X (forward)
    #  - Camera +X (right)          => Base -Y (left)
    #  - Camera +Y (down)           => Base -Z (up)
    #
    # Change the 0.10 and 0.15 values to roughly match your desktop setup!
    #
    T_base_camera = np.array([
        [ 0.0,  0.0,  1.0,  0.10], # Base X = Camera Z + 10cm offset
        [-1.0,  0.0,  0.0,  0.00], # Base Y = -Camera X
        [ 0.0, -1.0,  0.0,  0.15], # Base Z = -Camera Y + 15cm height
        [ 0.0,  0.0,  0.0,  1.00],
    ], dtype=np.float64)

    current_q = np.zeros(args.dof)
    traj_count = 0
    last_target_info = None  # for HUD display

    # ── 4. Setup actuator ────────────────────────────────────────────────────
    if args.mode == "live":
        try:
            from src.control.pca9685_driver import PCA9685Driver
            actuator = PCA9685Driver(dof=args.dof)
            actuator.enable()
            print("[INFO] PCA9685 ENABLED (live mode)")
        except Exception as e:
            print(f"[WARN] PCA9685 failed ({e}), using mock actuator")
            from src.control.mock_actuator import MockActuator
            actuator = MockActuator(dof=args.dof)
            actuator.enable()
    else:
        from src.control.mock_actuator import MockActuator
        actuator = MockActuator(dof=args.dof)
        actuator.enable()

    # ── 5. Open cameras ──────────────────────────────────────────────────────
    print("[INFO] Opening cameras...")
    stream = StereoStream(LEFT_CAM_ID, RIGHT_CAM_ID,
                          FRAME_WIDTH, FRAME_HEIGHT, fps=30).start()
    print("[INFO] Cameras ready.")

    # ── 6. Create windows ────────────────────────────────────────────────────
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_MAIN, _on_mouse)
    saved_conf = load_sgbm_config()
    create_trackbar_window(saved_conf)
    cv2.waitKey(1)

    prev_params = None
    matcher = None
    fps = 0.0
    prev_tick = cv2.getTickCount()
    pts_3d = None

    print("\n[LIVE] Bridge running.")
    print("       LEFT side = Camera feed  |  RIGHT side = Depth map")
    print("       Click on the DEPTH MAP (right side) to set target.")
    print("       Tune parameters with trackbars. (Will auto-save on quit)")
    print("       Press R = retract to home  |  Q/ESC = quit\n")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    while True:
        # FPS
        curr_tick = cv2.getTickCount()
        dt = (curr_tick - prev_tick) / cv2.getTickFrequency()
        if dt > 0:
            fps = 0.9 * fps + 0.1 / dt
        prev_tick = curr_tick

        # Read frames
        frame_l, frame_r = stream.read()
        if frame_l is None or frame_r is None:
            continue

        # Rectify
        rect_l = cv2.remap(frame_l, map1_l, map2_l, cv2.INTER_LINEAR)
        rect_r = cv2.remap(frame_r, map1_r, map2_r, cv2.INTER_LINEAR)
        gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

        # Read trackbars + rebuild matcher if changed
        params = read_trackbars()
        if params != prev_params:
            num_disp, block, uniq, sp_win, sp_rng, w_lam, w_sig = params
            matcher = create_sgbm(num_disp, block, uniq, sp_win, sp_rng)
            if use_wls:
                wls_filter.setLambda(w_lam)
                wls_filter.setSigmaColor(w_sig)
            prev_params = params
        else:
            num_disp = params[0]

        # Compute disparity
        disp_raw = matcher.compute(gray_l, gray_r)

        if use_wls:
            right_matcher = cv2.ximgproc.createRightMatcher(matcher)
            disp_raw_r = right_matcher.compute(gray_r, gray_l)
            wls_filter.setDepthDiscontinuityRadius(disp_raw.shape[1] // 100)
            disp_filtered = wls_filter.filter(disp_raw, gray_l,
                                              disparity_map_right=disp_raw_r)
            disp_float = disp_filtered.astype(np.float32) / 16.0
        else:
            disp_float = disp_raw.astype(np.float32) / 16.0

        disp_float = np.clip(disp_float, 0, None)

        # Reproject to 3D (X, Y, Z in mm)
        pts_3d = cv2.reprojectImageTo3D(disp_float, Q, handleMissingValues=True)
        mask_invalid = (disp_float <= 0) | ~np.isfinite(pts_3d[:, :, 2])
        pts_3d[mask_invalid] = [0, 0, np.nan]

        # ── Build display: [Camera Feed | Depth Map] ────────────────────────
        h, w = rect_l.shape[:2]

        # Left panel: camera feed with HUD
        cam_display = rect_l.copy()
        cv2.putText(cam_display, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(cam_display, f"Mode: {args.mode.upper()}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA)
        cv2.putText(cam_display, f"Trajectories: {traj_count}", (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        if last_target_info:
            info = last_target_info
            # Color coding success vs failure
            color_text = (100, 255, 100) if "OK" in info['label'] or "HOME" in info['label'] else (0, 0, 255)
            cv2.putText(cam_display, f"Last: {info['label']}", (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_text, 2, cv2.LINE_AA)
            cv2.putText(cam_display, f"  {info['detail']}", (10, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1, cv2.LINE_AA)

        # Right panel: depth map with crosshair + coords
        depth_colour = colorize_disparity(disp_float, num_disp)

        # ── Unscale mouse coords from display space to original image space ──
        # The display may be downscaled (e.g. 2560->1920),
        # so raw mouse coords must be divided by the scale factor.
        global _display_scale
        orig_mx = int(_mouse_x / _display_scale)
        orig_my = int(_mouse_y / _display_scale)

        # The depth map is the right half of the combined image
        depth_mx = orig_mx - w
        depth_my = orig_my
        depth_mx_c = max(0, min(depth_mx, w - 1))
        depth_my_c = max(0, min(depth_my, h - 1))

        # Draw crosshair on depth map if mouse is over it
        if 0 <= depth_mx < w and 0 <= depth_my < h:
            cv2.drawMarker(depth_colour, (depth_mx_c, depth_my_c),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
            # Show 3D coords at cursor
            if pts_3d is not None:
                X, Y, Z = pts_3d[depth_my_c, depth_mx_c]
                if np.isfinite(Z) and Z > 0:
                    txt = f"X:{X:+6.0f}  Y:{Y:+6.0f}  Z:{Z:6.0f} mm"
                else:
                    txt = "No depth at cursor"
                cv2.putText(depth_colour, txt, (10, h - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(depth_colour, "Click here to set target", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Combine side by side
        combined = cv2.hconcat([cam_display, depth_colour])

        # Downscale if too wide — store scale factor for mouse unscaling
        if combined.shape[1] > 1920:
            _display_scale = 1920 / combined.shape[1]
            combined = cv2.resize(combined, None, fx=_display_scale, fy=_display_scale)
        else:
            _display_scale = 1.0

        cv2.imshow(WIN_MAIN, combined)

        # ── Process clicks ───────────────────────────────────────────────────
        click = _pop_click()
        if click is not None:
            # Unscale click coords from display space to original image space
            cx = int(click[0] / _display_scale)
            cy = int(click[1] / _display_scale)
            # Check if click is on the depth map side (right half)
            if cx >= w:
                px = cx - w  # pixel in depth image
                py = cy
                px = max(0, min(px, w - 1))
                py = max(0, min(py, h - 1))

                if pts_3d is not None:
                    X_mm, Y_mm, Z_mm = pts_3d[py, px]

                    if np.isfinite(Z_mm) and Z_mm > 0:
                        # Convert mm -> meters
                        X_m = X_mm / 1000.0
                        Y_m = Y_mm / 1000.0
                        Z_m = Z_mm / 1000.0

                        # Camera -> Base frame
                        pt_h = np.array([X_m, Y_m, Z_m, 1.0])
                        pt_base = (T_base_camera @ pt_h)[:3]

                        logger.info("=" * 50)
                        logger.info("TARGET SELECTED")
                        logger.info("  Pixel:  (%d, %d)", px, py)
                        logger.info("  Camera: [%.4f, %.4f, %.4f] m", X_m, Y_m, Z_m)
                        logger.info("  Base:   [%.4f, %.4f, %.4f] m", *pt_base)

                        # Check workspace
                        dist = np.linalg.norm(pt_base)
                        ws_r = arm.workspace_radius()
                        if dist > ws_r:
                            logger.warning("  Target is %.3f m away but arm reach is %.3f m", dist, ws_r)
                            logger.warning("  Scaling target to workspace boundary...")
                            # Maintain direction direction relative to the base origin
                            pt_base = pt_base * (ws_r * 0.9 / dist)
                            logger.info("  Scaled base: [%.4f, %.4f, %.4f] m", *pt_base)
                            
                            # Double check if Z is still reachable (Z < 0 is underground)
                            if pt_base[2] < 0.05:
                                logger.warning("  Scaled target is underground or too close to floor! Clamping Z.")
                                pt_base[2] = 0.05

                        # IK
                        t0 = time.perf_counter()
                        q_goal, ik_info = arm.ik(pt_base, q0=current_q)
                        t_ik = (time.perf_counter() - t0) * 1000

                        if q_goal is None:
                            logger.warning("  IK FAILED (%.1f ms): %s", t_ik, ik_info)
                            last_target_info = {
                                "label": "IK FAILED",
                                "detail": f"Unreachable target ({dist:.2f}m)"
                            }
                            continue

                        logger.info("  IK OK: q=%s  (%.1f ms)", np.round(q_goal, 3), t_ik)

                        # Plan
                        t0 = time.perf_counter()
                        try:
                            path = planner.plan(current_q, q_goal, [])
                            t_plan = (time.perf_counter() - t0) * 1000
                            logger.info("  Plan: %d waypoints (%.1f ms)", len(path), t_plan)
                        except PlanningFailure as e:
                            logger.warning("  Planning FAILED: %s", e)
                            last_target_info = {"label": "PLAN FAILED", "detail": str(e)}
                            continue

                        # Smooth
                        traj = smooth_trajectory(path, dt=0.02)
                        traj, timestamps = time_parameterize(traj, max_vel=2.0, max_acc=5.0)
                        logger.info("  Trajectory: %d pts, %.2f s", traj.shape[0], timestamps[-1])

                        # Execute
                        if args.mode == "live" and actuator is not None:
                            logger.info("  EXECUTING on servos...")
                            for angles in traj:
                                actuator.set_angles(angles)
                                time.sleep(0.02)

                        # Export
                        traj_count += 1
                        csv_path = output_dir / f"traj_{traj_count:03d}.csv"
                        write_trajectory_csv(csv_path, traj, timestamps)
                        json_path = output_dir / f"traj_{traj_count:03d}.json"
                        write_trajectory_json(json_path, traj, timestamps, metadata={
                            "target_pixel": [px, py],
                            "target_camera_m": [X_m, Y_m, Z_m],
                            "target_base_m": pt_base.tolist(),
                            "ik_time_ms": t_ik,
                            "traj_points": int(traj.shape[0]),
                            "duration_s": float(timestamps[-1]),
                        })

                        current_q = traj[-1].copy()
                        last_target_info = {
                            "label": f"Traj #{traj_count} OK",
                            "detail": f"{traj.shape[0]} pts, {timestamps[-1]:.1f}s -> {csv_path.name}"
                        }
                        logger.info("  Saved: %s", csv_path)
                        logger.info("=" * 50)
                    else:
                        logger.warning("No valid depth at pixel (%d, %d)", px, py)
                        last_target_info = {"label": "NO DEPTH", "detail": f"Pixel ({px},{py}) has no depth data"}
            else:
                # Click was on camera side, ignore
                pass

        # ── Key handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break
        elif key in (ord('r'), ord('R')):
            logger.info("Retracting to home position...")
            try:
                path = planner.plan(current_q, np.zeros(args.dof), [])
                traj = smooth_trajectory(path, dt=0.02)
                traj, ts = time_parameterize(traj)
                if args.mode == "live" and actuator is not None:
                    for angles in traj:
                        actuator.set_angles(angles)
                        time.sleep(0.02)
                current_q = np.zeros(args.dof)
                logger.info("Retracted to home.")
                last_target_info = {"label": "HOME", "detail": "Retracted to home position"}
            except PlanningFailure:
                logger.warning("Retract planning failed!")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    stream.stop()
    if actuator is not None:
        actuator.close()
    cv2.destroyAllWindows()
    
    # Save SGBM config on exit!
    if prev_params:
        save_sgbm_config(prev_params)
        logger.info("Saved SGBM config to %s", SGBM_CONF_FILE.name)

    print(f"\n[DONE] Generated {traj_count} trajectories in: {output_dir}")
    if traj_count > 0:
        print(f"       Load in MATLAB:  data = readmatrix('traj_001.csv', 'CommentStyle', '%');")


if __name__ == "__main__":
    main()
