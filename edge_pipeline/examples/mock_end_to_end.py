#!/usr/bin/env python3
# =============================================================================
# mock_end_to_end.py — Full pipeline demo with mock camera and actuator
# =============================================================================
# Purpose:  Demonstrate the complete edge pipeline:
#   MockCamera → transform → IK → plan → smooth → trajectory.csv
#
# This script runs without any hardware and produces a MATLAB-compatible CSV.
#
# Usage:
#   python examples/mock_end_to_end.py
#   python examples/mock_end_to_end.py --dof 4 --output my_trajectory.csv
# =============================================================================
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera.mock_camera import MockCamera
from src.camera.camera_interface import Detection
from src.collision.checker import CollisionChecker, Obstacle
from src.control.mock_actuator import MockActuator
from src.export.csv_export import write_trajectory_csv
from src.export.json_export import write_trajectory_json
from src.kinematics.arm_kinematics import ArmKinematics
from src.planning.planner import Planner, PlannerConfig, PlannerMode, PlanningFailure
from src.planning.trajectory import smooth_trajectory, time_parameterize
from src.transform.transforms import pixel_depth_to_camera, camera_to_base

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge Pipeline — Mock End-to-End Demo")
    parser.add_argument("--dof", type=int, default=3, choices=[3, 4])
    parser.add_argument("--output", type=str, default="output/trajectory.csv")
    parser.add_argument("--planner", type=str, default="rrt_star", choices=["apf", "rrt_star"])
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Setup ────────────────────────────────────────────────────────────
    logger.info("=== Mock End-to-End Pipeline Demo (DOF=%d) ===", args.dof)

    # Camera intrinsics (typical 720p)
    K = np.array([
        [600.0, 0.0, 640.0],
        [0.0, 600.0, 360.0],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)

    # Camera-to-base transform (ASSUMPTION: camera is 0.3 m above base, looking down)
    T_base_camera = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0,  0.3],
        [0.0,  0.0,  0.0,  1.0],
    ], dtype=np.float64)

    # Arm kinematics
    arm = ArmKinematics(dof=args.dof)
    logger.info("Arm workspace radius: %.3f m", arm.workspace_radius())

    # Collision checker
    def fk_frames(q):
        frames = [np.eye(4)]
        for i in range(arm.dof):
            q_partial = np.zeros(arm.dof)
            q_partial[:i + 1] = q[:i + 1]
            frames.append(arm.fk(q_partial))
        return frames

    cc = CollisionChecker(
        link_radii=[0.02] * arm.dof,
        fk_frames=fk_frames,
    )

    # Planner
    planner_mode = PlannerMode.APF if args.planner == "apf" else PlannerMode.RRT_STAR
    planner = Planner(
        dof=args.dof,
        joint_limits=arm.joint_limits,
        collision_checker=cc,
        config=PlannerConfig(mode=planner_mode, max_iterations=5000, max_time_s=3.0),
    )

    # Mock actuator
    actuator = MockActuator(dof=args.dof)
    actuator.enable()

    # Mock camera with custom detections
    detections = [
        Detection(label="bolt", u=640, v=360, depth_m=0.25, confidence=0.92),
        Detection(label="nut", u=500, v=300, depth_m=0.20, confidence=0.88),
    ]
    camera = MockCamera(detections=detections, fps=15.0)

    # ── Pipeline ─────────────────────────────────────────────────────────
    frame = camera.get_frame()
    all_trajectories = []
    current_q = np.zeros(args.dof)

    for det in frame.detections:
        logger.info("--- Processing detection: %s (u=%.0f, v=%.0f, z=%.3f m) ---",
                     det.label, det.u, det.v, det.depth_m)

        # 1. Transform
        t0 = time.perf_counter()
        point_cam = pixel_depth_to_camera(det.u, det.v, det.depth_m, K)
        point_base = camera_to_base(point_cam, T_base_camera)
        t_transform = (time.perf_counter() - t0) * 1000
        logger.info("  Transform: camera=%s → base=%s  (%.2f ms)",
                     np.round(point_cam, 4), np.round(point_base, 4), t_transform)

        # 2. IK
        t0 = time.perf_counter()
        q_goal, ik_info = arm.ik(point_base, q0=current_q)
        t_ik = (time.perf_counter() - t0) * 1000
        if q_goal is None:
            logger.warning("  IK FAILED: %s — skipping", ik_info)
            continue
        logger.info("  IK: q=%s  error=%.4f m  (%.2f ms)",
                     np.round(q_goal, 4), ik_info.get("error_m", -1), t_ik)

        # 3. Planning
        t0 = time.perf_counter()
        try:
            path = planner.plan(current_q, q_goal, [])
            t_plan = (time.perf_counter() - t0) * 1000
            logger.info("  Plan: %d waypoints  (%.2f ms)", len(path), t_plan)
        except PlanningFailure as e:
            logger.warning("  Planning FAILED: %s", e)
            continue

        # 4. Smoothing
        t0 = time.perf_counter()
        traj = smooth_trajectory(path, dt=0.02)
        traj, timestamps = time_parameterize(traj, max_vel=2.0, max_acc=5.0)
        t_smooth = (time.perf_counter() - t0) * 1000
        logger.info("  Smooth: %d samples, %.2f s duration  (%.2f ms)",
                     traj.shape[0], timestamps[-1], t_smooth)

        # 5. Execute (mock)
        for angles in traj:
            actuator.set_angles(angles)

        current_q = traj[-1].copy()
        all_trajectories.append((traj, timestamps, det))

    # ── Export ────────────────────────────────────────────────────────────
    if all_trajectories:
        # Concatenate all trajectories
        all_traj = np.vstack([t[0] for t in all_trajectories])
        # Rebuild timestamps (cumulative)
        cumulative_t = []
        offset = 0.0
        for traj, ts, _ in all_trajectories:
            cumulative_t.append(ts + offset)
            offset = cumulative_t[-1][-1] + 0.02
        all_ts = np.concatenate(cumulative_t)

        csv_path = write_trajectory_csv(output_path, all_traj, all_ts)
        json_path = write_trajectory_json(
            output_path.with_suffix(".json"), all_traj, all_ts,
            metadata={"source": "mock_end_to_end", "dof": args.dof},
        )

        logger.info("=== Output Files ===")
        logger.info("  CSV:  %s  (%d points × %d DOF)", csv_path, all_traj.shape[0], all_traj.shape[1])
        logger.info("  JSON: %s", json_path)
    else:
        logger.warning("No trajectories produced!")

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=== Demo Complete ===")
    logger.info("  Mock actuator received %d commands", len(actuator.command_log))
    camera.release()
    actuator.close()


if __name__ == "__main__":
    main()
