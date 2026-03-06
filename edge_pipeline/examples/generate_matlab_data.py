#!/usr/bin/env python3
# =============================================================================
# generate_matlab_data.py — Generate multiple trajectory scenarios for MATLAB
# =============================================================================
# Produces several CSV files with different motions for MATLAB simulation:
#   1. Single reach (home → target)
#   2. Multi-point pick sequence
#   3. Sweep motion (workspace scan)
#   4. Return-to-home with intermediate waypoints
# =============================================================================
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kinematics.arm_kinematics import ArmKinematics
from src.collision.checker import CollisionChecker
from src.planning.planner import Planner, PlannerConfig, PlannerMode, PlanningFailure
from src.planning.trajectory import smooth_trajectory, time_parameterize
from src.export.csv_export import write_trajectory_csv


def make_planner(arm: ArmKinematics) -> Planner:
    def fk_frames(q):
        frames = [np.eye(4)]
        for i in range(arm.dof):
            q_partial = np.zeros(arm.dof)
            q_partial[:i + 1] = q[:i + 1]
            frames.append(arm.fk(q_partial))
        return frames

    cc = CollisionChecker([0.02] * arm.dof, fk_frames)
    return Planner(
        dof=arm.dof,
        joint_limits=arm.joint_limits,
        collision_checker=cc,
        config=PlannerConfig(mode=PlannerMode.RRT_STAR, max_iterations=3000, max_time_s=2.0),
    )


def plan_and_smooth(planner, start, goal):
    path = planner.plan(start, goal, [])
    traj = smooth_trajectory(path, dt=0.02)
    traj, ts = time_parameterize(traj, max_vel=2.0, max_acc=5.0)
    return traj, ts


def main():
    output_dir = Path("output/matlab_scenarios")
    output_dir.mkdir(parents=True, exist_ok=True)

    arm = ArmKinematics(dof=3)
    planner = make_planner(arm)

    print("=" * 60)
    print("Generating MATLAB simulation data")
    print("=" * 60)

    # ── Scenario 1: Single Reach ─────────────────────────────────────────
    print("\n1. Single reach (home → target)...")
    q_target = np.array([0.5, 0.8, -0.4])
    traj, ts = plan_and_smooth(planner, np.zeros(3), q_target)
    write_trajectory_csv(output_dir / "scenario_single_reach.csv", traj, ts)
    print(f"   -> {traj.shape[0]} points, {ts[-1]:.2f}s")

    # ── Scenario 2: Multi-Point Pick Sequence ────────────────────────────
    print("\n2. Multi-point pick sequence...")
    pick_configs = [
        np.array([0.0, 0.0, 0.0]),     # Home
        np.array([0.3, 0.6, -0.2]),     # Pick 1
        np.array([0.0, 0.3, 0.0]),      # Lift
        np.array([-0.5, 0.6, -0.3]),    # Place 1
        np.array([0.0, 0.3, 0.0]),      # Lift
        np.array([0.8, 0.4, -0.5]),     # Pick 2
        np.array([0.0, 0.3, 0.0]),      # Lift
        np.array([-0.3, 0.7, -0.1]),    # Place 2
        np.array([0.0, 0.0, 0.0]),      # Home
    ]

    all_traj = []
    all_ts = []
    offset = 0.0
    current = pick_configs[0]
    for i in range(1, len(pick_configs)):
        try:
            traj, ts = plan_and_smooth(planner, current, pick_configs[i])
            all_traj.append(traj)
            all_ts.append(ts + offset)
            offset = all_ts[-1][-1] + 0.5  # 0.5s pause between moves
            current = pick_configs[i]
        except PlanningFailure:
            print(f"   Skipping waypoint {i} (planning failed)")
            continue

    combined_traj = np.vstack(all_traj)
    combined_ts = np.concatenate(all_ts)
    write_trajectory_csv(output_dir / "scenario_pick_sequence.csv", combined_traj, combined_ts)
    print(f"   → {combined_traj.shape[0]} points, {combined_ts[-1]:.2f}s")

    # ── Scenario 3: Workspace Sweep ──────────────────────────────────────
    print("\n3. Workspace sweep (scanning motion)...")
    sweep_angles = np.linspace(-1.5, 1.5, 8)
    sweep_configs = [np.array([a, 0.5, -0.3]) for a in sweep_angles]

    all_traj = []
    all_ts = []
    offset = 0.0
    current = np.zeros(3)
    for cfg in sweep_configs:
        try:
            traj, ts = plan_and_smooth(planner, current, cfg)
            all_traj.append(traj)
            all_ts.append(ts + offset)
            offset = all_ts[-1][-1] + 0.1
            current = cfg
        except PlanningFailure:
            continue

    # Return home
    try:
        traj, ts = plan_and_smooth(planner, current, np.zeros(3))
        all_traj.append(traj)
        all_ts.append(ts + offset)
    except PlanningFailure:
        pass

    combined_traj = np.vstack(all_traj)
    combined_ts = np.concatenate(all_ts)
    write_trajectory_csv(output_dir / "scenario_sweep.csv", combined_traj, combined_ts)
    print(f"   → {combined_traj.shape[0]} points, {combined_ts[-1]:.2f}s")

    # ── Scenario 4: Stress Test (rapid short moves) ──────────────────────
    print("\n4. Rapid short moves (stress test)...")
    rng = np.random.default_rng(42)
    current = np.zeros(3)
    all_traj = []
    all_ts = []
    offset = 0.0

    for i in range(15):
        delta = rng.uniform(-0.3, 0.3, 3)
        goal = np.clip(current + delta, -2.0, 2.0)
        try:
            traj, ts = plan_and_smooth(planner, current, goal)
            all_traj.append(traj)
            all_ts.append(ts + offset)
            offset = all_ts[-1][-1] + 0.05
            current = goal
        except PlanningFailure:
            continue

    combined_traj = np.vstack(all_traj)
    combined_ts = np.concatenate(all_ts)
    write_trajectory_csv(output_dir / "scenario_stress_test.csv", combined_traj, combined_ts)
    print(f"   → {combined_traj.shape[0]} points, {combined_ts[-1]:.2f}s")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("OUTPUT FILES (for MATLAB):")
    for f in sorted(output_dir.glob("*.csv")):
        data = np.loadtxt(f, delimiter=",", comments="%")
        print(f"  {f.name:40s}  {data.shape[0]:5d} pts  {data[-1,0]:.1f}s")
    print("=" * 60)
    print("\nIn MATLAB, run:")
    print("  >> cd matlab")
    print("  >> simulate_arm    % change csv_file path as needed")
    print("  >> workspace_analysis")


if __name__ == "__main__":
    main()
