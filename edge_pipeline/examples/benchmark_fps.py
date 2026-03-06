#!/usr/bin/env python3
# =============================================================================
# benchmark_fps.py — Microbenchmarks for pipeline stages
# =============================================================================
# Measures latency for: transform, IK, planning, smoothing.
# Outputs profiling.json with mean/median/p99 per stage.
#
# Usage:
#   python examples/benchmark_fps.py
#   python examples/benchmark_fps.py --iterations 500 --dof 4
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transform.transforms import pixel_depth_to_camera, camera_to_base, TransformWorkspace
from src.kinematics.arm_kinematics import ArmKinematics
from src.collision.checker import CollisionChecker
from src.planning.planner import Planner, PlannerConfig, PlannerMode, PlanningFailure
from src.planning.trajectory import smooth_trajectory, time_parameterize

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def benchmark_stage(name: str, func, n: int) -> dict:
    """Run a function n times and return timing statistics."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = np.array(times)
    return {
        "stage": name,
        "n_iterations": n,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "std_ms": float(np.std(times)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Stage Benchmarks")
    parser.add_argument("--iterations", "-n", type=int, default=200)
    parser.add_argument("--dof", type=int, default=3, choices=[3, 4])
    parser.add_argument("--output", type=str, default="output/profiling.json")
    args = parser.parse_args()

    n = args.iterations
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    K = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    arm = ArmKinematics(dof=args.dof)

    results = []

    # ── 1. Transform benchmark ───────────────────────────────────────────
    print(f"Benchmarking transform ({n} iterations)...")
    u, v, z = 640.0, 360.0, 0.35

    def run_transform():
        pt = pixel_depth_to_camera(u, v, z, K)
        camera_to_base(pt, T)

    results.append(benchmark_stage("transform_single", run_transform, n))

    # Batch transform
    u_batch = np.random.uniform(0, 1280, 100)
    v_batch = np.random.uniform(0, 720, 100)
    z_batch = np.random.uniform(0.1, 1.0, 100)

    def run_batch_transform():
        pts = pixel_depth_to_camera(u_batch, v_batch, z_batch, K)
        camera_to_base(pts, T)

    results.append(benchmark_stage("transform_batch_100", run_batch_transform, n))

    # Workspace transform
    ws = TransformWorkspace(max_points=100)

    def run_ws_transform():
        ws.pixel_to_base(u_batch, v_batch, z_batch, K, T)

    results.append(benchmark_stage("transform_workspace_100", run_ws_transform, n))

    # ── 2. IK benchmark ──────────────────────────────────────────────────
    print(f"Benchmarking IK ({n} iterations)...")
    # Generate reachable targets
    test_configs = [np.random.uniform(-1, 1, args.dof) for _ in range(n)]
    targets = [arm.fk_position(q) for q in test_configs]

    ik_times = []
    for target in targets:
        t0 = time.perf_counter()
        arm.ik(target)
        ik_times.append((time.perf_counter() - t0) * 1000)

    ik_times = np.array(ik_times)
    results.append({
        "stage": "ik",
        "n_iterations": n,
        "mean_ms": float(np.mean(ik_times)),
        "median_ms": float(np.median(ik_times)),
        "p99_ms": float(np.percentile(ik_times, 99)),
        "min_ms": float(np.min(ik_times)),
        "max_ms": float(np.max(ik_times)),
        "std_ms": float(np.std(ik_times)),
    })

    # ── 3. Planner benchmark ─────────────────────────────────────────────
    print(f"Benchmarking planner ({min(n, 50)} iterations)...")

    def fk_frames(q):
        frames = [np.eye(4)]
        for i in range(arm.dof):
            q_p = np.zeros(arm.dof)
            q_p[:i + 1] = q[:i + 1]
            frames.append(arm.fk(q_p))
        return frames

    cc = CollisionChecker([0.02] * arm.dof, fk_frames)
    planner = Planner(
        dof=args.dof,
        joint_limits=arm.joint_limits,
        collision_checker=cc,
        config=PlannerConfig(mode=PlannerMode.RRT_STAR, max_iterations=1000, max_time_s=1.0),
    )

    plan_times = []
    plan_n = min(n, 50)  # fewer iterations for planner (slower)
    for _ in range(plan_n):
        start = np.zeros(args.dof)
        goal = np.random.uniform(-0.8, 0.8, args.dof)
        t0 = time.perf_counter()
        try:
            planner.plan(start, goal, [])
        except PlanningFailure:
            pass
        plan_times.append((time.perf_counter() - t0) * 1000)

    plan_times = np.array(plan_times)
    results.append({
        "stage": "planner_rrt_star",
        "n_iterations": plan_n,
        "mean_ms": float(np.mean(plan_times)),
        "median_ms": float(np.median(plan_times)),
        "p99_ms": float(np.percentile(plan_times, 99)),
        "min_ms": float(np.min(plan_times)),
        "max_ms": float(np.max(plan_times)),
        "std_ms": float(np.std(plan_times)),
    })

    # ── 4. Smoothing benchmark ───────────────────────────────────────────
    print(f"Benchmarking trajectory smoothing ({n} iterations)...")
    waypoints = [np.zeros(args.dof), np.array([0.5] * args.dof), np.array([1.0] * args.dof)]

    def run_smooth():
        traj = smooth_trajectory(waypoints)
        time_parameterize(traj)

    results.append(benchmark_stage("smoothing", run_smooth, n))

    # ── Output ───────────────────────────────────────────────────────────
    output = {
        "benchmark_config": {
            "dof": args.dof,
            "n_iterations": n,
            "platform": sys.platform,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("PROFILING RESULTS")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['stage']:30s}  mean={r['mean_ms']:8.3f} ms  "
              f"median={r['median_ms']:8.3f} ms  p99={r['p99_ms']:8.3f} ms")
    print(f"{'='*60}")
    print(f"Results written to: {output_path}")

    # Overall FPS estimate (transform + IK only, excluding planner)
    transform_ms = results[0]["median_ms"]
    ik_ms = results[3]["median_ms"]
    est_fps = 1000.0 / (transform_ms + ik_ms)
    print(f"\nEstimated transform+IK FPS: {est_fps:.1f}")


if __name__ == "__main__":
    main()
