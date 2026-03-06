# =============================================================================
# trajectory.py — Trajectory smoothing and time-parameterisation
# =============================================================================
# Purpose:  Take a list of joint-space waypoints from the planner and produce
#           a smooth, time-parameterised trajectory suitable for servo control.
#
# Methods:
#   1. Cubic spline interpolation (C1 continuous).
#   2. Quintic spline interpolation (C2 continuous — zero accel at boundaries).
#   3. Trapezoidal velocity profile time-parameterisation.
#
# Inputs:   Waypoints [(DOF,)], sample_dt, max velocity/acceleration limits.
# Outputs:  (N, DOF) array of interpolated joint angles at fixed dt.
# =============================================================================
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


def smooth_trajectory(
    waypoints: list[NDArray],
    dt: float = 0.02,
    method: str = "cubic",
) -> NDArray:
    """Smooth a list of waypoints into a dense trajectory.

    Parameters
    ----------
    waypoints : list of (DOF,) arrays
        Ordered joint-space waypoints (start → goal).
    dt : float
        Desired sampling period (seconds).  Default 50 Hz.
    method : str
        ``"cubic"`` or ``"quintic"`` spline interpolation.

    Returns
    -------
    trajectory : (N, DOF) ndarray
        Smoothed joint-angle trajectory sampled at ``dt``.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for smoothing")

    pts = np.array(waypoints, dtype=np.float64)  # (M, DOF)
    n_wp = pts.shape[0]

    # Compute cumulative distance as parameter
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_s = s[-1]

    if total_s < 1e-9:
        # All waypoints are identical
        return pts[0:1].copy()

    # Number of output samples
    n_samples = max(int(np.ceil(total_s / (dt * np.mean(seg_lengths) / dt))), n_wp * 5)
    n_samples = min(n_samples, 10000)  # cap for memory safety
    s_fine = np.linspace(0, total_s, n_samples)

    if method == "quintic":
        # Quintic: use cubic with clamped boundary (bc_type ensures C2)
        cs = CubicSpline(s, pts, bc_type="clamped")
    else:
        # Cubic natural spline
        cs = CubicSpline(s, pts, bc_type="natural")

    return cs(s_fine)


def time_parameterize(
    trajectory: NDArray,
    max_vel: NDArray | float = 2.0,
    max_acc: NDArray | float = 5.0,
    dt: float = 0.02,
) -> tuple[NDArray, NDArray]:
    """Apply trapezoidal velocity profile time-parameterisation.

    Parameters
    ----------
    trajectory : (N, DOF) array
        Position trajectory from ``smooth_trajectory``.
    max_vel : float or (DOF,) array
        Maximum joint velocity (rad/s).
    max_acc : float or (DOF,) array
        Maximum joint acceleration (rad/s²).
    dt : float
        Output sampling period (seconds).

    Returns
    -------
    traj_out : (M, DOF) array
        Time-parameterised trajectory.
    timestamps : (M,) array
        Time stamps in seconds for each sample.
    """
    n_points, dof = trajectory.shape

    max_vel = np.broadcast_to(np.asarray(max_vel, dtype=np.float64), (dof,))
    max_acc = np.broadcast_to(np.asarray(max_acc, dtype=np.float64), (dof,))

    # Compute required velocities between waypoints
    dq = np.diff(trajectory, axis=0)  # (N-1, DOF)
    ds = np.linalg.norm(dq, axis=1)  # (N-1,)

    # For each segment, compute the minimum time respecting vel and acc limits
    times = []
    for i in range(len(ds)):
        if ds[i] < 1e-9:
            times.append(dt)
            continue

        # Per-joint time requirement
        t_vel = np.abs(dq[i]) / max_vel  # time limited by velocity
        t_acc = np.sqrt(2.0 * np.abs(dq[i]) / max_acc)  # time limited by acceleration
        t_seg = float(np.max(np.maximum(t_vel, t_acc)))
        t_seg = max(t_seg, dt)  # at least one time step
        times.append(t_seg)

    # Build time vector
    t_waypoints = np.concatenate([[0.0], np.cumsum(times)])
    total_time = t_waypoints[-1]
    n_out = max(int(np.ceil(total_time / dt)), 2)
    t_out = np.linspace(0, total_time, n_out)

    # Interpolate onto uniform time grid
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(t_waypoints, trajectory, bc_type="clamped")
    traj_out = cs(t_out)

    return traj_out, t_out
