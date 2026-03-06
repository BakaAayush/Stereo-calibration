# =============================================================================
# csv_export.py — Write trajectory data to CSV for MATLAB ingestion
# =============================================================================
# Purpose:  Export joint-space trajectories as CSV files compatible with
#           MATLAB's ``readmatrix()`` function.  Includes a header comment
#           block describing DOF order, sampling time, and units.
#
# Output format:
#   # DOF: 3
#   # dt: 0.020 s
#   # units: radians
#   # columns: time_s, joint_1, joint_2, joint_3
#   0.000,0.0000,0.0000,0.0000
#   0.020,0.0123,0.0456,0.0789
#   ...
# =============================================================================
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def write_trajectory_csv(
    filepath: str | Path,
    trajectory: NDArray,
    timestamps: NDArray | None = None,
    dt: float = 0.02,
    units: str = "radians",
    joint_names: list[str] | None = None,
) -> Path:
    """Write a trajectory to CSV.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    trajectory : (N, DOF) array
        Joint-angle trajectory.
    timestamps : (N,) array | None
        Time stamps.  If ``None``, generated from ``dt``.
    dt : float
        Sampling period (used if timestamps is None).
    units : str
        ``"radians"`` or ``"degrees"``.
    joint_names : list[str] | None
        Column names.  Defaults to ``["joint_1", ...]``.

    Returns
    -------
    Path — the written file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_points, dof = trajectory.shape

    if timestamps is None:
        timestamps = np.arange(n_points, dtype=np.float64) * dt

    if joint_names is None:
        joint_names = [f"joint_{i+1}" for i in range(dof)]

    with open(filepath, "w", newline="") as f:
        # Header comments (MATLAB's readmatrix ignores lines starting with %)
        f.write(f"% DOF: {dof}\n")
        f.write(f"% dt: {dt:.4f} s\n")
        f.write(f"% units: {units}\n")
        f.write(f"% columns: time_s, {', '.join(joint_names)}\n")

        writer = csv.writer(f)
        for i in range(n_points):
            row = [f"{timestamps[i]:.6f}"] + [f"{trajectory[i, j]:.6f}" for j in range(dof)]
            writer.writerow(row)

    logger.info("Trajectory CSV written: %s (%d points, %d DOF)", filepath, n_points, dof)
    return filepath
