# =============================================================================
# json_export.py — Write trajectory data to JSON
# =============================================================================
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def write_trajectory_json(
    filepath: str | Path,
    trajectory: NDArray,
    timestamps: NDArray | None = None,
    dt: float = 0.02,
    units: str = "radians",
    joint_names: list[str] | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write a trajectory to JSON.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    trajectory : (N, DOF) array
        Joint-angle trajectory.
    timestamps : (N,) array | None
        Time stamps.
    dt : float
        Sampling period.
    units : str
        Angle units.
    joint_names : list[str] | None
        Joint column names.
    metadata : dict | None
        Additional metadata to include.

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

    data = {
        "header": {
            "dof": dof,
            "dt_s": dt,
            "units": units,
            "joint_names": joint_names,
            "n_points": n_points,
        },
        "trajectory": {
            "timestamps": timestamps.tolist(),
            "joint_angles": trajectory.tolist(),
        },
    }
    if metadata:
        data["metadata"] = metadata

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Trajectory JSON written: %s (%d points)", filepath, n_points)
    return filepath
