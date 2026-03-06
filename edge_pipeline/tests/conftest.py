# =============================================================================
# conftest.py — Shared pytest fixtures for the edge pipeline test suite
# =============================================================================
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

# Add project root to sys.path so tests can import src.*
_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from src.kinematics.arm_kinematics import ArmKinematics, DHRow
from src.collision.checker import CollisionChecker, Obstacle


@pytest.fixture
def intrinsics() -> np.ndarray:
    """Typical 720p camera intrinsic matrix."""
    return np.array([
        [600.0, 0.0, 640.0],
        [0.0, 600.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@pytest.fixture
def T_identity() -> np.ndarray:
    """Identity camera-to-base transform (camera = base)."""
    return np.eye(4, dtype=np.float64)


@pytest.fixture
def arm_3dof() -> ArmKinematics:
    """3-DOF arm with default DH parameters."""
    return ArmKinematics(dof=3)


@pytest.fixture
def arm_4dof() -> ArmKinematics:
    """4-DOF arm with default DH parameters."""
    return ArmKinematics(dof=4)


@pytest.fixture
def simple_obstacles() -> list[Obstacle]:
    """A small set of sphere obstacles for testing."""
    return [
        Obstacle(center=(0.2, 0.0, 0.1), radius=0.03),
        Obstacle(center=(-0.1, 0.15, 0.05), radius=0.02),
    ]
