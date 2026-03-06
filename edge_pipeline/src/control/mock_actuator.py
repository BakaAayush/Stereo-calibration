# =============================================================================
# mock_actuator.py — Logging-only actuator stub for tests and simulation
# =============================================================================
# Purpose:  Provide a safe, no-hardware actuator that logs all commands.
#           Used in simulation mode and unit tests.
# =============================================================================
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .actuator_interface import ActuatorInterface

logger = logging.getLogger(__name__)


class MockActuator(ActuatorInterface):
    """Mock actuator that logs commands without moving any hardware.

    Parameters
    ----------
    dof : int
        Number of joints.
    joint_limits : (DOF, 2) array | None
        Defaults to [-π, π] per joint.
    """

    def __init__(
        self,
        dof: int = 3,
        joint_limits: NDArray | None = None,
        max_rate: float = 3.0,
    ) -> None:
        if joint_limits is None:
            joint_limits = np.tile([-np.pi, np.pi], (dof, 1))
        super().__init__(dof=dof, joint_limits=joint_limits, max_rate=max_rate)
        self.command_log: list[NDArray] = []

    def _send_angles(self, angles: NDArray) -> None:
        self.command_log.append(angles.copy())
        logger.debug("MockActuator: angles=%s", np.round(angles, 4))

    def _read_angles(self) -> NDArray:
        return self._last_angles.copy()

    def close(self) -> None:
        logger.info("MockActuator closed (%d commands logged)", len(self.command_log))
