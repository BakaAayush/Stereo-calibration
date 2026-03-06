# =============================================================================
# actuator_interface.py — Abstract actuator protocol
# =============================================================================
# Purpose:  Define the contract for any actuator backend (physical or mock).
#           All motion commands go through this interface, ensuring rate limits,
#           safety checks, and dead-man behaviour are uniformly enforced.
#
# Inputs:   Joint angles, velocity/acceleration limits.
# Outputs:  Actuation commands (via subclass implementation).
# =============================================================================
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class ActuatorInterface(ABC):
    """Base class for all actuator drivers.

    Subclasses must implement ``_send_angles`` and ``_read_angles``.
    This base class enforces rate limiting, soft joint limits,
    and a software dead-man switch.

    Parameters
    ----------
    dof : int
        Number of joints.
    joint_limits : (DOF, 2) array
        Per-joint (min_rad, max_rad).
    max_rate : float
        Maximum angular velocity per joint (rad/s).  Commands that would
        exceed this are slewed.
    deadman_timeout_s : float
        If no command is received within this interval, joints are frozen.
    """

    def __init__(
        self,
        dof: int,
        joint_limits: NDArray,
        max_rate: float = 1.5,
        deadman_timeout_s: float = 1.0,
    ) -> None:
        self.dof = dof
        self.limits = np.asarray(joint_limits, dtype=np.float64)
        self.max_rate = max_rate
        self.deadman_timeout = deadman_timeout_s

        self._last_cmd_time = time.monotonic()
        self._last_angles = np.zeros(dof, dtype=np.float64)
        self._enabled = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def enable(self) -> None:
        """Enable actuation (must be called before ``set_angles``)."""
        self._enabled = True
        self._last_cmd_time = time.monotonic()

    def disable(self) -> None:
        """Disable actuation — all subsequent set_angles calls are no-ops."""
        self._enabled = False

    def set_angles(self, angles: NDArray) -> NDArray:
        """Command the joints to the given angles (with safety enforcement).

        Parameters
        ----------
        angles : (DOF,) array — desired joint angles in radians.

        Returns
        -------
        actual : (DOF,) array — the angles actually sent (after clamping / rate limiting).
        """
        now = time.monotonic()
        angles = np.asarray(angles, dtype=np.float64).ravel()
        assert angles.shape[0] == self.dof

        # Dead-man check
        if not self._enabled:
            return self._last_angles.copy()

        dt = now - self._last_cmd_time
        if dt > self.deadman_timeout:
            self.disable()
            return self._last_angles.copy()

        # Soft joint limits
        angles = np.clip(angles, self.limits[:, 0], self.limits[:, 1])

        # Rate limiting (slew)
        if dt > 0:
            max_delta = self.max_rate * dt
            delta = angles - self._last_angles
            delta = np.clip(delta, -max_delta, max_delta)
            angles = self._last_angles + delta

        # Dispatch to subclass
        self._send_angles(angles)
        self._last_angles = angles.copy()
        self._last_cmd_time = now
        return angles

    def get_angles(self) -> NDArray:
        """Read current joint angles from hardware (or last commanded)."""
        return self._read_angles()

    def go_home(self) -> NDArray:
        """Return to the joint-space home position (mid-range)."""
        home = (self.limits[:, 0] + self.limits[:, 1]) / 2.0
        return self.set_angles(home)

    # ------------------------------------------------------------------ #
    # Abstract methods (subclass must implement)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _send_angles(self, angles: NDArray) -> None:
        """Low-level: send joint angles to the hardware."""
        ...

    @abstractmethod
    def _read_angles(self) -> NDArray:
        """Low-level: read joint angles from the hardware."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release hardware resources."""
        ...
