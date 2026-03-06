# =============================================================================
# arm_kinematics.py — FK, IK, and Jacobian for 3–4 DOF robot arms
# =============================================================================
# Purpose:  Provide a unified ArmKinematics class that:
#   1. Uses roboticstoolbox-python as primary backend (if available).
#   2. Falls back to ikpy + custom gradient-descent IK.
#   3. Supports configurable DH parameters, joint limits, and DOF (3 or 4).
#   4. Null-space projection for joint-limit avoidance.
#
# Inputs:   DH parameter table, joint limits, desired poses.
# Outputs:  Joint angles (IK), end-effector poses (FK), Jacobians.
#
# Performance: IK target < 10 ms median for 3-DOF on Pi 5.
# =============================================================================
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Try importing roboticstoolbox; set flag ──────────────────────────────
_HAS_RTB = False
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    _HAS_RTB = True
except ImportError:
    rtb = None  # type: ignore[assignment]
    SE3 = None  # type: ignore[assignment,misc]

# ── Try importing ikpy as fallback ───────────────────────────────────────
_HAS_IKPY = False
try:
    import ikpy.chain
    import ikpy.link
    _HAS_IKPY = True
except ImportError:
    ikpy = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class DHRow:
    """Standard DH parameters for one joint.

    Convention: modified DH (Craig) —
        a (link length), alpha (link twist), d (link offset), theta_offset.
    ``theta_offset`` is added to the variable joint angle.

    ASSUMPTION: All joints are revolute.
    """

    a: float        # link length  (m)
    alpha: float    # link twist   (rad)
    d: float        # link offset  (m)
    theta_offset: float = 0.0  # constant offset added to joint angle


# ── Default DH tables (ASSUMPTION: generic educational arm) ──────────────
# These are representative parameters.  Replace with your actual arm.
DEFAULT_DH_3DOF: list[DHRow] = [
    #     a       alpha      d    theta_offset
    DHRow(0.0,    np.pi/2,  0.077, 0.0),   # Joint 1 — base rotation
    DHRow(0.130,  0.0,      0.0,   0.0),   # Joint 2 — shoulder
    DHRow(0.124,  0.0,      0.0,   0.0),   # Joint 3 — elbow
]

DEFAULT_DH_4DOF: list[DHRow] = [
    DHRow(0.0,    np.pi/2,  0.077, 0.0),   # Joint 1 — base rotation
    DHRow(0.130,  0.0,      0.0,   0.0),   # Joint 2 — shoulder
    DHRow(0.124,  0.0,      0.0,   0.0),   # Joint 3 — elbow
    DHRow(0.126,  0.0,      0.0,   0.0),   # Joint 4 — wrist
]


class ArmKinematics:
    """Configurable FK / IK / Jacobian for 3 or 4 DOF revolute arms.

    Parameters
    ----------
    dof : int
        Degrees of freedom (3 or 4).
    dh_params : list[DHRow] | None
        DH parameter table.  Defaults to generic arm if ``None``.
    joint_limits : list[tuple[float, float]] | None
        Per-joint (min, max) in radians.  Defaults to [-π, π] each.
    ik_timeout_s : float
        Maximum wall-clock time for a single IK call (seconds).
    ik_tol : float
        IK position tolerance (metres).
    backend : str
        ``"auto"`` (try rtb then ikpy), ``"rtb"``, or ``"ikpy"``.
    """

    def __init__(
        self,
        dof: int = 3,
        dh_params: Sequence[DHRow] | None = None,
        joint_limits: Sequence[tuple[float, float]] | None = None,
        ik_timeout_s: float = 0.05,
        ik_tol: float = 1e-3,
        backend: str = "auto",
    ) -> None:
        assert dof in (3, 4), f"Supported DOF: 3 or 4, got {dof}"
        self.dof = dof
        self.dh = list(dh_params) if dh_params else (DEFAULT_DH_3DOF if dof == 3 else DEFAULT_DH_4DOF)
        assert len(self.dh) == dof

        if joint_limits is not None:
            self.joint_limits = np.array(joint_limits, dtype=np.float64)
        else:
            self.joint_limits = np.tile([-np.pi, np.pi], (dof, 1))

        self.ik_timeout_s = ik_timeout_s
        self.ik_tol = ik_tol

        # Select backend
        self._backend = self._select_backend(backend)
        logger.info("ArmKinematics: dof=%d, backend=%s", dof, self._backend)

        # Build kinematic model
        if self._backend == "rtb":
            self._robot = self._build_rtb_robot()
        else:
            self._chain = self._build_ikpy_chain()
            self._robot = None

        # Pre-allocate Jacobian workspace
        self._jac_ws = np.zeros((6, dof), dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    def _select_backend(self, pref: str) -> str:
        if pref == "rtb" and _HAS_RTB:
            return "rtb"
        if pref == "ikpy" and _HAS_IKPY:
            return "ikpy"
        if pref == "auto":
            if _HAS_RTB:
                return "rtb"
            if _HAS_IKPY:
                return "ikpy"
        raise ImportError(
            "Neither roboticstoolbox-python nor ikpy is installed. "
            "Install at least one:  pip install ikpy"
        )

    # ------------------------------------------------------------------ #
    # Robot construction — roboticstoolbox
    # ------------------------------------------------------------------ #
    def _build_rtb_robot(self) -> "rtb.DHRobot":
        links = []
        for i, dh in enumerate(self.dh):
            link = rtb.RevoluteMDH(
                a=dh.a,
                alpha=dh.alpha,
                d=dh.d,
                offset=dh.theta_offset,
                qlim=self.joint_limits[i],
            )
            links.append(link)
        robot = rtb.DHRobot(links, name=f"arm_{self.dof}dof")
        return robot

    # ------------------------------------------------------------------ #
    # Robot construction — ikpy
    # ------------------------------------------------------------------ #
    def _build_ikpy_chain(self) -> "ikpy.chain.Chain":
        links = [ikpy.link.OriginLink()]
        for i, dh in enumerate(self.dh):
            bounds = tuple(self.joint_limits[i].tolist())
            links.append(
                ikpy.link.URDFLink(
                    name=f"joint_{i}",
                    origin_translation=[dh.a, 0, dh.d],
                    origin_orientation=[dh.alpha, 0, 0],
                    rotation=[0, 0, 1],
                    bounds=bounds,
                )
            )
        return ikpy.chain.Chain(name=f"arm_{self.dof}dof", links=links)

    # ================================================================== #
    # FORWARD KINEMATICS
    # ================================================================== #
    def fk(self, q: NDArray) -> NDArray:
        """Compute forward kinematics.

        Parameters
        ----------
        q : (DOF,) array — joint angles in radians.

        Returns
        -------
        T : (4, 4) ndarray — homogeneous transform of end-effector in base frame.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        assert q.shape[0] == self.dof

        if self._backend == "rtb":
            T = self._robot.fkine(q)
            return np.array(T.A, dtype=np.float64)
        else:
            # ikpy returns (4,4) but expects padded q with origin link
            q_padded = np.concatenate([[0.0], q])
            T = self._chain.forward_kinematics(q_padded)
            return np.array(T, dtype=np.float64)

    def fk_position(self, q: NDArray) -> NDArray:
        """Return just the end-effector position (x, y, z) in metres."""
        T = self.fk(q)
        return T[:3, 3]

    # ================================================================== #
    # INVERSE KINEMATICS
    # ================================================================== #
    def ik(
        self,
        target_pos: NDArray,
        target_rot: NDArray | None = None,
        q0: NDArray | None = None,
    ) -> tuple[NDArray | None, dict]:
        """Compute inverse kinematics to reach a target position.

        Parameters
        ----------
        target_pos : (3,) array — desired [x, y, z] in base frame.
        target_rot : (3, 3) array | None — desired orientation (optional).
        q0 : (DOF,) array | None — initial joint guess.

        Returns
        -------
        q : (DOF,) array or None — joint solution, or None on failure.
        info : dict — ``{"converged": bool, "error_m": float, "time_ms": float}``.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64).ravel()
        if q0 is None:
            q0 = np.zeros(self.dof, dtype=np.float64)
        q0 = np.asarray(q0, dtype=np.float64).ravel()

        t0 = time.perf_counter()

        if self._backend == "rtb":
            result = self._ik_rtb(target_pos, target_rot, q0)
        else:
            result = self._ik_ikpy(target_pos, q0)

        # If primary IK failed, try gradient descent fallback
        if result[0] is None:
            logger.warning("Primary IK failed; trying gradient-descent fallback")
            result = self._ik_gradient_descent(target_pos, q0)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        if result[0] is not None:
            result[1]["time_ms"] = elapsed_ms
            # Clamp to joint limits
            result = (self.clamp_joints(result[0]), result[1])
        else:
            result[1]["time_ms"] = elapsed_ms

        return result

    def _ik_rtb(self, pos: NDArray, rot: NDArray | None, q0: NDArray) -> tuple:
        """IK using roboticstoolbox's numerical solver."""
        if rot is not None:
            T_target = SE3.Rt(rot, pos)
        else:
            T_target = SE3(pos)

        try:
            sol = self._robot.ik_LM(
                T_target,
                q0=q0,
                tol=self.ik_tol,
                ilimit=500,
                slimit=10,
                mask=[1, 1, 1, 0, 0, 0] if rot is None else [1, 1, 1, 1, 1, 1],
            )
            if sol.success:
                err = np.linalg.norm(self.fk_position(sol.q) - pos)
                return (sol.q, {"converged": True, "error_m": float(err)})
        except Exception as e:
            logger.debug("rtb IK exception: %s", e)

        return (None, {"converged": False, "error_m": float("inf")})

    def _ik_ikpy(self, pos: NDArray, q0: NDArray) -> tuple:
        """IK using ikpy's built-in solver."""
        try:
            q0_padded = np.concatenate([[0.0], q0])
            q_full = self._chain.inverse_kinematics(
                target_position=pos,
                initial_position=q0_padded,
            )
            q = q_full[1: self.dof + 1]
            err = np.linalg.norm(self.fk_position(q) - pos)
            if err < self.ik_tol * 10:  # ikpy is less precise
                return (q, {"converged": True, "error_m": float(err)})
        except Exception as e:
            logger.debug("ikpy IK exception: %s", e)

        return (None, {"converged": False, "error_m": float("inf")})

    def _ik_gradient_descent(
        self,
        target_pos: NDArray,
        q0: NDArray,
        lr: float = 0.5,
        max_iter: int = 500,
    ) -> tuple:
        """Fallback gradient-descent IK with Jacobian pseudo-inverse.

        Uses the geometric Jacobian to iteratively descend toward the target.
        Applies null-space projection for joint-limit avoidance.
        """
        q = q0.copy()
        deadline = time.perf_counter() + self.ik_timeout_s

        for i in range(max_iter):
            if time.perf_counter() > deadline:
                break

            pos = self.fk_position(q)
            err_vec = target_pos - pos
            err_norm = np.linalg.norm(err_vec)

            if err_norm < self.ik_tol:
                return (q.copy(), {"converged": True, "error_m": float(err_norm)})

            J = self.jacobian(q)[:3, :]  # positional Jacobian only
            J_pinv = np.linalg.pinv(J)
            dq = J_pinv @ err_vec

            # Null-space projection for joint-limit avoidance
            null_proj = np.eye(self.dof) - J_pinv @ J
            q_mid = (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2.0
            dq_null = null_proj @ (q_mid - q) * 0.1
            dq += dq_null

            q = q + lr * dq
            q = self.clamp_joints(q)

        err = np.linalg.norm(self.fk_position(q) - target_pos)
        converged = err < self.ik_tol
        return (q if converged else None, {"converged": converged, "error_m": float(err)})

    # ================================================================== #
    # JACOBIAN
    # ================================================================== #
    def jacobian(self, q: NDArray) -> NDArray:
        """Compute the 6×DOF geometric Jacobian at configuration ``q``.

        Returns
        -------
        J : (6, DOF) ndarray — rows 0-2 are linear velocity, 3-5 are angular.
        """
        q = np.asarray(q, dtype=np.float64).ravel()

        if self._backend == "rtb":
            return np.array(self._robot.jacob0(q), dtype=np.float64)

        # Numerical Jacobian (central difference) for ikpy backend
        eps = 1e-6
        J = self._jac_ws
        pos0 = self.fk_position(q)
        for i in range(self.dof):
            q_plus = q.copy()
            q_plus[i] += eps
            q_minus = q.copy()
            q_minus[i] -= eps
            dp = (self.fk_position(q_plus) - self.fk_position(q_minus)) / (2 * eps)
            J[:3, i] = dp
            # Angular part: approximate as rotation axis z_i
            T_i = self.fk(q)  # simplified — use z-axis of frame
            J[3:, i] = T_i[:3, 2]

        return J.copy()

    # ================================================================== #
    # UTILITIES
    # ================================================================== #
    def clamp_joints(self, q: NDArray) -> NDArray:
        """Clamp joint angles to their limits."""
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    def is_within_limits(self, q: NDArray) -> bool:
        """Check if all joints are within limits."""
        q = np.asarray(q, dtype=np.float64)
        return bool(np.all(q >= self.joint_limits[:, 0]) and np.all(q <= self.joint_limits[:, 1]))

    @property
    def home_position(self) -> NDArray:
        """Return the joint-space home (mid-range) configuration."""
        return (self.joint_limits[:, 0] + self.joint_limits[:, 1]) / 2.0

    def workspace_radius(self) -> float:
        """Approximate maximum reach of the arm (sum of link lengths)."""
        return sum(dh.a + abs(dh.d) for dh in self.dh)
