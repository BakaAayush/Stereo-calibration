# =============================================================================
# test_kinematics.py — Unit tests for FK, IK, and Jacobian
# =============================================================================
import numpy as np
import pytest

from src.kinematics.arm_kinematics import ArmKinematics


class TestForwardKinematics:
    """FK validation for 3-DOF and 4-DOF arms."""

    def test_home_position_3dof(self, arm_3dof: ArmKinematics):
        """FK at zero config should return a valid 4×4 transform."""
        q = np.zeros(3)
        T = arm_3dof.fk(q)
        assert T.shape == (4, 4)
        # Bottom row should be [0, 0, 0, 1]
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_home_position_4dof(self, arm_4dof: ArmKinematics):
        """FK at zero config for 4-DOF."""
        q = np.zeros(4)
        T = arm_4dof.fk(q)
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1], atol=1e-10)

    def test_fk_different_configs(self, arm_3dof: ArmKinematics):
        """FK at different configs should give different poses."""
        T0 = arm_3dof.fk(np.array([0.0, 0.0, 0.0]))
        T1 = arm_3dof.fk(np.array([0.5, 0.3, -0.2]))
        assert not np.allclose(T0, T1)

    def test_fk_position_matches(self, arm_3dof: ArmKinematics):
        """fk_position should match fk[:3, 3]."""
        q = np.array([0.3, -0.5, 0.8])
        T = arm_3dof.fk(q)
        pos = arm_3dof.fk_position(q)
        np.testing.assert_allclose(pos, T[:3, 3], atol=1e-10)


class TestInverseKinematics:
    """IK validation: FK(IK(target)) ≈ target."""

    def test_ik_round_trip_3dof(self, arm_3dof: ArmKinematics):
        """IK should find a config whose FK matches the target position."""
        # Use FK to generate a reachable target
        q_known = np.array([0.3, 0.5, -0.3])
        target = arm_3dof.fk_position(q_known)

        q_sol, info = arm_3dof.ik(target, q0=np.zeros(3))
        assert q_sol is not None, f"IK failed: {info}"
        assert info["converged"], f"IK did not converge: {info}"

        actual = arm_3dof.fk_position(q_sol)
        np.testing.assert_allclose(actual, target, atol=0.01)

    def test_ik_round_trip_4dof(self, arm_4dof: ArmKinematics):
        """IK round-trip for 4-DOF arm."""
        q_known = np.array([0.2, 0.4, -0.1, 0.3])
        target = arm_4dof.fk_position(q_known)

        q_sol, info = arm_4dof.ik(target, q0=np.zeros(4))
        assert q_sol is not None, f"IK failed: {info}"

        actual = arm_4dof.fk_position(q_sol)
        np.testing.assert_allclose(actual, target, atol=0.01)

    def test_ik_multiple_poses_3dof(self, arm_3dof: ArmKinematics):
        """Test IK for several different poses."""
        test_configs = [
            [0.0, 0.0, 0.0],
            [0.5, -0.3, 0.7],
            [-0.4, 0.6, -0.2],
            [1.0, 0.3, -0.8],
            [-0.8, -0.5, 0.4],
        ]
        for q_ref in test_configs:
            q_ref = np.array(q_ref)
            target = arm_3dof.fk_position(q_ref)
            q_sol, info = arm_3dof.ik(target)
            assert q_sol is not None, f"IK failed for config {q_ref}: {info}"

            actual = arm_3dof.fk_position(q_sol)
            np.testing.assert_allclose(actual, target, atol=0.01,
                err_msg=f"FK/IK mismatch for config {q_ref}")

    def test_ik_respects_joint_limits(self, arm_3dof: ArmKinematics):
        """IK solution should be within joint limits."""
        q_ref = np.array([0.3, 0.5, -0.3])
        target = arm_3dof.fk_position(q_ref)
        q_sol, _ = arm_3dof.ik(target)
        if q_sol is not None:
            assert arm_3dof.is_within_limits(q_sol)


class TestJacobian:
    """Jacobian computation tests."""

    def test_jacobian_shape(self, arm_3dof: ArmKinematics):
        """Jacobian should be 6×DOF."""
        q = np.zeros(3)
        J = arm_3dof.jacobian(q)
        assert J.shape == (6, 3)

    def test_jacobian_shape_4dof(self, arm_4dof: ArmKinematics):
        J = arm_4dof.jacobian(np.zeros(4))
        assert J.shape == (6, 4)

    def test_jacobian_numerical_consistency(self, arm_3dof: ArmKinematics):
        """Jacobian columns should approximate FK position change for small dq."""
        q = np.array([0.3, -0.2, 0.5])
        J = arm_3dof.jacobian(q)[:3, :]  # positional Jacobian

        eps = 1e-5
        for i in range(3):
            q_plus = q.copy()
            q_plus[i] += eps
            dp = arm_3dof.fk_position(q_plus) - arm_3dof.fk_position(q)
            np.testing.assert_allclose(dp / eps, J[:, i], atol=0.05,
                err_msg=f"Jacobian column {i} inconsistent")


class TestUtilities:
    """Test utility methods."""

    def test_clamp_joints(self, arm_3dof: ArmKinematics):
        q = np.array([10.0, -10.0, 0.0])
        clamped = arm_3dof.clamp_joints(q)
        assert arm_3dof.is_within_limits(clamped)

    def test_home_position(self, arm_3dof: ArmKinematics):
        home = arm_3dof.home_position
        assert home.shape == (3,)
        assert arm_3dof.is_within_limits(home)

    def test_workspace_radius(self, arm_3dof: ArmKinematics):
        r = arm_3dof.workspace_radius()
        assert r > 0
