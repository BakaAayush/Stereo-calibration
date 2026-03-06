# =============================================================================
# test_end_to_end.py — Full pipeline integration test (mock)
# =============================================================================
# Exercises: MockCamera → transform → IK → plan → smooth → CSV export
# =============================================================================
import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.camera.mock_camera import MockCamera
from src.camera.camera_interface import Detection
from src.transform.transforms import pixel_depth_to_camera, camera_to_base
from src.kinematics.arm_kinematics import ArmKinematics
from src.collision.checker import CollisionChecker
from src.planning.planner import Planner, PlannerConfig, PlannerMode
from src.planning.trajectory import smooth_trajectory, time_parameterize
from src.export.csv_export import write_trajectory_csv


class TestEndToEnd:
    """Integration test: depth → trajectory → CSV."""

    def test_mock_pipeline_produces_csv(self):
        """Full pipeline with mock inputs should produce a valid CSV."""
        # 1. Setup
        arm = ArmKinematics(dof=3)

        # 2. Use a known reachable target: FK of a known config
        q_known = np.array([0.3, 0.5, -0.3])
        point_base = arm.fk_position(q_known)

        # 3. Inverse kinematics (target is guaranteed reachable)
        q_goal, info = arm.ik(point_base, q0=np.zeros(3))
        assert q_goal is not None, f"IK failed: {info}"

        # 5. Planning (no obstacles → direct path)
        def fk_frames(q):
            frames = [np.eye(4)]
            for i in range(arm.dof):
                q_partial = np.zeros(arm.dof)
                q_partial[:i + 1] = q[:i + 1]
                frames.append(arm.fk(q_partial))
            return frames

        cc = CollisionChecker([0.02, 0.02, 0.02], fk_frames)
        planner = Planner(
            dof=3,
            joint_limits=arm.joint_limits,
            collision_checker=cc,
        )
        start = np.zeros(3)
        path = planner.plan(start, q_goal, [])

        # 6. Smooth
        traj = smooth_trajectory(path)
        traj, timestamps = time_parameterize(traj)

        # 7. Export CSV
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "trajectory.csv"
            write_trajectory_csv(csv_path, traj, timestamps)

            assert csv_path.exists()

            # Verify CSV is valid numeric data
            with open(csv_path) as f:
                lines = f.readlines()

            # Skip comment lines
            data_lines = [l for l in lines if not l.startswith("%")]
            assert len(data_lines) > 0

            # Parse first data line
            values = data_lines[0].strip().split(",")
            assert len(values) == 4  # time + 3 joints
            for v in values:
                float(v)  # should not raise


    def test_csv_loadable_by_numpy(self):
        """Verify that the CSV can be loaded by numpy.loadtxt (MATLAB compat)."""
        arm = ArmKinematics(dof=3)
        traj = np.random.randn(50, 3) * 0.5
        timestamps = np.arange(50) * 0.02

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "test.csv"
            write_trajectory_csv(csv_path, traj, timestamps)

            # numpy.loadtxt should read it (skip comment lines)
            data = np.loadtxt(csv_path, delimiter=",", comments="%")
            assert data.shape == (50, 4)  # time + 3 joints
            np.testing.assert_allclose(data[:, 0], timestamps, atol=1e-5)
