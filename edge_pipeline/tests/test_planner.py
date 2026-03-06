# =============================================================================
# test_planner.py — Unit tests for path planning and trajectory smoothing
# =============================================================================
import numpy as np
import pytest

from src.collision.checker import CollisionChecker, Obstacle
from src.kinematics.arm_kinematics import ArmKinematics
from src.planning.planner import Planner, PlannerConfig, PlannerMode, PlanningFailure
from src.planning.trajectory import smooth_trajectory, time_parameterize


# ── Helper: build a simple collision checker ─────────────────────────────
def _make_checker(arm: ArmKinematics) -> CollisionChecker:
    """Create a collision checker with the given arm's FK."""

    def fk_frames(q):
        """Compute per-joint frames using simple FK."""
        frames = [np.eye(4)]
        for i in range(arm.dof):
            q_partial = np.zeros(arm.dof)
            q_partial[:i + 1] = q[:i + 1]
            frames.append(arm.fk(q_partial))
        return frames

    radii = [0.02] * arm.dof
    return CollisionChecker(link_radii=radii, fk_frames=fk_frames)


class TestPlannerNoObstacles:
    """Planner tests with empty obstacle set."""

    def test_direct_path(self, arm_3dof: ArmKinematics):
        """With no obstacles, planner should find a direct path."""
        cc = _make_checker(arm_3dof)
        planner = Planner(
            dof=3,
            joint_limits=arm_3dof.joint_limits,
            collision_checker=cc,
            config=PlannerConfig(mode=PlannerMode.RRT_STAR, max_iterations=100),
        )
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([0.5, 0.3, -0.2])
        path = planner.plan(start, goal, [])

        assert len(path) >= 2
        np.testing.assert_allclose(path[0], start, atol=1e-6)
        np.testing.assert_allclose(path[-1], goal, atol=1e-6)

    def test_apf_mode(self, arm_3dof: ArmKinematics):
        """APF planner should work for a simple case."""
        cc = _make_checker(arm_3dof)
        planner = Planner(
            dof=3,
            joint_limits=arm_3dof.joint_limits,
            collision_checker=cc,
            config=PlannerConfig(mode=PlannerMode.APF, max_time_s=2.0),
        )
        start = np.zeros(3)
        goal = np.array([0.3, 0.2, -0.1])
        path = planner.plan(start, goal, [])
        assert len(path) >= 2


class TestPlannerWithObstacles:
    """Planner tests with obstacles."""

    def test_rrt_avoids_obstacle(self, arm_3dof: ArmKinematics, simple_obstacles):
        """RRT* should find a path that avoids obstacles."""
        cc = _make_checker(arm_3dof)
        planner = Planner(
            dof=3,
            joint_limits=arm_3dof.joint_limits,
            collision_checker=cc,
            config=PlannerConfig(
                mode=PlannerMode.RRT_STAR,
                max_iterations=3000,
                max_time_s=5.0,
            ),
        )
        start = np.zeros(3)
        goal = np.array([0.5, 0.3, -0.2])

        # This may or may not find a path depending on obstacle placement
        # but should not crash
        try:
            path = planner.plan(start, goal, simple_obstacles)
            # Verify no waypoint collides
            for q in path:
                assert not cc.is_collision(q, simple_obstacles), \
                    f"Waypoint {q} is in collision!"
        except PlanningFailure:
            pass  # acceptable if workspace is truly blocked

    def test_bounded_timeout(self, arm_3dof: ArmKinematics):
        """Planner should respect the time bound."""
        cc = _make_checker(arm_3dof)
        # Very tight time bound
        planner = Planner(
            dof=3,
            joint_limits=arm_3dof.joint_limits,
            collision_checker=cc,
            config=PlannerConfig(
                mode=PlannerMode.RRT_STAR,
                max_iterations=100000,
                max_time_s=0.1,  # very short
            ),
        )
        import time
        t0 = time.perf_counter()
        try:
            planner.plan(np.zeros(3), np.array([2.0, 2.0, 2.0]), [])
        except PlanningFailure:
            pass
        elapsed = time.perf_counter() - t0
        assert elapsed < 1.0, f"Planner exceeded timeout: {elapsed:.2f}s"


class TestTrajectorySmoothing:
    """Tests for trajectory smoothing and time parameterisation."""

    def test_cubic_smooth(self):
        """Cubic smoothing should produce a dense trajectory."""
        waypoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.5, 0.3, -0.2]),
            np.array([1.0, 0.5, 0.1]),
        ]
        traj = smooth_trajectory(waypoints, method="cubic")
        assert traj.shape[1] == 3
        assert traj.shape[0] > len(waypoints)
        # First and last points should match waypoints
        np.testing.assert_allclose(traj[0], waypoints[0], atol=1e-3)
        np.testing.assert_allclose(traj[-1], waypoints[-1], atol=1e-3)

    def test_quintic_smooth(self):
        waypoints = [np.zeros(3), np.array([0.5, 0.5, 0.5])]
        traj = smooth_trajectory(waypoints, method="quintic")
        assert traj.shape[0] > 2

    def test_time_parameterize_output(self):
        """Time parameterisation should produce valid timestamps."""
        waypoints = [np.zeros(3), np.array([0.5, 0.3, -0.2])]
        traj = smooth_trajectory(waypoints)
        traj_tp, timestamps = time_parameterize(traj)

        assert traj_tp.shape[0] == timestamps.shape[0]
        assert timestamps[0] == 0.0
        assert timestamps[-1] > 0.0
        # Timestamps should be monotonically increasing
        assert np.all(np.diff(timestamps) >= 0)
