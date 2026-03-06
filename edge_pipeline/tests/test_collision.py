# =============================================================================
# test_collision.py — Unit tests for collision checking
# =============================================================================
import numpy as np
import pytest

from src.collision.checker import CollisionChecker, Obstacle, LinkGeometry


class TestPointSegmentDistance:
    """Test the vectorised point–segment distance function."""

    def test_point_on_segment(self):
        """Distance from a point on the segment should be 0."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        p = np.array([[0.5, 0.0, 0.0]])
        d = CollisionChecker._point_segment_dist_batch(p, a, b)
        np.testing.assert_allclose(d, [0.0], atol=1e-10)

    def test_point_at_endpoint(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d = CollisionChecker._point_segment_dist_batch(p, a, b)
        np.testing.assert_allclose(d, [0.0, 0.0], atol=1e-10)

    def test_point_perpendicular(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        p = np.array([[0.5, 1.0, 0.0]])
        d = CollisionChecker._point_segment_dist_batch(p, a, b)
        np.testing.assert_allclose(d, [1.0], atol=1e-10)

    def test_batch_distances(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        p = np.array([
            [0.5, 0.0, 0.0],  # on segment
            [0.5, 1.0, 0.0],  # 1m away
            [2.0, 0.0, 0.0],  # past endpoint
        ])
        d = CollisionChecker._point_segment_dist_batch(p, a, b)
        np.testing.assert_allclose(d, [0.0, 1.0, 1.0], atol=1e-10)


class TestCollisionChecker:
    """Integration tests for collision checking."""

    def test_no_obstacles(self, arm_3dof):
        """No obstacles should never collide."""
        def fk_frames(q):
            frames = [np.eye(4)]
            for i in range(arm_3dof.dof):
                q_partial = np.zeros(arm_3dof.dof)
                q_partial[:i + 1] = q[:i + 1]
                frames.append(arm_3dof.fk(q_partial))
            return frames

        cc = CollisionChecker([0.02, 0.02, 0.02], fk_frames)
        assert not cc.is_collision(np.zeros(3), [])

    def test_collision_detection(self, arm_3dof):
        """An obstacle at the end-effector should trigger collision."""
        def fk_frames(q):
            frames = [np.eye(4)]
            for i in range(arm_3dof.dof):
                q_partial = np.zeros(arm_3dof.dof)
                q_partial[:i + 1] = q[:i + 1]
                frames.append(arm_3dof.fk(q_partial))
            return frames

        cc = CollisionChecker([0.02, 0.02, 0.02], fk_frames)
        ee_pos = arm_3dof.fk_position(np.zeros(3))
        obstacle = Obstacle(center=tuple(ee_pos), radius=0.05)
        assert cc.is_collision(np.zeros(3), [obstacle])

    def test_path_collision(self, arm_3dof):
        """is_path_collision should detect collision along a path."""
        def fk_frames(q):
            frames = [np.eye(4)]
            for i in range(arm_3dof.dof):
                q_partial = np.zeros(arm_3dof.dof)
                q_partial[:i + 1] = q[:i + 1]
                frames.append(arm_3dof.fk(q_partial))
            return frames

        cc = CollisionChecker([0.02, 0.02, 0.02], fk_frames)
        ee_pos = arm_3dof.fk_position(np.array([0.3, 0.0, 0.0]))
        obstacle = Obstacle(center=tuple(ee_pos), radius=0.05)

        configs = np.linspace([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], 20)
        assert cc.is_path_collision(configs, [obstacle])
