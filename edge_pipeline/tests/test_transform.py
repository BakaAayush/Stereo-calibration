# =============================================================================
# test_transform.py — Unit tests for coordinate transforms
# =============================================================================
import numpy as np
import pytest

from src.transform.transforms import (
    pixel_depth_to_camera,
    camera_to_base,
    batch_pixel_to_base,
    TransformWorkspace,
)


class TestPixelDepthToCamera:
    """Tests for pixel + depth → camera frame conversion."""

    def test_principal_point_unit_depth(self, intrinsics):
        """At the principal point with depth=1, camera coords should be (0, 0, 1)."""
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        result = pixel_depth_to_camera(cx, cy, 1.0, intrinsics)
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1e-10)

    def test_known_offset(self, intrinsics):
        """One pixel right and one pixel down from principal point at depth=1."""
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        result = pixel_depth_to_camera(cx + 1, cy + 1, 1.0, intrinsics)
        expected = np.array([1.0 / fx, 1.0 / fy, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_depth_scaling(self, intrinsics):
        """Camera X, Y should scale linearly with depth."""
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        r1 = pixel_depth_to_camera(cx + 100, cy + 50, 1.0, intrinsics)
        r2 = pixel_depth_to_camera(cx + 100, cy + 50, 2.0, intrinsics)
        np.testing.assert_allclose(r2[:2], r1[:2] * 2.0, atol=1e-10)

    def test_batch(self, intrinsics):
        """Batch transform of N points."""
        u = np.array([640, 400, 900], dtype=np.float64)
        v = np.array([360, 300, 500], dtype=np.float64)
        z = np.array([0.35, 0.42, 0.28], dtype=np.float64)
        result = pixel_depth_to_camera(u, v, z, intrinsics)
        assert result.shape == (3, 3)


class TestCameraToBase:
    """Tests for camera → base frame transformation."""

    def test_identity_transform(self, T_identity):
        """With identity transform, camera = base."""
        point = np.array([0.1, 0.2, 0.3])
        result = camera_to_base(point, T_identity)
        np.testing.assert_allclose(result, point, atol=1e-10)

    def test_translation_only(self):
        """Pure translation: base origin shifted from camera."""
        T = np.eye(4)
        T[:3, 3] = [1.0, 2.0, 3.0]
        point = np.array([0.0, 0.0, 0.0])
        result = camera_to_base(point, T)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], atol=1e-10)

    def test_batch_transform(self, T_identity):
        """Batch of N points."""
        pts = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
        result = camera_to_base(pts, T_identity)
        np.testing.assert_allclose(result, pts, atol=1e-10)

    def test_round_trip(self, intrinsics, T_identity):
        """pixel → camera → base → verify consistency."""
        u, v, z = 500.0, 400.0, 0.5
        cam = pixel_depth_to_camera(u, v, z, intrinsics)
        base = camera_to_base(cam, T_identity)
        np.testing.assert_allclose(base, cam, atol=1e-10)


class TestBatchPixelToBase:
    """Test end-to-end batch transform."""

    def test_batch(self, intrinsics, T_identity):
        u = np.array([640, 400], dtype=np.float64)
        v = np.array([360, 300], dtype=np.float64)
        z = np.array([0.35, 0.42], dtype=np.float64)
        result = batch_pixel_to_base(u, v, z, intrinsics, T_identity)
        assert result.shape == (2, 3)


class TestTransformWorkspace:
    """Test the pre-allocated workspace for hot-loop reuse."""

    def test_workspace_matches_functional(self, intrinsics, T_identity):
        ws = TransformWorkspace(max_points=10)
        u = np.array([640, 400, 900], dtype=np.float64)
        v = np.array([360, 300, 500], dtype=np.float64)
        z = np.array([0.35, 0.42, 0.28], dtype=np.float64)

        expected = batch_pixel_to_base(u, v, z, intrinsics, T_identity)
        result = ws.pixel_to_base(u, v, z, intrinsics, T_identity)
        np.testing.assert_allclose(result, expected, atol=1e-10)
