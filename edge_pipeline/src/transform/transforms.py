# =============================================================================
# transforms.py — Coordinate-frame transforms (pixel → camera → base)
# =============================================================================
# Purpose:  Convert 2-D pixel + depth into 3-D camera-frame coordinates, then
#           transform into the robot base frame using a 4×4 homogeneous
#           extrinsic matrix.  All math is NumPy-vectorised for throughput.
#
# Key formulas
# ------------
# Pixel → Camera (pinhole model, undistorted):
#   X_c = (u - cx) * Z / fx
#   Y_c = (v - cy) * Z / fy
#   Z_c = Z
#
# Camera → Base:
#   P_base = T_base_camera @ [X_c, Y_c, Z_c, 1]^T
#
# Complexity:  O(N) for N points, fully vectorised — no Python loops.
# =============================================================================
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pixel_depth_to_camera(
    u: float | NDArray,
    v: float | NDArray,
    z: float | NDArray,
    K: NDArray,
) -> NDArray:
    """Convert pixel coordinates + depth to 3-D camera-frame point(s).

    Parameters
    ----------
    u, v : float or (N,) array
        Pixel coordinates (column, row) in the rectified image.
    z : float or (N,) array
        Depth in metres from the camera optical centre.
    K : (3, 3) array
        Camera intrinsic matrix::

            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    Returns
    -------
    points : (3,) or (N, 3) ndarray  —  [X, Y, Z] in camera frame (metres).

    Notes
    -----
    Assumes the image is already rectified (no distortion).  If distortion
    is present, undistort the pixel coordinates first using ``cv2.undistortPoints``.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

    return np.stack([X, Y, Z], axis=-1)


def camera_to_base(
    point_cam: NDArray,
    T_base_camera: NDArray,
) -> NDArray:
    """Transform point(s) from camera frame to robot base frame.

    Parameters
    ----------
    point_cam : (3,) or (N, 3) array
        Point(s) in camera frame [X, Y, Z] (metres).
    T_base_camera : (4, 4) array
        Homogeneous transform from camera frame to base frame.

    Returns
    -------
    point_base : (3,) or (N, 3) ndarray
        Point(s) in base frame.
    """
    pts = np.asarray(point_cam, dtype=np.float64)
    T = np.asarray(T_base_camera, dtype=np.float64)

    single = pts.ndim == 1
    if single:
        pts = pts.reshape(1, 3)

    # Build Nx4 homogeneous coordinates
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])  # (N, 4)

    # Transform:  P_base = T @ P_cam^T  →  (4, N)  →  transpose → (N, 4)
    pts_base_h = (T @ pts_h.T).T  # (N, 4)

    result = pts_base_h[:, :3]
    return result[0] if single else result


def batch_pixel_to_base(
    u: NDArray,
    v: NDArray,
    z: NDArray,
    K: NDArray,
    T_base_camera: NDArray,
) -> NDArray:
    """End-to-end batch: pixel + depth → base-frame 3-D points.

    Parameters
    ----------
    u, v, z : (N,) arrays
    K : (3, 3) intrinsic matrix
    T_base_camera : (4, 4) extrinsic matrix

    Returns
    -------
    (N, 3) ndarray of points in base frame.
    """
    pts_cam = pixel_depth_to_camera(u, v, z, K)
    return camera_to_base(pts_cam, T_base_camera)


# ── Pre-allocated workspace for hot-loop reuse (optional perf tweak) ─────
class TransformWorkspace:
    """Pre-allocated arrays to avoid per-call allocation in tight loops.

    Usage::

        ws = TransformWorkspace(max_points=100)
        result = ws.pixel_to_base(u, v, z, K, T)
    """

    def __init__(self, max_points: int = 256) -> None:
        self._n = max_points
        self._pts_h = np.empty((max_points, 4), dtype=np.float64)
        self._pts_h[:, 3] = 1.0
        self._out = np.empty((max_points, 4), dtype=np.float64)

    def pixel_to_base(
        self, u: NDArray, v: NDArray, z: NDArray, K: NDArray, T: NDArray,
    ) -> NDArray:
        n = len(u)
        assert n <= self._n, f"Batch {n} exceeds workspace capacity {self._n}"

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        self._pts_h[:n, 0] = (u - cx) * z / fx
        self._pts_h[:n, 1] = (v - cy) * z / fy
        self._pts_h[:n, 2] = z

        np.dot(self._pts_h[:n], T.T, out=self._out[:n])
        return self._out[:n, :3].copy()
