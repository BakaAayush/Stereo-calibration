# =============================================================================
# mock_camera.py — Synthetic depth source for testing
# =============================================================================
# Purpose:  Provide a deterministic DepthSource that generates synthetic
#           detections at known pixel/depth locations.  Used by unit tests
#           and the mock end-to-end example.
#
# Inputs:   Configuration (image size, intrinsics, detection list).
# Outputs:  DepthFrame with synthetic data.
# Complexity: O(N) where N = number of synthetic detections per frame.
# =============================================================================
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .camera_interface import Detection, DepthFrame, DepthSource


class MockCamera:
    """Generates synthetic depth frames for testing.

    Parameters
    ----------
    detections : list[Detection]
        Fixed list of detections returned every frame.
    image_size : tuple[int, int]
        (width, height) of the simulated image.
    fps : float
        Simulated frame-rate — ``get_frame`` sleeps to match this rate.
    """

    def __init__(
        self,
        detections: Sequence[Detection] | None = None,
        image_size: tuple[int, int] = (1280, 720),
        fps: float = 15.0,
    ) -> None:
        self._detections = list(detections) if detections else self._default_detections()
        self._w, self._h = image_size
        self._period = 1.0 / max(fps, 1.0)
        self._last_t = 0.0
        self._frame_idx = 0

    # --------------------------------------------------------------------- #
    # DepthSource protocol
    # --------------------------------------------------------------------- #
    def get_frame(self) -> DepthFrame:
        """Return a synthetic frame, throttled to the configured FPS."""
        now = time.monotonic()
        wait = self._period - (now - self._last_t)
        if wait > 0:
            time.sleep(wait)
        self._last_t = time.monotonic()

        # Build a simple depth map (plane at 0.5 m with random noise)
        depth_map = np.full((self._h, self._w), 0.5, dtype=np.float32)
        depth_map += np.random.default_rng(self._frame_idx).normal(0, 0.005, depth_map.shape).astype(np.float32)

        self._frame_idx += 1
        return DepthFrame(
            timestamp=self._last_t,
            detections=self._detections,
            depth_map=depth_map,
            rgb=None,
        )

    def release(self) -> None:
        """No-op for mock camera."""
        pass

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _default_detections() -> list[Detection]:
        """Provide a small set of canned detections for quick testing."""
        return [
            Detection(label="bolt", u=640.0, v=360.0, depth_m=0.35, confidence=0.92),
            Detection(label="nut", u=400.0, v=300.0, depth_m=0.42, confidence=0.87),
            Detection(label="cup", u=900.0, v=500.0, depth_m=0.28, confidence=0.95),
        ]
