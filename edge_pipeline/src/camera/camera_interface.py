# =============================================================================
# camera_interface.py — Abstract depth-source protocol
# =============================================================================
# Purpose:  Define the contract between the external depth-sensing pipeline
#           and the edge kinematic pipeline.  The depth pipeline (stereo SGBM)
#           produces 2-D pixel coordinates + depth value for each detected
#           object.  This module defines the data structures and the abstract
#           protocol that any depth source must satisfy.
#
# Inputs:   None directly — implementations provide concrete data.
# Outputs:  Detection dataclass and DepthSource protocol.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection with associated depth.

    Attributes
    ----------
    label : str
        Class name (e.g. ``"bolt"``, ``"cup"``).
    u : float
        Horizontal pixel coordinate (column) in the rectified left image.
    v : float
        Vertical pixel coordinate (row) in the rectified left image.
    depth_m : float
        Depth in **metres** from the camera optical centre, obtained from
        the stereo disparity map.
    confidence : float
        Detection confidence in [0, 1].
    bbox : tuple[int, int, int, int] | None
        Optional bounding box ``(x1, y1, x2, y2)`` in pixel coords.
    """

    label: str
    u: float
    v: float
    depth_m: float
    confidence: float = 1.0
    bbox: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class DepthFrame:
    """Container returned by a DepthSource on each call to `get_frame`.

    Attributes
    ----------
    timestamp : float
        Monotonic timestamp (seconds) of the capture.
    detections : list[Detection]
        Detected objects with pixel + depth information.
    depth_map : np.ndarray | None
        Full HxW float32 depth map in metres (optional; may be None if only
        point detections are available).
    rgb : np.ndarray | None
        Optional HxWx3 uint8 RGB image for visualisation / logging.
    """

    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    depth_map: np.ndarray | None = None
    rgb: np.ndarray | None = None


@runtime_checkable
class DepthSource(Protocol):
    """Protocol that any depth source must implement.

    The edge pipeline calls ``get_frame()`` in a loop (or from a queue
    consumer) to receive the latest detections + depth.
    """

    def get_frame(self) -> DepthFrame | None:
        """Return the latest depth frame, or ``None`` if unavailable."""
        ...

    def release(self) -> None:
        """Release any held resources (cameras, file handles, etc.)."""
        ...
