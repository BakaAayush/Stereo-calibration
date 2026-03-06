# src.camera — Camera interface and mock depth source
"""Depth-source abstractions. The real depth pipeline (stereo SGBM) is external;
this module defines the callback API consumed by the edge pipeline."""

from .camera_interface import DepthSource, Detection
from .mock_camera import MockCamera

__all__ = ["DepthSource", "Detection", "MockCamera"]
