# src.service — Headless daemon / main loop
"""Main pipeline service orchestrating detection→IK→planning→control."""

from .daemon import EdgePipelineService

__all__ = ["EdgePipelineService"]
