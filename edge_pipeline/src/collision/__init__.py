# src.collision — Collision checking with capsule/sphere approximations
"""Vectorized collision detection for robot arm link geometries."""

from .checker import CollisionChecker, Obstacle

__all__ = ["CollisionChecker", "Obstacle"]
