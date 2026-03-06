# src.kinematics — Forward/Inverse Kinematics for 3–4 DOF arms
"""Kinematics module with roboticstoolbox primary and ikpy fallback backends."""

from .arm_kinematics import ArmKinematics

__all__ = ["ArmKinematics"]
