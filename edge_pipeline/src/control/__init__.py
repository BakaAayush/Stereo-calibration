# src.control — Actuator hardware abstraction
"""Actuator interface, PCA9685 servo driver, and mock actuator."""

from .actuator_interface import ActuatorInterface
from .mock_actuator import MockActuator

__all__ = ["ActuatorInterface", "MockActuator"]
