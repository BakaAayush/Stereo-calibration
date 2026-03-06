# =============================================================================
# pca9685_driver.py — PCA9685 I2C servo driver
# =============================================================================
# Purpose:  Low-level driver for PCA9685 16-channel PWM controller, commonly
#           used to drive hobby servos on robot arms.
#
# Hardware: PCA9685 connected to Raspberry Pi via I2C (bus 1 by default).
#           Each channel drives one servo (SG90 / MG996R / etc.).
#
# Features:
#   - PWM-to-angle calibration table per channel.
#   - Angle-to-PWM mapping using linear interpolation.
#   - Safety: rate limiting inherited from ActuatorInterface.
#
# ASSUMPTION: Default calibration assumes SG90-type servos:
#   0°   → 500 µs pulse width
#   180° → 2500 µs pulse width
#   at 50 Hz PWM frequency.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .actuator_interface import ActuatorInterface

logger = logging.getLogger(__name__)

# ── Try importing smbus2 for real I2C ────────────────────────────────────
_HAS_SMBUS = False
try:
    import smbus2
    _HAS_SMBUS = True
except ImportError:
    smbus2 = None  # type: ignore[assignment]

# PCA9685 register addresses
_MODE1 = 0x00
_PRESCALE = 0xFE
_LED0_ON_L = 0x06


@dataclass(frozen=True, slots=True)
class ServoCalibration:
    """Calibration for one servo channel.

    Attributes
    ----------
    channel : int
        PCA9685 channel (0–15).
    min_pulse_us : float
        Pulse width in µs at minimum angle.
    max_pulse_us : float
        Pulse width in µs at maximum angle.
    min_angle_rad : float
        Joint angle corresponding to min pulse.
    max_angle_rad : float
        Joint angle corresponding to max pulse.
    """

    channel: int
    min_pulse_us: float = 500.0
    max_pulse_us: float = 2500.0
    min_angle_rad: float = -np.pi / 2
    max_angle_rad: float = np.pi / 2


class PCA9685Driver(ActuatorInterface):
    """PCA9685-based servo driver for hobby servos.

    Parameters
    ----------
    calibrations : list[ServoCalibration]
        Per-joint calibration (one per DOF, in joint order).
    i2c_bus : int
        I2C bus number (1 on Raspberry Pi).
    i2c_address : int
        PCA9685 I2C address (default 0x40).
    pwm_freq_hz : int
        PWM frequency (typically 50 Hz for servos).
    """

    def __init__(
        self,
        calibrations: Sequence[ServoCalibration] | None = None,
        dof: int = 3,
        joint_limits: NDArray | None = None,
        i2c_bus: int = 1,
        i2c_address: int = 0x40,
        pwm_freq_hz: int = 50,
        max_rate: float = 1.5,
    ) -> None:
        # Default calibrations
        if calibrations is None:
            calibrations = [
                ServoCalibration(channel=i) for i in range(dof)
            ]
        self.calibrations = list(calibrations)
        actual_dof = len(self.calibrations)

        if joint_limits is None:
            joint_limits = np.array(
                [[c.min_angle_rad, c.max_angle_rad] for c in self.calibrations],
                dtype=np.float64,
            )

        super().__init__(
            dof=actual_dof,
            joint_limits=joint_limits,
            max_rate=max_rate,
        )

        self._address = i2c_address
        self._freq = pwm_freq_hz
        self._bus = None

        # Initialise hardware
        if _HAS_SMBUS:
            try:
                self._bus = smbus2.SMBus(i2c_bus)
                self._init_pca9685()
                logger.info("PCA9685 initialised on bus %d, addr 0x%02X", i2c_bus, i2c_address)
            except Exception as e:
                logger.warning("PCA9685 init failed: %s — running in dry-run mode", e)
                self._bus = None
        else:
            logger.warning("smbus2 not available — PCA9685 in dry-run mode")

    # ------------------------------------------------------------------ #
    # PCA9685 low-level init
    # ------------------------------------------------------------------ #
    def _init_pca9685(self) -> None:
        """Reset and configure the PCA9685."""
        if self._bus is None:
            return

        # Reset
        self._bus.write_byte_data(self._address, _MODE1, 0x00)

        # Set PWM frequency
        prescale = int(round(25_000_000.0 / (4096 * self._freq) - 1))
        old_mode = self._bus.read_byte_data(self._address, _MODE1)
        self._bus.write_byte_data(self._address, _MODE1, (old_mode & 0x7F) | 0x10)  # sleep
        self._bus.write_byte_data(self._address, _PRESCALE, prescale)
        self._bus.write_byte_data(self._address, _MODE1, old_mode)

        import time
        time.sleep(0.005)
        self._bus.write_byte_data(self._address, _MODE1, old_mode | 0xA1)

    # ------------------------------------------------------------------ #
    # Angle ↔ PWM conversion
    # ------------------------------------------------------------------ #
    def _angle_to_pwm(self, angle_rad: float, cal: ServoCalibration) -> int:
        """Convert joint angle (rad) to 12-bit PCA9685 tick count."""
        # Linear interpolation
        t = (angle_rad - cal.min_angle_rad) / (cal.max_angle_rad - cal.min_angle_rad)
        t = np.clip(t, 0.0, 1.0)
        pulse_us = cal.min_pulse_us + t * (cal.max_pulse_us - cal.min_pulse_us)

        # Convert µs to 12-bit tick (period = 1e6 / freq µs, 4096 ticks per period)
        period_us = 1_000_000.0 / self._freq
        tick = int(round(pulse_us / period_us * 4096))
        return np.clip(tick, 0, 4095)

    def _set_pwm(self, channel: int, on: int, off: int) -> None:
        """Write PWM on/off ticks for a channel."""
        if self._bus is None:
            return
        reg = _LED0_ON_L + 4 * channel
        self._bus.write_byte_data(self._address, reg, on & 0xFF)
        self._bus.write_byte_data(self._address, reg + 1, on >> 8)
        self._bus.write_byte_data(self._address, reg + 2, off & 0xFF)
        self._bus.write_byte_data(self._address, reg + 3, off >> 8)

    # ------------------------------------------------------------------ #
    # ActuatorInterface implementation
    # ------------------------------------------------------------------ #
    def _send_angles(self, angles: NDArray) -> None:
        for i, (angle, cal) in enumerate(zip(angles, self.calibrations)):
            tick = self._angle_to_pwm(float(angle), cal)
            self._set_pwm(cal.channel, 0, tick)

    def _read_angles(self) -> NDArray:
        # PCA9685 is open-loop; return last commanded angles
        return self._last_angles.copy()

    def close(self) -> None:
        """Release the I2C bus."""
        if self._bus is not None:
            # Set all channels off
            for cal in self.calibrations:
                self._set_pwm(cal.channel, 0, 0)
            self._bus.close()
            self._bus = None
            logger.info("PCA9685 driver closed")
