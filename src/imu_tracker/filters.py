from __future__ import annotations

import math
from typing import cast

import numpy as np

from imu_tracker.models import Vector3


class FirstOrderLowPass:
    """Stateful first-order low-pass filter for scalar or vector samples."""

    def __init__(self, cutoff_hz: float, sample_rate_hz: float) -> None:
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        dt = 1.0 / sample_rate_hz
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        self._alpha = dt / (rc + dt)
        self._state: Vector3 | None = None

    def update(self, value: Vector3) -> Vector3:
        value = np.asarray(value, dtype=np.float64)
        if self._state is None:
            self._state = cast(Vector3, value.copy())
        else:
            self._state = self._state + self._alpha * (value - self._state)
        return cast(Vector3, self._state.copy())


class FirstOrderHighPass:
    """Stateful first-order high-pass filter for scalar or vector samples."""

    def __init__(self, cutoff_hz: float, sample_rate_hz: float) -> None:
        if cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")

        dt = 1.0 / sample_rate_hz
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        self._alpha = rc / (rc + dt)
        self._previous_input: Vector3 | None = None
        self._previous_output: Vector3 | None = None

    def update(self, value: Vector3) -> Vector3:
        value = np.asarray(value, dtype=np.float64)
        if self._previous_input is None or self._previous_output is None:
            self._previous_input = cast(Vector3, value.copy())
            self._previous_output = np.zeros_like(value)
            return cast(Vector3, self._previous_output.copy())

        output = self._alpha * (self._previous_output + value - self._previous_input)
        self._previous_input = cast(Vector3, value.copy())
        self._previous_output = cast(Vector3, output.copy())
        return output
