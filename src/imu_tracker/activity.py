from __future__ import annotations

from collections import deque

import numpy as np

from imu_tracker.config import ActivityConfig
from imu_tracker.models import ActivityLabel, ImuSample


class ActivityDetector:
    """Windowed heuristic detector for common wearable IMU activities."""

    def __init__(self, config: ActivityConfig, sample_rate_hz: float) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self._config = config
        self._max_samples = max(4, int(config.window_seconds * sample_rate_hz))
        self._accel_magnitudes: deque[float] = deque(maxlen=self._max_samples)
        self._gyro_magnitudes: deque[float] = deque(maxlen=self._max_samples)

    def update(self, sample: ImuSample) -> ActivityLabel:
        accel_mag = float(np.linalg.norm(sample.accel_g))
        gyro_mag = float(np.linalg.norm(sample.gyro_dps))
        self._accel_magnitudes.append(accel_mag)
        self._gyro_magnitudes.append(gyro_mag)

        if accel_mag >= self._config.fall_accel_g:
            return ActivityLabel.FALL
        if gyro_mag >= self._config.gesture_gyro_dps:
            return ActivityLabel.GESTURE
        if len(self._accel_magnitudes) < self._max_samples // 2:
            return ActivityLabel.UNKNOWN

        accel_std = float(np.std(np.array(self._accel_magnitudes, dtype=np.float64)))
        if accel_std < self._config.idle_std_g:
            return ActivityLabel.IDLE
        if accel_std >= self._config.running_std_g:
            return ActivityLabel.RUNNING
        if accel_std >= self._config.walking_std_g:
            return ActivityLabel.WALKING
        return ActivityLabel.UNKNOWN
