import numpy as np

from imu_tracker.activity import ActivityDetector
from imu_tracker.config import ActivityConfig
from imu_tracker.models import ActivityLabel, ImuSample


def test_fall_detection_has_priority() -> None:
    detector = ActivityDetector(ActivityConfig(), sample_rate_hz=100)
    sample = ImuSample(timestamp=0.0, accel_g=np.array([0.0, 0.0, 2.8]), gyro_dps=np.zeros(3))

    assert detector.update(sample) == ActivityLabel.FALL


def test_gesture_detection_uses_gyro_peak() -> None:
    detector = ActivityDetector(ActivityConfig(), sample_rate_hz=100)
    sample = ImuSample(
        timestamp=0.0,
        accel_g=np.array([0.0, 0.0, 1.0]),
        gyro_dps=np.array([0.0, 220.0, 0.0]),
    )

    assert detector.update(sample) == ActivityLabel.GESTURE


def test_idle_detection_after_window() -> None:
    detector = ActivityDetector(ActivityConfig(window_seconds=0.2), sample_rate_hz=100)

    label = ActivityLabel.UNKNOWN
    for index in range(20):
        label = detector.update(
            ImuSample(
                timestamp=index / 100.0,
                accel_g=np.array([0.0, 0.0, 1.0]),
                gyro_dps=np.zeros(3),
            )
        )

    assert label == ActivityLabel.IDLE
