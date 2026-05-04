import numpy as np

from imu_tracker.fusion import MadgwickImuFusion, normalize_quaternion


def test_quaternion_normalization() -> None:
    q = normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))

    assert np.allclose(q, np.array([1.0, 0.0, 0.0, 0.0]))


def test_stationary_orientation_stays_level() -> None:
    fusion = MadgwickImuFusion(sample_rate_hz=100.0, beta=0.08)

    orientation = None
    for _ in range(200):
        orientation = fusion.update(
            accel_g=np.array([0.0, 0.0, 1.0]),
            gyro_dps=np.array([0.0, 0.0, 0.0]),
        )

    assert orientation is not None
    assert abs(orientation.roll) < 1.0
    assert abs(orientation.pitch) < 1.0
