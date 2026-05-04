import numpy as np

from imu_tracker.filters import FirstOrderHighPass, FirstOrderLowPass


def test_lowpass_converges_to_constant_signal() -> None:
    filt = FirstOrderLowPass(cutoff_hz=5.0, sample_rate_hz=100.0)
    value = np.array([1.0, -2.0, 0.5])

    output = value
    for _ in range(100):
        output = filt.update(value)

    assert np.allclose(output, value, atol=1e-3)


def test_highpass_removes_constant_signal() -> None:
    filt = FirstOrderHighPass(cutoff_hz=0.5, sample_rate_hz=100.0)
    value = np.array([1.0, 1.0, 1.0])

    output = value
    for _ in range(300):
        output = filt.update(value)

    assert np.linalg.norm(output) < 1e-3
