from __future__ import annotations

from collections.abc import Iterator

from imu_tracker.activity import ActivityDetector
from imu_tracker.config import TrackerConfig
from imu_tracker.filters import FirstOrderHighPass, FirstOrderLowPass
from imu_tracker.fusion import MadgwickImuFusion
from imu_tracker.models import ImuSample, TrackerState


class ImuTracker:
    def __init__(self, config: TrackerConfig) -> None:
        sample_rate = config.fusion.sample_rate_hz
        self._accel_lowpass = FirstOrderLowPass(config.filters.accel_lowpass_cutoff_hz, sample_rate)
        self._gyro_lowpass = FirstOrderLowPass(config.filters.gyro_lowpass_cutoff_hz, sample_rate)
        self._activity_highpass = FirstOrderHighPass(
            config.filters.activity_highpass_cutoff_hz,
            sample_rate,
        )
        self._fusion = MadgwickImuFusion(sample_rate, config.fusion.madgwick_beta)
        self._activity = ActivityDetector(config.activity, sample_rate)

    def update(self, sample: ImuSample) -> TrackerState:
        filtered_accel = self._accel_lowpass.update(sample.accel_g)
        filtered_gyro = self._gyro_lowpass.update(sample.gyro_dps)
        activity_accel = self._activity_highpass.update(filtered_accel) + filtered_accel
        activity_sample = ImuSample(
            timestamp=sample.timestamp,
            accel_g=activity_accel,
            gyro_dps=filtered_gyro,
        )
        orientation = self._fusion.update(filtered_accel, filtered_gyro)
        activity = self._activity.update(activity_sample)
        return TrackerState(
            sample=sample,
            filtered_accel_g=filtered_accel,
            filtered_gyro_dps=filtered_gyro,
            orientation=orientation,
            activity=activity,
        )

    def run(self, samples: Iterator[ImuSample]) -> Iterator[TrackerState]:
        for sample in samples:
            yield self.update(sample)
