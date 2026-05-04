from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterConfig:
    accel_lowpass_cutoff_hz: float = 8.0
    gyro_lowpass_cutoff_hz: float = 20.0
    activity_highpass_cutoff_hz: float = 0.25


@dataclass(frozen=True)
class FusionConfig:
    sample_rate_hz: float = 100.0
    madgwick_beta: float = 0.08


@dataclass(frozen=True)
class ActivityConfig:
    window_seconds: float = 1.25
    idle_std_g: float = 0.035
    walking_std_g: float = 0.12
    running_std_g: float = 0.32
    fall_accel_g: float = 2.4
    gesture_gyro_dps: float = 180.0


@dataclass(frozen=True)
class TrackerConfig:
    filters: FilterConfig = FilterConfig()
    fusion: FusionConfig = FusionConfig()
    activity: ActivityConfig = ActivityConfig()
