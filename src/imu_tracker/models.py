from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
QuaternionArray = NDArray[np.float64]


class ActivityLabel(str, Enum):
    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    FALL = "fall"
    GESTURE = "gesture"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ImuSample:
    timestamp: float
    accel_g: Vector3
    gyro_dps: Vector3

    @classmethod
    def from_values(
        cls,
        timestamp: float,
        accel_x: float,
        accel_y: float,
        accel_z: float,
        gyro_x: float,
        gyro_y: float,
        gyro_z: float,
    ) -> ImuSample:
        return cls(
            timestamp=timestamp,
            accel_g=np.array([accel_x, accel_y, accel_z], dtype=np.float64),
            gyro_dps=np.array([gyro_x, gyro_y, gyro_z], dtype=np.float64),
        )


@dataclass(frozen=True)
class Orientation:
    quaternion: QuaternionArray
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class TrackerState:
    sample: ImuSample
    filtered_accel_g: Vector3
    filtered_gyro_dps: Vector3
    orientation: Orientation
    activity: ActivityLabel
