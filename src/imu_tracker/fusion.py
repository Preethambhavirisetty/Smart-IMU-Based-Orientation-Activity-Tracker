from __future__ import annotations

import math
from typing import cast

import numpy as np

from imu_tracker.models import Orientation, QuaternionArray, Vector3


def normalize_quaternion(q: QuaternionArray) -> QuaternionArray:
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("quaternion norm cannot be zero")
    return q / norm


def quaternion_to_euler_degrees(q: QuaternionArray) -> tuple[float, float, float]:
    w, x, y, z = normalize_quaternion(q)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def quaternion_to_rotation_matrix(q: QuaternionArray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class MadgwickImuFusion:
    """Madgwick IMU filter using accelerometer and gyroscope data.

    This implementation omits magnetometer correction, so yaw is integrated from gyro and can
    drift over time. Roll and pitch are gravity-corrected.
    """

    def __init__(self, sample_rate_hz: float, beta: float = 0.08) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self._sample_period = 1.0 / sample_rate_hz
        self._beta = beta
        self._q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def quaternion(self) -> QuaternionArray:
        return cast(QuaternionArray, self._q.copy())

    def update(self, accel_g: Vector3, gyro_dps: Vector3) -> Orientation:
        accel = np.asarray(accel_g, dtype=np.float64)
        gyro_rad = np.radians(np.asarray(gyro_dps, dtype=np.float64))
        accel_norm = float(np.linalg.norm(accel))

        q1, q2, q3, q4 = self._q
        gx, gy, gz = gyro_rad

        q_dot = 0.5 * np.array(
            [
                -q2 * gx - q3 * gy - q4 * gz,
                q1 * gx + q3 * gz - q4 * gy,
                q1 * gy - q2 * gz + q4 * gx,
                q1 * gz + q2 * gy - q3 * gx,
            ],
            dtype=np.float64,
        )

        if accel_norm > 0.0:
            ax, ay, az = accel / accel_norm
            gradient = self._gradient(q1, q2, q3, q4, ax, ay, az)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm > 0.0:
                q_dot -= self._beta * (gradient / gradient_norm)

        self._q = normalize_quaternion(self._q + q_dot * self._sample_period)
        roll, pitch, yaw = quaternion_to_euler_degrees(self._q)
        return Orientation(quaternion=self.quaternion, roll=roll, pitch=pitch, yaw=yaw)

    @staticmethod
    def _gradient(
        q1: float,
        q2: float,
        q3: float,
        q4: float,
        ax: float,
        ay: float,
        az: float,
    ) -> QuaternionArray:
        f = np.array(
            [
                2.0 * (q2 * q4 - q1 * q3) - ax,
                2.0 * (q1 * q2 + q3 * q4) - ay,
                2.0 * (0.5 - q2 * q2 - q3 * q3) - az,
            ],
            dtype=np.float64,
        )
        jacobian = np.array(
            [
                [-2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2],
                [2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3],
                [0.0, -4.0 * q2, -4.0 * q3, 0.0],
            ],
            dtype=np.float64,
        )
        return cast(QuaternionArray, jacobian.T @ f)
