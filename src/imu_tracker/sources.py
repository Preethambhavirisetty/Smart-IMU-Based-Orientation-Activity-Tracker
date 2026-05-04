from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TextIO

import numpy as np

from imu_tracker.models import ImuSample


class ImuSource:
    def samples(self) -> Iterator[ImuSample]:
        raise NotImplementedError


@dataclass(frozen=True)
class SimulatedImuSource(ImuSource):
    sample_rate_hz: float = 100.0
    duration_seconds: float | None = 20.0
    realtime: bool = True
    seed: int = 7

    def samples(self) -> Iterator[ImuSample]:
        rng = np.random.default_rng(self.seed)
        dt = 1.0 / self.sample_rate_hz
        total_samples = None
        if self.duration_seconds is not None:
            total_samples = int(self.duration_seconds * self.sample_rate_hz)

        index = 0
        started_at = time.monotonic()
        while total_samples is None or index < total_samples:
            t = index * dt
            phase = self._phase(t)
            accel = self._accel_for_phase(phase, t) + rng.normal(0.0, 0.015, size=3)
            gyro = self._gyro_for_phase(phase, t) + rng.normal(0.0, 0.4, size=3)
            yield ImuSample(timestamp=t, accel_g=accel, gyro_dps=gyro)

            index += 1
            if self.realtime:
                target = started_at + index * dt
                sleep_for = target - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)

    @staticmethod
    def _phase(t: float) -> str:
        if t < 4:
            return "idle"
        if t < 9:
            return "walk"
        if t < 14:
            return "run"
        if t < 16:
            return "gesture"
        if t < 17:
            return "fall"
        return "idle"

    @staticmethod
    def _accel_for_phase(phase: str, t: float) -> np.ndarray:
        if phase == "walk":
            return np.array([0.12 * math.sin(2 * math.pi * 1.8 * t), 0.02, 1.0], dtype=np.float64)
        if phase == "run":
            return np.array([0.38 * math.sin(2 * math.pi * 3.0 * t), 0.05, 1.0], dtype=np.float64)
        if phase == "fall":
            return np.array([0.5, 0.2, 2.8 if 16.2 < t < 16.35 else 0.25], dtype=np.float64)
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    @staticmethod
    def _gyro_for_phase(phase: str, t: float) -> np.ndarray:
        if phase == "gesture":
            return np.array([35.0, 220.0 * math.sin(2 * math.pi * 2.0 * t), 20.0], dtype=np.float64)
        if phase == "run":
            return np.array([12.0, 5.0, 18.0 * math.sin(2 * math.pi * 2.0 * t)], dtype=np.float64)
        if phase == "walk":
            return np.array([6.0, 3.0, 8.0 * math.sin(2 * math.pi * 1.6 * t)], dtype=np.float64)
        return np.array([0.0, 0.0, 6.0], dtype=np.float64)


@dataclass(frozen=True)
class CsvImuSource(ImuSource):
    file: TextIO

    def samples(self) -> Iterator[ImuSample]:
        reader = csv.DictReader(self.file)
        for row in reader:
            yield ImuSample.from_values(
                timestamp=float(row["timestamp"]),
                accel_x=float(row["accel_x"]),
                accel_y=float(row["accel_y"]),
                accel_z=float(row["accel_z"]),
                gyro_x=float(row["gyro_x"]),
                gyro_y=float(row["gyro_y"]),
                gyro_z=float(row["gyro_z"]),
            )


@dataclass(frozen=True)
class SerialImuSource(ImuSource):
    port: str
    baud: int = 115200

    def samples(self) -> Iterator[ImuSample]:
        import serial

        with serial.Serial(self.port, self.baud, timeout=1) as connection:
            while True:
                line = connection.readline().decode("utf-8", errors="replace").strip()
                if not line or line.startswith("timestamp"):
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 7:
                    continue
                yield ImuSample.from_values(*(float(part) for part in parts))
