from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt

from imu_tracker.fusion import quaternion_to_rotation_matrix
from imu_tracker.models import TrackerState


class OrientationVisualizer:
    def __init__(self) -> None:
        self._figure = plt.figure("IMU Orientation & Activity Tracker")
        self._axis = self._figure.add_subplot(111, projection="3d")
        self._configure_axis(self._axis)
        self._text = self._axis.text2D(0.02, 0.95, "", transform=self._axis.transAxes)

    def update(self, state: TrackerState) -> None:
        self._axis.cla()
        self._configure_axis(self._axis)
        rotation = quaternion_to_rotation_matrix(state.orientation.quaternion)
        basis = rotation @ np.eye(3)
        colors = ["#d1495b", "#00798c", "#edae49"]
        labels = ["X", "Y", "Z"]
        for vector, color, label in zip(basis.T, colors, labels, strict=True):
            self._axis.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=color, linewidth=3)
            self._axis.text(vector[0] * 1.1, vector[1] * 1.1, vector[2] * 1.1, label, color=color)

        roll = state.orientation.roll
        pitch = state.orientation.pitch
        yaw = state.orientation.yaw
        self._text = self._axis.text2D(
            0.02,
            0.95,
            f"Activity: {state.activity.value}\n"
            f"Roll: {roll:6.1f}  Pitch: {pitch:6.1f}  Yaw: {yaw:6.1f}",
            transform=self._axis.transAxes,
        )
        plt.pause(0.001)

    @staticmethod
    def _configure_axis(axis: Any) -> None:
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)
        axis.set_zlim(-1.2, 1.2)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.set_box_aspect((1, 1, 1))
        axis.grid(True, alpha=0.25)
