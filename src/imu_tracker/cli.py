from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator

from imu_tracker.config import FusionConfig, TrackerConfig
from imu_tracker.models import ImuSample
from imu_tracker.pipeline import ImuTracker
from imu_tracker.sources import CsvImuSource, ImuSource, SerialImuSource, SimulatedImuSource


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imu-tracker",
        description="Smart IMU orientation and activity tracker.",
    )
    parser.add_argument("--sample-rate", type=float, default=100.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Run with generated IMU data.")
    simulate.add_argument("--duration", type=float, default=20.0)
    simulate.add_argument("--no-realtime", action="store_true")
    simulate.add_argument("--visualize", action="store_true")

    serial = subparsers.add_parser("serial", help="Run with CSV data from a serial port.")
    serial.add_argument("--port", required=True)
    serial.add_argument("--baud", type=int, default=115200)
    serial.add_argument("--visualize", action="store_true")

    csv_parser = subparsers.add_parser("csv", help="Run with a CSV file.")
    csv_parser.add_argument("path")
    csv_parser.add_argument("--visualize", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = TrackerConfig(fusion=FusionConfig(sample_rate_hz=args.sample_rate))
    tracker = ImuTracker(config)

    if args.command == "simulate":
        source: ImuSource = SimulatedImuSource(
            sample_rate_hz=args.sample_rate,
            duration_seconds=args.duration,
            realtime=not args.no_realtime,
        )
        return _run_tracker(tracker, source.samples(), visualize=args.visualize)

    if args.command == "serial":
        source = SerialImuSource(port=args.port, baud=args.baud)
        return _run_tracker(tracker, source.samples(), visualize=args.visualize)

    if args.command == "csv":
        with open(args.path, encoding="utf-8", newline="") as file:
            source = CsvImuSource(file)
            return _run_tracker(tracker, source.samples(), visualize=args.visualize)

    return 2


def _run_tracker(tracker: ImuTracker, samples: Iterator[ImuSample], visualize: bool) -> int:
    visualizer = None
    if visualize:
        from imu_tracker.visualization import OrientationVisualizer

        visualizer = OrientationVisualizer()
    try:
        for state in tracker.run(samples):
            if visualizer is not None:
                visualizer.update(state)
            else:
                print(
                    f"{state.sample.timestamp:8.3f}s  "
                    f"{state.activity.value:8s}  "
                    f"roll={state.orientation.roll:7.2f}  "
                    f"pitch={state.orientation.pitch:7.2f}  "
                    f"yaw={state.orientation.yaw:7.2f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
