# Smart IMU-Based Orientation & Activity Tracker

A clean Python project for learning and prototyping an IMU-based motion tracker with:

- Raw accelerometer and gyroscope ingestion
- Low-pass and high-pass signal filtering
- Madgwick sensor fusion for real-time orientation
- Activity detection for idle, walking, running, falling, and gestures
- 3D orientation visualization with Matplotlib
- Simulated IMU stream for development without hardware
- Optional serial input for MPU6050 / LSM6DSOX firmware output

## Project Layout

```text
.
├── pyproject.toml
├── README.md
├── examples/
│   └── sample_serial_format.csv
├── src/
│   └── imu_tracker/
│       ├── __main__.py
│       ├── activity.py
│       ├── cli.py
│       ├── config.py
│       ├── filters.py
│       ├── fusion.py
│       ├── models.py
│       ├── pipeline.py
│       ├── sources.py
│       └── visualization.py
└── tests/
    ├── test_activity.py
    ├── test_filters.py
    └── test_fusion.py
```

## Quick Start

Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run a simulated real-time tracker:

```bash
imu-tracker simulate --duration 20 --visualize
```

Run without visualization and print activity/orientation:

```bash
imu-tracker simulate --duration 10
```

Use serial input from a microcontroller:

```bash
imu-tracker serial --port /dev/tty.usbmodem1101 --baud 115200 --visualize
```

Expected serial line format:

```csv
timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
```

Accelerometer units are `g`; gyroscope units are `deg/s`.

## Hardware Notes

For an MPU6050 or LSM6DSOX, your firmware should:

1. Configure accelerometer range, for example `+-2g` or `+-4g`.
2. Configure gyroscope range, for example `+-250 dps` or `+-500 dps`.
3. Calibrate gyro bias while the device is stationary.
4. Stream timestamped samples as CSV over USB serial or Bluetooth UART.

## Development

Run tests:

```bash
pytest
```

Format and lint:

```bash
ruff check src tests
ruff format src tests
```

Type check:

```bash
mypy src
```

