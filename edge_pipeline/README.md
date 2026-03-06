# Edge Pipeline — Vision-Guided Robotic Arm (Edge Compute)

> Full edge-side pipeline for a 3–4 DOF robot arm: coordinate transforms, inverse kinematics, collision-aware path planning, trajectory smoothing, and servo control — optimised for **Raspberry Pi 5 (8 GB)**.

## Quick Start

```bash
# On Raspberry Pi 5 (first time, with internet)
chmod +x install_deps.sh && sudo ./install_deps.sh

# Activate venv
source .venv/bin/activate

# Run mock demo (no hardware needed)
python examples/mock_end_to_end.py

# Run tests
python -m pytest tests/ -v

# Run benchmarks
python examples/benchmark_fps.py
```

## System Overview

```
Depth Pipeline (external)        Edge Pipeline (this repo)
┌─────────────────────┐         ┌──────────────────────────────────────┐
│ Stereo cameras      │         │                                      │
│ + SGBM disparity    │──queue──│ Transform → IK → Plan → Smooth → ── │
│ + Object detection  │         │               ↓           ↓     ↓    │
└─────────────────────┘         │         Collision     CSV/JSON  PCA  │
                                │          check        export   9685  │
                                └──────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| **Depth input** | `src/camera/` | Consumes depth frames via callback API |
| **Transform** | `src/transform/` | Pixel + depth → camera frame → base frame |
| **Kinematics** | `src/kinematics/` | FK / IK with roboticstoolbox + ikpy fallback |
| **Collision** | `src/collision/` | Capsule–sphere collision checker (vectorised) |
| **Planning** | `src/planning/` | APF + bounded RRT* + cubic/quintic smoothing |
| **Control** | `src/control/` | PCA9685 servo driver + MockActuator |
| **Export** | `src/export/` | CSV / JSON trajectory + SCP transfer |
| **Service** | `src/service/` | Headless daemon (live / simulation modes) |

## Hardware Requirements

| Item | Specification |
|------|---------------|
| SBC | Raspberry Pi 5 (8 GB) |
| Power | 27 W USB-C PD supply |
| Cooling | Active fan heatsink |
| OS | Raspberry Pi OS Bookworm (64-bit) |
| Servo driver | PCA9685 (I2C) |
| Robot arm | 3–4 DOF with hobby servos |

## Expected Performance (Pi 5)

| Metric | Target | Notes |
|--------|--------|-------|
| Transform | < 3 ms / point | Vectorised NumPy |
| IK (3-DOF) | < 10 ms median | ikpy / rtb backend |
| Plan (RRT*) | < 2 s | bounded 5000 iterations |
| Overall loop | 10–15 FPS | transform + IK only |

## Project Structure

```
edge_pipeline/
├── src/
│   ├── camera/          # Depth source interface + mock
│   ├── transform/       # Coordinate transforms
│   ├── kinematics/      # FK / IK / Jacobian
│   ├── collision/       # Collision checking
│   ├── planning/        # APF + RRT* + trajectory smoothing
│   ├── control/         # PCA9685 driver + mock
│   ├── export/          # CSV / JSON / SCP export
│   └── service/         # Headless daemon
├── tests/               # pytest suite
├── examples/            # Demo scripts + benchmarks
├── tools/               # YOLO conversion utilities
├── docs/                # System diagrams + MATLAB snippets
├── systemd/             # Systemd unit file
├── ci/                  # GitHub Actions CI
├── install_deps.sh      # Pi provisioning
├── setup_venv.sh        # Venv creation
└── requirements.txt     # Pinned dependencies
```

## Acceptance Tests

1. **`mock_end_to_end.py`** — produces `trajectory.csv`; loadable by `numpy.loadtxt` and MATLAB `readmatrix()`.
2. **Unit tests** — `pytest tests/ -v` passes on x86_64 and Pi.
3. **Benchmarks** — `benchmark_fps.py` produces `profiling.json` with per-stage latencies.
4. **Safety** — planner timeout triggers safe retract to home + failure telemetry export.

## Running on Raspberry Pi

```bash
# 1. Clone/copy this directory to Pi
# 2. Provision (once, with internet)
sudo ./install_deps.sh

# 3. Run as daemon
sudo cp systemd/edge-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now edge-pipeline

# 4. View logs
journalctl -u edge-pipeline -f
```

## MATLAB Ingestion

```matlab
% Read trajectory CSV
data = readmatrix('trajectory.csv', 'CommentStyle', '%');
t = data(:, 1);           % time (s)
q = data(:, 2:end);       % joint angles (rad)
plot(t, q); legend("q1","q2","q3");
```

## License

Internal project — not for public distribution.
