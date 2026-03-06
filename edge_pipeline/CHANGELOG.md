# Changelog

## v0.1.0 — Initial Implementation (2026-03-04)

### Added

- **`src/camera/`** — `DepthSource` protocol, `Detection` / `DepthFrame` data classes, `MockCamera` for testing
- **`src/transform/`** — `pixel_depth_to_camera()`, `camera_to_base()`, `batch_pixel_to_base()`, `TransformWorkspace`
- **`src/kinematics/`** — `ArmKinematics` class with:
  - Dual backend: `roboticstoolbox-python` (primary) + `ikpy` (fallback)
  - Gradient-descent IK with null-space joint-limit avoidance
  - Numerical Jacobian computation
  - Configurable 3-DOF and 4-DOF via DH parameters
- **`src/collision/`** — `CollisionChecker` with capsule–sphere model, vectorised distance computation
- **`src/planning/`** — `Planner` facade with:
  - APF (Artificial Potential Field) planner
  - Bounded RRT* with adaptive rewire radius
  - Cubic / quintic spline trajectory smoothing
  - Trapezoidal velocity time-parameterisation
- **`src/control/`** — `ActuatorInterface` ABC with rate limiting, dead-man switch, soft joint limits; `PCA9685Driver` (I2C); `MockActuator`
- **`src/export/`** — CSV + JSON trajectory writers; SCP push utility
- **`src/service/`** — `EdgePipelineService` headless daemon with producer/consumer threading, safe retract, structured JSON logging
- **Tests** — `test_transform`, `test_kinematics`, `test_planner`, `test_collision`, `test_end_to_end` (pytest)
- **Examples** — `mock_end_to_end.py` (full pipeline demo), `benchmark_fps.py` (profiling)
- **Tools** — `convert_yolo.py` (YOLOv8 → TFLite/ONNX)
- **Deployment** — `install_deps.sh`, `setup_venv.sh`, `systemd/edge-pipeline.service`, `ci/ci.yml`
- **Docs** — `README.md`, `DESIGN_DECISIONS.md`, `PERFORMANCE_REPORT.md`, `SUMMARY.md`
