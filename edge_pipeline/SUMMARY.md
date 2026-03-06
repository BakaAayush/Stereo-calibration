# Summary

## What Was Implemented

The complete **edge compute portion** of the Vision-Guided Robotic Arm Kinematic Pipeline, targeting Raspberry Pi 5 (8 GB). Depth sensing (stereo cameras + SGBM disparity) is handled by the existing `Depth sensing/` project; this pipeline consumes its output.

### Modules Delivered

| Module | Files | Description |
|--------|-------|-------------|
| Camera | 2 | `DepthSource` protocol + `MockCamera` |
| Transform | 1 | Pixel→camera→base coordinate transforms (vectorised NumPy) |
| Kinematics | 1 | `ArmKinematics` class: FK, IK (dual backend), Jacobian, null-space projection |
| Collision | 1 | Capsule–sphere collision checker (vectorised) |
| Planning | 2 | APF planner, bounded RRT*, cubic/quintic smoothing, time parameterisation |
| Control | 3 | `ActuatorInterface` ABC, `PCA9685Driver` (I2C), `MockActuator` |
| Export | 3 | CSV writer (MATLAB-compatible), JSON writer, SCP push |
| Service | 1 | Headless daemon with producer/consumer, safe retract, JSON telemetry |

### Tests: 5 files, 30+ test cases
### Examples: 2 scripts (demo + benchmarks)
### Tools: 1 YOLO conversion utility
### Deployment: systemd unit, GitHub Actions CI, install scripts

---

## What Remains

| Item | Status | Notes |
|------|--------|-------|
| Real hardware testing (PCA9685 + servos) | **Not tested** | Requires Pi + I2C hardware |
| `roboticstoolbox` on ARM64 | **May not install** | ikpy fallback always works |
| Hailo / Coral USB accelerator backends | **Stub only** | Need hardware + SDK |
| Dynamic obstacle tracking | **Not implemented** | Out of scope per PRD |
| Encoder feedback (closed-loop control) | **Not implemented** | Hobby servos are open-loop |

---

## How to Run Acceptance Tests on Pi

```bash
# 1. Copy edge_pipeline/ to Pi (SCP or USB)
scp -r edge_pipeline/ pi@raspberrypi:~/

# 2. Provision (once, internet required)
cd ~/edge_pipeline
sudo ./install_deps.sh
source .venv/bin/activate

# 3. Run all tests
python -m pytest tests/ -v --tb=short

# 4. Run mock demo
python examples/mock_end_to_end.py
# → produces output/trajectory.csv

# 5. Run benchmarks
python examples/benchmark_fps.py
# → produces output/profiling.json

# 6. Verify MATLAB compatibility
python -c "import numpy as np; d = np.loadtxt('output/trajectory.csv', delimiter=',', comments='%'); print(f'{d.shape[0]} points × {d.shape[1]} cols')"
```

---

## Hardware-Specific Notes for Calibration

1. **DH Parameters:** Edit `src/kinematics/arm_kinematics.py` — update `DEFAULT_DH_3DOF` or `DEFAULT_DH_4DOF` with your arm's measured link lengths (a, d in metres) and twist angles (alpha in radians).

2. **Servo Calibration:** Edit `ServoCalibration` entries in `src/control/pca9685_driver.py`:
   - Measure the pulse width (µs) that corresponds to 0° and 180° for each servo.
   - Map these to the joint angle range in radians.

3. **Camera Extrinsics:** Measure the 4×4 transform `T_base_camera` from camera frame to arm base frame. Update in `examples/mock_end_to_end.py` or your deployment config.

4. **Camera Intrinsics:** Use the calibration data from your `Depth sensing/stereo_calibration.npz` file. Load with:
   ```python
   data = np.load("stereo_calibration.npz")
   K = data["K1"]  # left camera intrinsics
   ```
