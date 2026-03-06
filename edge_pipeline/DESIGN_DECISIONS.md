# Design Decisions

Trade-off rationale for key architectural choices in the edge pipeline.

---

## 1. `roboticstoolbox` + `ikpy` Dual Backend

**Decision:** Use `roboticstoolbox-python` as primary kinematics backend, with `ikpy` as automatic fallback.

**Rationale:**
- `roboticstoolbox` is the most complete Python robotics library (Jacobians, dynamics, multiple IK solvers) but has spotty ARM64 wheel availability.
- `ikpy` is pure Python, always installs, and handles basic FK/IK adequately for 3–4 DOF arms.
- Runtime detection (`try/except ImportError`) ensures the code works on both x86 (CI) and ARM64 (Pi) without configuration.

**Trade-off:** ikpy's IK is less precise; mitigated by the gradient-descent fallback solver.

---

## 2. APF vs RRT* Planning

**Decision:** Offer both APF (fast, deterministic) and bounded RRT* (more capable).

| | APF | RRT* |
|---|---|---|
| **Speed** | ~1 ms | 10–2000 ms |
| **Optimality** | Local minimum risk | Asymptotically optimal |
| **Dynamic obstacles** | Needs recompute | Handles well |

**Recommendation:** Use APF for static, well-characterised workspaces; RRT* when obstacles may change or when higher path quality is needed.

---

## 3. Bounded Planning (Time + Iterations)

**Decision:** All planners have hard `max_iterations` and `max_time_s` limits.

**Rationale:** On a resource-constrained Pi 5, unbounded planning can block the control loop. Safety requires predictable worst-case latency. On failure, the system executes a safe retract to home.

---

## 4. Capsule–Sphere Collision Model

**Decision:** Approximate links as capsules and obstacles as spheres.

**Rationale:**
- Capsule–sphere intersection is O(1) per pair and vectorises trivially with NumPy.
- Sufficient accuracy for 3–4 DOF arms with cylindrical links.
- Full mesh collision (e.g., FCL/Bullet) would be overkill and too slow for Pi.

---

## 5. INT8 Quantisation for NN Inference

**Decision:** Require TFLite or ONNX with INT8 quantisation for any neural network inference on Pi.

**Rationale:**
- Full PyTorch CPU on ARM64 uses ~2 GB RAM and is 5–10× slower than quantised TFLite.
- INT8 YOLOv8n on Pi 5 CPU achieves ~5–8 FPS; FP32 PyTorch achieves < 1 FPS.
- `tools/convert_yolo.py` automates conversion.

---

## 6. Producer/Consumer Threading

**Decision:** Camera/depth producer on a separate thread; IK/planning on the main thread.

**Rationale:**
- Camera capture is I/O-bound (USB/CSI); IK/planning is CPU-bound.
- Thread-safe queue decouples them; frame drops are acceptable if consumer is slow.
- `multiprocessing` is available as an escalation path if GIL becomes a bottleneck.

---

## 7. PCA9685 Open-Loop Control

**Decision:** No encoder feedback; rely on open-loop PWM commands.

**Rationale:**
- Hobby servos (SG90, MG996R) have built-in position controllers.
- Adding encoders requires additional hardware and ADC.
- Calibration table maps angles ↔ PWM ticks for each servo.

**ASSUMPTION:** Servo accuracy is sufficient for the application (~2° error typical for SG90).

---

## 8. CSV Export with MATLAB % Comments

**Decision:** Use `%` as comment character in CSV files.

**Rationale:** MATLAB's `readmatrix()` natively ignores lines beginning with `%`, making the CSV directly ingestible without preprocessing.

---

## 9. No GUI / Headless Only

**Decision:** No desktop GUI or `cv2.imshow` in production code.

**Rationale:** Pi runs headless in production. All output is via structured JSON logs, CSV files, and systemd journal. Debug visualisation (if needed) can be done by SSH-forwarding exported data.

---

## Assumptions

| # | Assumption | Justification |
|---|-----------|---------------|
| 1 | Default DH params are generic educational arm (link lengths 77–130 mm) | No specific arm model provided; users must calibrate |
| 2 | Camera intrinsics: fx=fy=600, principal point at (640, 360) | Typical 720p parameters; calibration data from Depth Sensing pipeline overrides |
| 3 | SG90-type servo range: 500–2500 µs at 50 Hz | Standard hobby servo specs |
| 4 | Camera mounted ~0.3 m above arm base, pointing down | Common eye-in-hand / fixed camera setup |
| 5 | Obstacles are static spheres provided at startup | Dynamic obstacle tracking is out of scope |
| 6 | OpenBLAS is available as system BLAS on Pi OS Bookworm | `apt install libopenblas-dev` provides it |
