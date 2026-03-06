# Performance Report

Profiling results and tuning guide for the edge pipeline on Raspberry Pi 5 (8 GB).

## Benchmark Methodology

- **Script:** `examples/benchmark_fps.py`
- **Iterations:** 200 per stage (50 for planner)
- **Hardware:** Raspberry Pi 5 (8 GB), active cooling, 27 W PD
- **Software:** Python 3.11, NumPy with OpenBLAS, ikpy backend
- **Command:** `python examples/benchmark_fps.py --iterations 200 --dof 3`

## Expected Results (Pi 5, 3-DOF)

| Stage | Mean (ms) | Median (ms) | P99 (ms) | Target |
|-------|-----------|-------------|----------|--------|
| Transform (single) | < 0.1 | < 0.1 | < 0.5 | < 3 ms ✅ |
| Transform (batch 100) | < 0.5 | < 0.5 | < 1.0 | < 3 ms ✅ |
| IK (ikpy) | 3–8 | 5 | 15 | < 10 ms ✅ |
| IK (rtb) | 2–5 | 3 | 10 | < 10 ms ✅ |
| Planner (RRT*, no obs) | 0.5–50 | 5 | 200 | < 2000 ms ✅ |
| Smoothing | 0.5–2 | 1 | 3 | — |

### Overall Pipeline FPS

```
Transform + IK only:  ~100–200 FPS (x86), ~50–100 FPS (Pi 5)
Full pipeline (incl. planning):  10–30 FPS depending on planner mode
```

**Target: 10–15 FPS overall** — achieved with APF planning or when direct path is available.

## Profiling Output Format

`benchmark_fps.py` produces `output/profiling.json`:

```json
{
  "benchmark_config": {
    "dof": 3,
    "n_iterations": 200,
    "platform": "linux"
  },
  "results": [
    {
      "stage": "transform_single",
      "mean_ms": 0.05,
      "median_ms": 0.04,
      "p99_ms": 0.12,
      ...
    }
  ]
}
```

## Tuning Guide

### Planning Speed

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_iterations` | 5000 | ↑ = better paths, ↑ time |
| `max_time_s` | 2.0 | Hard timeout; safety bound |
| `step_size` | 0.15 rad | ↑ = faster exploration, ↓ precision |
| `goal_bias` | 0.10 | ↑ = faster convergence, ↓ exploration |

**Recommendation:** For real-time on Pi, use `max_iterations=2000, max_time_s=1.0`.

### IK Convergence

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ik_timeout_s` | 0.05 | Max time for gradient-descent fallback |
| `ik_tol` | 1e-3 m | Position tolerance |

### Memory Optimisation

- Use `TransformWorkspace` for hot-loop transforms (avoids repeated allocation).
- Pre-allocate Jacobian workspace (done automatically in `ArmKinematics`).
- Set `OPENBLAS_NUM_THREADS=4` in environment to use all Pi 5 cores.

### Thermal Management

- Enable active cooling (fan heatsink required for sustained operation).
- Monitor with: `vcgencmd measure_temp`
- If throttling occurs (>80°C), reduce `max_iterations` or add cooling.

## Running Benchmarks

```bash
# Quick benchmark
python examples/benchmark_fps.py

# Extended with 4-DOF
python examples/benchmark_fps.py --iterations 500 --dof 4 --output profiling_4dof.json
```
