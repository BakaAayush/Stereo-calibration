# =============================================================================
# daemon.py — Headless edge pipeline service (main loop)
# =============================================================================
# Purpose:  Orchestrate the full detection → IK → planning → control pipeline.
#           Runs as a headless daemon (no GUI) consuming depth frames and
#           producing either actuator commands (live mode) or CSV trajectories
#           (simulation mode).
#
# Architecture:
#   - Producer thread: reads from DepthSource, pushes DepthFrames to a queue.
#   - Consumer (main): dequeues frames, runs transform → IK → plan → smooth.
#   - In live mode: sends angles to ActuatorInterface.
#   - In simulation mode: writes trajectory CSV.
#
# Logging: Structured JSON lines (timestamp, module, level, metrics).
#
# Safety:
#   - Planner failure → safe stop → retract to home → export failure telemetry.
#   - Dead-man timeout on actuator if no new commands.
# =============================================================================
from __future__ import annotations

import json
import logging
import queue
import signal
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..camera.camera_interface import DepthFrame, DepthSource, Detection
from ..collision.checker import CollisionChecker, Obstacle
from ..control.actuator_interface import ActuatorInterface
from ..export.csv_export import write_trajectory_csv
from ..kinematics.arm_kinematics import ArmKinematics
from ..planning.planner import Planner, PlanningFailure
from ..planning.trajectory import smooth_trajectory, time_parameterize
from ..transform.transforms import pixel_depth_to_camera, camera_to_base

logger = logging.getLogger(__name__)


class ServiceMode(Enum):
    LIVE = "live"
    SIMULATION = "simulation"


class _JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": self.formatTime(record),
            "module": record.module,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "metrics"):
            entry["metrics"] = record.metrics
        if hasattr(record, "joint_angles"):
            entry["joint_angles"] = record.joint_angles
        return json.dumps(entry)


class EdgePipelineService:
    """Headless edge pipeline daemon.

    Parameters
    ----------
    depth_source : DepthSource
        Camera / depth frame provider.
    kinematics : ArmKinematics
        FK / IK engine.
    planner : Planner
        Path planner.
    actuator : ActuatorInterface
        Hardware driver (mock for simulation).
    collision_checker : CollisionChecker
        Collision detection.
    K : (3, 3) array
        Camera intrinsic matrix.
    T_base_camera : (4, 4) array
        Camera-to-base extrinsic.
    obstacles : list[Obstacle]
        Static workspace obstacles.
    mode : ServiceMode
        ``LIVE`` or ``SIMULATION``.
    output_dir : str or Path
        Directory for CSV / telemetry output.
    """

    def __init__(
        self,
        depth_source: DepthSource,
        kinematics: ArmKinematics,
        planner: Planner,
        actuator: ActuatorInterface,
        collision_checker: CollisionChecker,
        K: NDArray,
        T_base_camera: NDArray,
        obstacles: Sequence[Obstacle] | None = None,
        mode: ServiceMode = ServiceMode.SIMULATION,
        output_dir: str | Path = "output",
    ) -> None:
        self.depth_source = depth_source
        self.kin = kinematics
        self.planner = planner
        self.actuator = actuator
        self.cc = collision_checker
        self.K = np.asarray(K, dtype=np.float64)
        self.T = np.asarray(T_base_camera, dtype=np.float64)
        self.obstacles = list(obstacles) if obstacles else []
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._frame_queue: queue.Queue[DepthFrame] = queue.Queue(maxsize=5)
        self._running = False
        self._traj_count = 0

        # Current joint state
        self._current_q = kinematics.home_position.copy()

    # ------------------------------------------------------------------ #
    # Logging setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def setup_logging(log_file: str | Path | None = None) -> None:
        """Configure structured JSON logging."""
        root = logging.getLogger()
        root.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JSONFormatter())
        root.addHandler(handler)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(_JSONFormatter())
            root.addHandler(fh)

    # ------------------------------------------------------------------ #
    # Producer thread (depth frame acquisition)
    # ------------------------------------------------------------------ #
    def _producer(self) -> None:
        """Continuously fetch frames from the depth source."""
        while self._running:
            try:
                frame = self.depth_source.get_frame()
                if frame is not None:
                    try:
                        self._frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        pass  # drop frame if consumer is slow
            except Exception as e:
                logger.error("Producer error: %s", e)
                time.sleep(0.1)

    # ------------------------------------------------------------------ #
    # Main processing loop
    # ------------------------------------------------------------------ #
    def run(self, max_iterations: int | None = None) -> None:
        """Start the service.  Blocks until shutdown signal or max iterations."""
        self._running = True
        self.setup_logging(self.output_dir / "telemetry.jsonl")

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start producer thread
        producer_thread = threading.Thread(target=self._producer, daemon=True)
        producer_thread.start()

        if self.mode == ServiceMode.LIVE:
            self.actuator.enable()

        logger.info("EdgePipelineService started in %s mode", self.mode.value)

        iteration = 0
        while self._running:
            if max_iterations is not None and iteration >= max_iterations:
                break

            try:
                frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            self._process_frame(frame)
            iteration += 1

        # Shutdown
        self._shutdown()

    def _process_frame(self, frame: DepthFrame) -> None:
        """Process a single depth frame through the full pipeline."""
        t_start = time.perf_counter()
        metrics: dict = {}

        for det in frame.detections:
            t0 = time.perf_counter()

            # 1. Coordinate transform: pixel + depth → base frame
            point_cam = pixel_depth_to_camera(det.u, det.v, det.depth_m, self.K)
            point_base = camera_to_base(point_cam, self.T)
            t_transform = time.perf_counter()
            metrics["transform_ms"] = (t_transform - t0) * 1000

            # 2. Inverse kinematics
            q_goal, ik_info = self.kin.ik(point_base, q0=self._current_q)
            t_ik = time.perf_counter()
            metrics["ik_ms"] = (t_ik - t_transform) * 1000

            if q_goal is None:
                logger.warning(
                    "IK failed for target %s (error: %s)",
                    np.round(point_base, 4),
                    ik_info.get("error_m"),
                )
                continue

            # 3. Path planning
            try:
                path = self.planner.plan(self._current_q, q_goal, self.obstacles)
                t_plan = time.perf_counter()
                metrics["plan_ms"] = (t_plan - t_ik) * 1000
            except PlanningFailure as e:
                logger.warning("Planning failed: %s — executing safe retract", e)
                self._safe_retract()
                continue

            # 4. Trajectory smoothing + time parameterisation
            traj = smooth_trajectory(path, dt=0.02)
            traj, timestamps = time_parameterize(traj)
            t_smooth = time.perf_counter()
            metrics["smooth_ms"] = (t_smooth - t_plan) * 1000

            # 5. Execute or export
            if self.mode == ServiceMode.LIVE:
                self._execute_trajectory(traj, timestamps)
            else:
                self._export_trajectory(traj, timestamps, det)

            self._current_q = traj[-1].copy()

            metrics["total_ms"] = (time.perf_counter() - t0) * 1000
            logger.info(
                "Processed detection '%s'",
                det.label,
                extra={
                    "metrics": metrics,
                    "joint_angles": np.round(self._current_q, 4).tolist(),
                },
            )

    def _execute_trajectory(self, traj: NDArray, timestamps: NDArray) -> None:
        """Send trajectory to actuator in real-time."""
        t0 = time.monotonic()
        for i, angles in enumerate(traj):
            target_time = t0 + timestamps[i]
            self.actuator.set_angles(angles)
            # Wait until the scheduled time
            now = time.monotonic()
            if now < target_time:
                time.sleep(target_time - now)

    def _export_trajectory(self, traj: NDArray, timestamps: NDArray, det: Detection) -> None:
        """Write trajectory to CSV file."""
        self._traj_count += 1
        filename = self.output_dir / f"trajectory_{self._traj_count:04d}_{det.label}.csv"
        write_trajectory_csv(filename, traj, timestamps)

    def _safe_retract(self) -> None:
        """Emergency retract to home position."""
        logger.warning("Executing safe retract to home position")
        home = self.kin.home_position

        # Generate a simple linear retract trajectory
        path = [self._current_q.copy(), home.copy()]
        traj = smooth_trajectory(path, dt=0.05)
        traj, timestamps = time_parameterize(traj, max_vel=0.5, max_acc=1.0)

        # Export failure telemetry
        filename = self.output_dir / f"retract_{self._traj_count:04d}_failure.csv"
        write_trajectory_csv(filename, traj, timestamps)

        if self.mode == ServiceMode.LIVE:
            self._execute_trajectory(traj, timestamps)

        self._current_q = home.copy()

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        self._running = False
        self.actuator.disable()
        self.depth_source.release()
        self.actuator.close()
        logger.info("EdgePipelineService shut down")

    def _signal_handler(self, signum: int, frame) -> None:
        logger.info("Signal %d received — shutting down", signum)
        self._running = False
