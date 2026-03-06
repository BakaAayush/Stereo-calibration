# =============================================================================
# planner.py — Unified path planner with APF and bounded RRT* backends
# =============================================================================
# Purpose:  Provide a single Planner facade that supports two modes:
#   1. APF (Artificial Potential Field) — fast for static known workspaces.
#   2. RRTstarBounded — sampling-based with iteration/time bounds.
#
# Both return a list of joint-space waypoints or raise PlanningFailure.
#
# Inputs:   start config, goal config, obstacles, planner parameters.
# Outputs:  List of (DOF,) waypoints or PlanningFailure.
#
# Performance: RRT* bounded to configurable max_iterations and max_time_s.
# =============================================================================
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..collision.checker import CollisionChecker, Obstacle

logger = logging.getLogger(__name__)


class PlannerMode(enum.Enum):
    APF = "apf"
    RRT_STAR = "rrt_star"


class PlanningFailure(Exception):
    """Raised when the planner cannot find a valid path within bounds."""
    pass


@dataclass
class PlannerConfig:
    """Planner configuration parameters.

    Attributes
    ----------
    mode : PlannerMode
        Planning algorithm to use.
    max_iterations : int
        Maximum iterations for RRT*.
    max_time_s : float
        Wall-clock timeout for planning (seconds).
    step_size : float
        RRT* extension step size in configuration space (radians).
    goal_bias : float
        Probability of sampling the goal directly [0, 1].
    rewire_radius : float
        RRT* rewire neighbourhood radius (radians).
    apf_step_size : float
        APF gradient descent step size.
    apf_max_steps : int
        APF maximum gradient steps.
    """

    mode: PlannerMode = PlannerMode.RRT_STAR
    max_iterations: int = 5000
    max_time_s: float = 2.0
    step_size: float = 0.15
    goal_bias: float = 0.10
    rewire_radius: float = 0.5
    apf_step_size: float = 0.05
    apf_max_steps: int = 500


class Planner:
    """Unified path planner.

    Parameters
    ----------
    dof : int
        Degrees of freedom.
    joint_limits : (DOF, 2) array
        Per-joint [min, max] in radians.
    collision_checker : CollisionChecker
        Collision checking instance.
    config : PlannerConfig | None
        Planner configuration.
    """

    def __init__(
        self,
        dof: int,
        joint_limits: NDArray,
        collision_checker: CollisionChecker,
        config: PlannerConfig | None = None,
    ) -> None:
        self.dof = dof
        self.limits = np.asarray(joint_limits, dtype=np.float64)
        self.cc = collision_checker
        self.cfg = config or PlannerConfig()
        self._rng = np.random.default_rng(42)

    def plan(
        self,
        start: NDArray,
        goal: NDArray,
        obstacles: Sequence[Obstacle],
    ) -> list[NDArray]:
        """Plan a collision-free path from ``start`` to ``goal``.

        Returns
        -------
        path : list[(DOF,) ndarray]
            Ordered list of waypoints in joint space (includes start and goal).

        Raises
        ------
        PlanningFailure
            If no path found within the configured bounds.
        """
        start = np.asarray(start, dtype=np.float64).ravel()
        goal = np.asarray(goal, dtype=np.float64).ravel()

        # Quick check: start and goal must be collision-free
        if self.cc.is_collision(start, obstacles):
            raise PlanningFailure("Start configuration is in collision")
        if self.cc.is_collision(goal, obstacles):
            raise PlanningFailure("Goal configuration is in collision")

        # Direct path check (skip planning if straight line is free)
        if self._direct_path_free(start, goal, obstacles):
            logger.info("Direct path is collision-free; skipping planner")
            return [start.copy(), goal.copy()]

        if self.cfg.mode == PlannerMode.APF:
            return self._plan_apf(start, goal, obstacles)
        else:
            return self._plan_rrt_star(start, goal, obstacles)

    # ================================================================== #
    # APF Planner
    # ================================================================== #
    def _plan_apf(
        self, start: NDArray, goal: NDArray, obstacles: Sequence[Obstacle],
    ) -> list[NDArray]:
        """Artificial Potential Field planner (fast, may get stuck in local minima)."""
        path = [start.copy()]
        q = start.copy()
        t0 = time.perf_counter()

        for step in range(self.cfg.apf_max_steps):
            if time.perf_counter() - t0 > self.cfg.max_time_s:
                raise PlanningFailure(f"APF timeout after {step} steps")

            # Attractive force toward goal
            diff = goal - q
            dist = np.linalg.norm(diff)
            if dist < self.cfg.step_size:
                path.append(goal.copy())
                logger.info("APF converged in %d steps", step)
                return path

            f_att = diff / dist  # unit vector toward goal

            # Repulsive force from obstacles (in config space, approximate)
            f_rep = np.zeros(self.dof, dtype=np.float64)
            d_min = self.cc.min_distance(q, obstacles)
            if d_min < 0.1:  # repulsion active within 10 cm
                # Numerical gradient of distance
                eps = 0.01
                for i in range(self.dof):
                    q_plus = q.copy()
                    q_plus[i] += eps
                    d_plus = self.cc.min_distance(q_plus, obstacles)
                    f_rep[i] = (d_plus - d_min) / eps
                f_rep_norm = np.linalg.norm(f_rep)
                if f_rep_norm > 1e-6:
                    f_rep = f_rep / f_rep_norm * (0.1 / max(d_min, 0.01))

            # Step
            f_total = f_att + f_rep
            f_total_norm = np.linalg.norm(f_total)
            if f_total_norm > 1e-6:
                q = q + self.cfg.apf_step_size * f_total / f_total_norm
                q = np.clip(q, self.limits[:, 0], self.limits[:, 1])

            if not self.cc.is_collision(q, obstacles):
                path.append(q.copy())
            else:
                # Try smaller step
                q = path[-1].copy()

        raise PlanningFailure(f"APF failed to converge in {self.cfg.apf_max_steps} steps")

    # ================================================================== #
    # RRT* Planner (bounded)
    # ================================================================== #
    def _plan_rrt_star(
        self, start: NDArray, goal: NDArray, obstacles: Sequence[Obstacle],
    ) -> list[NDArray]:
        """Bounded RRT* planner with iteration and time limits.

        Uses vectorised nearest-neighbour search and collision checking.
        Numba-compatible hot-loop structure for optional acceleration.
        """
        # Tree storage: nodes[i] = config, parent[i] = parent index, cost[i] = path cost
        max_nodes = self.cfg.max_iterations + 1
        nodes = np.empty((max_nodes, self.dof), dtype=np.float64)
        parent = np.full(max_nodes, -1, dtype=np.int32)
        cost = np.full(max_nodes, np.inf, dtype=np.float64)

        nodes[0] = start
        cost[0] = 0.0
        n_nodes = 1
        goal_idx = -1

        t0 = time.perf_counter()

        for iteration in range(self.cfg.max_iterations):
            if time.perf_counter() - t0 > self.cfg.max_time_s:
                logger.warning("RRT* timeout after %d iterations", iteration)
                break

            # Sample random configuration (with goal bias)
            if self._rng.random() < self.cfg.goal_bias:
                q_rand = goal.copy()
            else:
                q_rand = self._random_config()

            # Find nearest node (vectorised)
            diffs = nodes[:n_nodes] - q_rand
            dists = np.linalg.norm(diffs, axis=1)
            nearest_idx = int(np.argmin(dists))
            q_nearest = nodes[nearest_idx]

            # Steer toward q_rand
            direction = q_rand - q_nearest
            dist = dists[nearest_idx]
            if dist > self.cfg.step_size:
                q_new = q_nearest + self.cfg.step_size * direction / dist
            else:
                q_new = q_rand.copy()

            q_new = np.clip(q_new, self.limits[:, 0], self.limits[:, 1])

            # Collision check
            if self.cc.is_collision(q_new, obstacles):
                continue
            if self._edge_collision(q_nearest, q_new, obstacles):
                continue

            # RRT* rewiring: find near nodes
            new_cost = cost[nearest_idx] + np.linalg.norm(q_new - q_nearest)
            near_dists = np.linalg.norm(nodes[:n_nodes] - q_new, axis=1)
            near_mask = near_dists < self.rewire_radius_adaptive(n_nodes)
            near_indices = np.where(near_mask)[0]

            # Choose best parent
            best_parent = nearest_idx
            best_cost = new_cost

            for ni in near_indices:
                candidate_cost = cost[ni] + near_dists[ni]
                if candidate_cost < best_cost:
                    if not self._edge_collision(nodes[ni], q_new, obstacles):
                        best_parent = ni
                        best_cost = candidate_cost

            # Add node
            nodes[n_nodes] = q_new
            parent[n_nodes] = best_parent
            cost[n_nodes] = best_cost

            # Rewire near nodes
            for ni in near_indices:
                rewire_cost = best_cost + near_dists[ni]
                if rewire_cost < cost[ni]:
                    if not self._edge_collision(q_new, nodes[ni], obstacles):
                        parent[ni] = n_nodes
                        cost[ni] = rewire_cost

            # Check if we reached the goal
            if np.linalg.norm(q_new - goal) < self.cfg.step_size:
                if not self._edge_collision(q_new, goal, obstacles):
                    # Add goal node
                    n_nodes += 1
                    nodes[n_nodes] = goal
                    parent[n_nodes] = n_nodes - 1
                    cost[n_nodes] = best_cost + np.linalg.norm(q_new - goal)
                    goal_idx = n_nodes
                    n_nodes += 1
                    logger.info("RRT* found path in %d iterations", iteration)
                    break

            n_nodes += 1

        if goal_idx < 0:
            raise PlanningFailure(
                f"RRT* failed: {n_nodes} nodes explored in "
                f"{time.perf_counter() - t0:.2f}s"
            )

        # Extract path
        path = []
        idx = goal_idx
        while idx >= 0:
            path.append(nodes[idx].copy())
            idx = parent[idx]
        path.reverse()
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _random_config(self) -> NDArray:
        """Sample a random configuration within joint limits."""
        return self._rng.uniform(self.limits[:, 0], self.limits[:, 1])

    def _direct_path_free(
        self, a: NDArray, b: NDArray, obstacles: Sequence[Obstacle], n_checks: int = 20,
    ) -> bool:
        """Check if the straight-line path between a and b is collision-free."""
        configs = np.linspace(a, b, n_checks)
        return not self.cc.is_path_collision(configs, obstacles)

    def _edge_collision(
        self, a: NDArray, b: NDArray, obstacles: Sequence[Obstacle], n_checks: int = 10,
    ) -> bool:
        """Check a single edge for collisions (intermediate samples)."""
        configs = np.linspace(a, b, n_checks)
        return self.cc.is_path_collision(configs, obstacles)

    def rewire_radius_adaptive(self, n: int) -> float:
        """Adaptive RRT* rewire radius: shrinks as tree grows."""
        if n < 2:
            return self.cfg.rewire_radius
        d = self.dof
        gamma = self.cfg.rewire_radius
        return min(gamma * (np.log(n) / n) ** (1.0 / d), self.cfg.rewire_radius)
