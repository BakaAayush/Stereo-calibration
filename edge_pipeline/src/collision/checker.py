# =============================================================================
# checker.py — Vectorized collision detection (capsule/sphere)
# =============================================================================
# Purpose:  Check whether a robot configuration collides with a set of static
#           obstacles.  Each link is modelled as a capsule (cylinder with
#           hemispherical end-caps) and obstacles as spheres.  All heavy math
#           is NumPy-vectorised (no Python loops over obstacles).
#
# Inputs:   Joint configuration, forward kinematics callable, obstacle list.
# Outputs:  Boolean collision flag + per-link minimum distances.
#
# Complexity: O(L * O) where L = number of links, O = number of obstacles.
#             Fully vectorised over O for each link.
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Obstacle:
    """Sphere obstacle in the base frame.

    Attributes
    ----------
    center : tuple[float, float, float]
        (x, y, z) in metres, robot base frame.
    radius : float
        Sphere radius in metres.
    """

    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True, slots=True)
class LinkGeometry:
    """Capsule approximation for one robot link.

    Defined by its two endpoints in the base frame (computed from FK)
    and a radius that inflates the collision boundary.
    """

    p0: NDArray  # (3,) start point
    p1: NDArray  # (3,) end point
    radius: float  # capsule radius (m)


class CollisionChecker:
    """Collision checker using capsule-link / sphere-obstacle model.

    Parameters
    ----------
    link_radii : list[float]
        Collision radius for each link capsule (metres).
    fk_frames : callable
        ``fk_frames(q) -> list[(4,4) ndarray]``  — returns the homogeneous
        transform of each joint frame (including base and tool).  Length
        should be ``DOF + 1`` (base → joint1 → … → tool).
    safety_margin : float
        Extra clearance added to all distance checks (metres).
    """

    def __init__(
        self,
        link_radii: Sequence[float],
        fk_frames: Callable[[NDArray], list[NDArray]],
        safety_margin: float = 0.01,
    ) -> None:
        self.link_radii = list(link_radii)
        self.fk_frames = fk_frames
        self.margin = safety_margin

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def is_collision(
        self,
        q: NDArray,
        obstacles: Sequence[Obstacle],
    ) -> bool:
        """Return ``True`` if configuration ``q`` collides with any obstacle."""
        if not obstacles:
            return False
        links = self._config_to_links(q)
        obs_arr, obs_r = self._obstacles_to_arrays(obstacles)
        for link in links:
            if self._capsule_spheres_collide(link, obs_arr, obs_r):
                return True
        return False

    def is_path_collision(
        self,
        configs: NDArray,
        obstacles: Sequence[Obstacle],
    ) -> bool:
        """Check a batch of configurations (N×DOF) for any collision."""
        for q in configs:
            if self.is_collision(q, obstacles):
                return True
        return False

    def min_distance(
        self,
        q: NDArray,
        obstacles: Sequence[Obstacle],
    ) -> float:
        """Return the minimum distance between any link and any obstacle."""
        if not obstacles:
            return float("inf")
        links = self._config_to_links(q)
        obs_arr, obs_r = self._obstacles_to_arrays(obstacles)
        d_min = float("inf")
        for link in links:
            d = self._capsule_spheres_distance(link, obs_arr, obs_r)
            d_min = min(d_min, d)
        return d_min

    # ------------------------------------------------------------------ #
    # Internal: geometry construction
    # ------------------------------------------------------------------ #
    def _config_to_links(self, q: NDArray) -> list[LinkGeometry]:
        """Compute capsule representations for all links at config ``q``."""
        frames = self.fk_frames(q)
        links = []
        for i in range(len(frames) - 1):
            p0 = frames[i][:3, 3]
            p1 = frames[i + 1][:3, 3]
            r = self.link_radii[i] if i < len(self.link_radii) else 0.02
            links.append(LinkGeometry(p0=p0, p1=p1, radius=r))
        return links

    @staticmethod
    def _obstacles_to_arrays(
        obstacles: Sequence[Obstacle],
    ) -> tuple[NDArray, NDArray]:
        """Stack obstacles into vectorised arrays."""
        centers = np.array([o.center for o in obstacles], dtype=np.float64)  # (M, 3)
        radii = np.array([o.radius for o in obstacles], dtype=np.float64)  # (M,)
        return centers, radii

    # ------------------------------------------------------------------ #
    # Vectorised capsule–sphere distance
    # ------------------------------------------------------------------ #
    @staticmethod
    def _point_segment_dist_batch(
        p: NDArray,     # (M, 3) — obstacle centre points
        a: NDArray,     # (3,) — segment start
        b: NDArray,     # (3,) — segment end
    ) -> NDArray:
        """Vectorised min distance from M points to segment a-b.

        Returns (M,) array of distances.
        """
        ab = b - a  # (3,)
        ab_sq = np.dot(ab, ab)

        if ab_sq < 1e-12:
            # Degenerate segment (zero-length link)
            return np.linalg.norm(p - a, axis=1)

        # Parameterise: t = clamp(dot(ap, ab) / |ab|^2, 0, 1)
        ap = p - a  # (M, 3)
        t = np.clip(np.dot(ap, ab) / ab_sq, 0.0, 1.0)  # (M,)

        # Nearest point on segment for each obstacle
        nearest = a + np.outer(t, ab)  # (M, 3)
        return np.linalg.norm(p - nearest, axis=1)  # (M,)

    def _capsule_spheres_collide(
        self,
        link: LinkGeometry,
        obs_centers: NDArray,
        obs_radii: NDArray,
    ) -> bool:
        """Check if a capsule collides with any sphere (vectorised)."""
        dists = self._point_segment_dist_batch(obs_centers, link.p0, link.p1)
        thresholds = link.radius + obs_radii + self.margin
        return bool(np.any(dists < thresholds))

    def _capsule_spheres_distance(
        self,
        link: LinkGeometry,
        obs_centers: NDArray,
        obs_radii: NDArray,
    ) -> float:
        """Minimum clearance between capsule and spheres."""
        dists = self._point_segment_dist_batch(obs_centers, link.p0, link.p1)
        clearances = dists - link.radius - obs_radii - self.margin
        return float(np.min(clearances))
