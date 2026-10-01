"""Spatial indexing (neighbour search) and collision detection.

Neighbour queries use a KD-tree (``scipy.spatial.cKDTree``), rebuilt every
tick: O(N log N) build and O(log N + k) per query, instead of the O(N^2)
all-pairs check. A vectorised brute-force backend with identical semantics
is kept as a reference implementation (used by the tests), and as a fallback
when SciPy is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .events import EventBus, EventCategory, Severity
from .types import COLLISION_SEVERITY, CollisionState

log = logging.getLogger(__name__)

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - exercised only without SciPy
    cKDTree = None
    log.warning("SciPy not available - falling back to O(N^2) neighbour search")


class SpatialIndex:
    """Radius / k-nearest neighbour queries over a set of 3D points."""

    def __init__(self, use_kdtree: bool = True) -> None:
        self.use_kdtree = use_kdtree and cKDTree is not None
        self._points = np.zeros((0, 3))
        self._tree = None

    def rebuild(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=np.float64)
        self._points = pts.reshape(-1, pts.shape[-1] if pts.ndim == 2 and pts.shape[-1] in (2, 3) else 3)
        self._tree = cKDTree(self._points) if self.use_kdtree and len(self._points) else None

    def __len__(self) -> int:
        return len(self._points)

    def query_neighbors(self, radius: float, max_neighbors: int) -> list[np.ndarray]:
        """For every point, the indices of up to ``max_neighbors`` *other* points
        within ``radius``, sorted by increasing distance."""
        n = len(self._points)
        if n == 0:
            return []
        k = min(max_neighbors + 1, n)
        if self._tree is not None:
            dist, idx = self._tree.query(self._points, k=k, distance_upper_bound=radius)
            dist = dist.reshape(n, k)
            idx = idx.reshape(n, k)
        else:
            diff = self._points[:, None, :] - self._points[None, :, :]
            d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
            idx = np.argsort(d, axis=1, kind="stable")[:, :k]
            dist = np.take_along_axis(d, idx, axis=1)
            dist = np.where(dist <= radius, dist, np.inf)
        result: list[np.ndarray] = []
        own = np.arange(n)[:, None]
        # Exclude self explicitly rather than dropping column 0: coincident points
        # may be returned in any order.
        valid = np.isfinite(dist) & (idx != own)
        for i in range(n):
            result.append(idx[i][valid[i]][:max_neighbors])
        return result

    def query_ball(self, points: np.ndarray, radius: float) -> list[list[int]]:
        """For each query point (external to the index), the indices of indexed points within ``radius``."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, self._points.shape[1] if len(self._points) else 3)
        if len(self._points) == 0:
            return [[] for _ in range(len(pts))]
        if self._tree is not None:
            return [list(r) for r in self._tree.query_ball_point(pts, radius)]
        diff = pts[:, None, :] - self._points[None, :, :]
        d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        return [list(np.flatnonzero(row <= radius)) for row in d]

    def query_pairs(self, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """All unordered index pairs (i < j) closer than ``radius`` and their distances."""
        n = len(self._points)
        if n < 2:
            return np.zeros((0, 2), dtype=np.intp), np.zeros(0)
        if self._tree is not None:
            pairs = self._tree.query_pairs(radius, output_type="ndarray")
        else:
            i, j = np.triu_indices(n, k=1)
            d = np.linalg.norm(self._points[i] - self._points[j], axis=1)
            mask = d <= radius
            pairs = np.stack((i[mask], j[mask]), axis=1)
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.intp), np.zeros(0)
        pairs = np.sort(pairs, axis=1)
        dists = np.linalg.norm(self._points[pairs[:, 0]] - self._points[pairs[:, 1]], axis=1)
        return pairs, dists


def nearest_neighbor_distances(points: np.ndarray) -> np.ndarray:
    """Distance from every point to its nearest other point (inf when there is only one point)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(pts)
    if n < 2:
        return np.full(n, np.inf)
    if cKDTree is not None:
        d, _ = cKDTree(pts).query(pts, k=2)
        return d[:, 1]
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


@dataclass(slots=True)
class CollisionReport:
    states: list[CollisionState]
    nearest: np.ndarray            # per point, distance to the nearest conflicting point (inf if none)
    min_separation: float          # minimum pairwise distance among active points within warning range
    warning_pairs: int
    avoidance_pairs: int
    collision_pairs: int


class CollisionMonitor:
    """Classifies pairwise separation and records escalations as events.

    ``d < collision -> COLLISION``, ``d < separation -> AVOIDANCE``,
    ``d < warning -> WARNING``. Only active (airborne) vehicles take part.
    """

    def __init__(self, separation: float, warning: float, collision: float, events: EventBus | None = None,
                 min_separation: float | None = None) -> None:
        self.separation = separation
        self.warning = warning
        self.collision = collision
        # Hard floor (swarm.min_separation): every pair that drops below it is counted and logged.
        self.min_separation = separation if min_separation is None else min_separation
        self._events = events
        self._pair_levels: dict[tuple[int, int], CollisionState] = {}
        self._hard_pairs: set[tuple[int, int]] = set()
        self.total_violations = 0
        self.total_collisions = 0
        self.total_hard_violations = 0
        self.lowest_separation = float("inf")   # smallest pairwise distance ever observed between airborne drones

    def classify(self, distance: float) -> CollisionState:
        if distance < self.collision:
            return CollisionState.COLLISION
        if distance < self.separation:
            return CollisionState.AVOIDANCE
        if distance < self.warning:
            return CollisionState.WARNING
        return CollisionState.CLEAR

    def update(self, ids: list[int], index: SpatialIndex, active: np.ndarray, time: float) -> CollisionReport:
        n = len(ids)
        states = [CollisionState.CLEAR] * n
        nearest = np.full(n, np.inf)
        pairs, dists = index.query_pairs(self.warning)
        if len(pairs):
            keep = active[pairs[:, 0]] & active[pairs[:, 1]]
            pairs, dists = pairs[keep], dists[keep]
        counts = {CollisionState.WARNING: 0, CollisionState.AVOIDANCE: 0, CollisionState.COLLISION: 0}
        current: dict[tuple[int, int], CollisionState] = {}
        hard: set[tuple[int, int]] = set()
        if len(dists):
            self.lowest_separation = min(self.lowest_separation, float(dists.min()))
        for (i, j), d in zip(pairs, dists):
            if d < self.min_separation:
                key = (ids[i], ids[j])
                hard.add(key)
                if key not in self._hard_pairs:
                    self.total_hard_violations += 1
                    if self._events is not None:
                        self._events.emit(EventCategory.COLLISION, "min_separation_breach",
                                          f"D{key[0]:02d} <-> D{key[1]:02d} at {d:.2f} m "
                                          f"(hard floor {self.min_separation:.1f} m)",
                                          severity=Severity.CRITICAL, time=time, drone_a=key[0], drone_b=key[1],
                                          distance=round(float(d), 3))
            level = self.classify(float(d))
            if level == CollisionState.CLEAR:
                continue
            counts[level] += 1
            for k in (i, j):
                if COLLISION_SEVERITY[level] > COLLISION_SEVERITY[states[k]]:
                    states[k] = level
                if d < nearest[k]:
                    nearest[k] = d
            key = (ids[i], ids[j])
            current[key] = level
            previous = self._pair_levels.get(key, CollisionState.CLEAR)
            if COLLISION_SEVERITY[level] > COLLISION_SEVERITY[previous] and level != CollisionState.WARNING:
                self._record(key, level, float(d), time)
        self._pair_levels = current
        self._hard_pairs = hard
        return CollisionReport(
            states=states,
            nearest=nearest,
            min_separation=float(dists.min()) if len(dists) else float("inf"),
            warning_pairs=counts[CollisionState.WARNING],
            avoidance_pairs=counts[CollisionState.AVOIDANCE],
            collision_pairs=counts[CollisionState.COLLISION],
        )

    def _record(self, key: tuple[int, int], level: CollisionState, distance: float, time: float) -> None:
        if level == CollisionState.COLLISION:
            self.total_collisions += 1
            kind, severity = "collision", Severity.CRITICAL
        else:
            self.total_violations += 1
            kind, severity = "separation_violation", Severity.WARNING
        if self._events is not None:
            self._events.emit(
                EventCategory.COLLISION, kind,
                f"D{key[0]:02d} <-> D{key[1]:02d} at {distance:.2f} m ({level})",
                severity=severity, time=time, drone_a=key[0], drone_b=key[1],
                distance=round(distance, 3), level=str(level),
            )
