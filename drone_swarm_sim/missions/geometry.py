"""Planar polygon geometry in the local ENU frame (x = East, y = North, metres)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

Polygon = np.ndarray
"""(k, 2) float array of vertices, implicitly closed (last vertex connects to the first)."""


class PolygonError(ValueError):
    """Invalid polygon."""


def as_polygon(value: Any, name: str = "polygon", *, min_points: int = 3, max_points: int = 200) -> Polygon:
    """Validate ``[[x, y], ...]`` and return a (k, 2) array. Rejects degenerate and self-intersecting shapes."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        raise PolygonError(f"{name} must be a list of [x, y] points") from None
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise PolygonError(f"{name} must be a list of [x, y] points")
    arr = arr[:, :2]
    if not np.all(np.isfinite(arr)):
        raise PolygonError(f"{name} contains non-finite coordinates")
    if len(arr) > 1 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]                      # accept explicitly closed rings
    if not min_points <= len(arr) <= max_points:
        raise PolygonError(f"{name} needs {min_points}..{max_points} vertices (got {len(arr)})")
    if abs(polygon_area(arr)) < 1.0:
        raise PolygonError(f"{name} is degenerate (area below 1 m²)")
    if self_intersects(arr):
        raise PolygonError(f"{name} intersects itself")
    return arr


def polygon_area(poly: Polygon) -> float:
    """Signed area (shoelace): positive for counter-clockwise vertex order."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def points_in_polygon(points: np.ndarray, poly: Polygon) -> np.ndarray:
    """Even-odd ray casting for many points at once: (n, >=2) -> (n,) bool. Boundary points are unspecified."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, np.asarray(points).shape[-1])[:, :2]
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    x, y = pts[:, 0][:, None], pts[:, 1][:, None]
    x1, y1 = poly[:, 0][None, :], poly[:, 1][None, :]
    x2, y2 = np.roll(poly[:, 0], -1)[None, :], np.roll(poly[:, 1], -1)[None, :]
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    hits = straddles & (x < x_cross)
    return (np.count_nonzero(hits, axis=1) % 2) == 1


def point_in_polygon(point: Sequence[float], poly: Polygon) -> bool:
    return bool(points_in_polygon(np.asarray(point, dtype=np.float64)[None, :2], poly)[0])


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def segments_intersect(p1: Sequence[float], p2: Sequence[float], q1: Sequence[float], q2: Sequence[float]) -> bool:
    """Proper or touching intersection of segments p1-p2 and q1-q2."""
    p1, p2, q1, q2 = (np.asarray(v, dtype=np.float64)[:2] for v in (p1, p2, q1, q2))
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if d1 * d2 < 0 and d3 * d4 < 0:          # endpoints strictly on opposite sides of each other's line
        return True

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: float) -> bool:
        return abs(d) < 1e-9 and min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 \
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9

    return on_segment(q1, q2, p1, d1) or on_segment(q1, q2, p2, d2) or on_segment(p1, p2, q1, d3) \
        or on_segment(p1, p2, q2, d4)


def self_intersects(poly: Polygon) -> bool:
    """True if two non-adjacent edges of the polygon intersect."""
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                     # adjacent through the closing edge
            if segments_intersect(a1, a2, poly[j], poly[(j + 1) % n]):
                return True
    return False


def segment_crosses_polygon(p1: Sequence[float], p2: Sequence[float], poly: Polygon) -> bool:
    """True if the segment enters the polygon (crosses an edge or has an endpoint inside)."""
    if point_in_polygon(p1, poly) or point_in_polygon(p2, poly):
        return True
    n = len(poly)
    return any(segments_intersect(p1, p2, poly[i], poly[(i + 1) % n]) for i in range(n))


def distance_to_polygon_edge(point: Sequence[float], poly: Polygon) -> float:
    """Shortest distance from a point to the polygon boundary."""
    p = np.asarray(point, dtype=np.float64)[:2]
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    closest = a + ab * t[:, None]
    return float(np.min(np.linalg.norm(closest - p, axis=1)))


def polygon_to_list(poly: Polygon, digits: int = 2) -> list[list[float]]:
    return [[round(float(x), digits), round(float(y), digits)] for x, y in poly]


def bounds_contains_polygon(poly: Polygon, bounds_min: Sequence[float], bounds_max: Sequence[float]) -> bool:
    return bool(np.all(poly >= np.asarray(bounds_min)[:2] - 1e-6) and np.all(poly <= np.asarray(bounds_max)[:2] + 1e-6))


def path_length(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def heading_deg(dx: float, dy: float) -> float:
    """Compass heading of a direction vector (0 = North, clockwise)."""
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 360.0
