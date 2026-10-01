"""Static obstacles (buildings, towers, trees) loaded from a YAML scene file.

Scene file (``environment.scene_file``)::

    obstacles:
      - {type: building, name: Hangar, x: 320, y: 180, width: 60, depth: 35, height: 22, rotation: 15}
      - {type: tower, name: Mast, x: -300, y: 260, radius: 3, height: 70}
      - {type: tree, x: 250, y: -300, radius: 4, height: 12}
      - {type: forest, name: Woods, x: -350, y: -280, radius: 70, count: 30, height: [8, 16],
         tree_radius: [2.5, 4.5], seed: 3}

Positions are local ENU metres; heights are metres above the terrain at the obstacle's centre;
``rotation`` turns a building counter-clockwise (degrees, ENU). Unknown keys are rejected.

Geometry is an extruded 2-D shape (oriented rectangle or circle) from the ground to ``top``. The
signed distance from a point ``p`` combines the horizontal distance ``d_h`` to the footprint (negative
inside) and the vertical distance ``d_v = p_z - top``::

    sdf = hypot(max(d_h, 0), max(d_v, 0)) + min(max(d_h, d_v), 0)

Neighbour queries go through the platform's KD-tree :class:`~simulation.spatial.SpatialIndex` over the
obstacle centres (radius = influence + largest bounding radius), and only the candidates it returns
get the exact signed-distance computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from .spatial import SpatialIndex

OBSTACLE_TYPES = ("building", "tower", "tree", "forest")
_KEYS = {
    "building": {"type", "name", "x", "y", "width", "depth", "height", "rotation"},
    "tower": {"type", "name", "x", "y", "radius", "height"},
    "tree": {"type", "name", "x", "y", "radius", "height"},
    "forest": {"type", "name", "x", "y", "radius", "count", "height", "tree_radius", "seed"},
}


class SceneError(ValueError):
    """Invalid scene file."""


@dataclass(slots=True)
class Obstacle:
    kind: str                      # building | tower | tree
    name: str
    x: float
    y: float
    base: float                    # ground height at the centre [m, ENU z]
    height: float                  # above the base [m]
    radius: float = 0.0            # cylinders
    half_w: float = 0.0            # boxes: half extents in the local frame
    half_d: float = 0.0
    rotation: float = 0.0          # rad, CCW

    @property
    def top(self) -> float:
        return self.base + self.height

    @property
    def is_box(self) -> bool:
        return self.kind == "building"

    @property
    def bound_radius(self) -> float:
        return math.hypot(self.half_w, self.half_d) if self.is_box else self.radius

    def footprint(self) -> list[list[float]]:
        if self.is_box:
            c, s = math.cos(self.rotation), math.sin(self.rotation)
            pts = [(-self.half_w, -self.half_d), (self.half_w, -self.half_d), (self.half_w, self.half_d), (-self.half_w, self.half_d)]
            return [[round(self.x + c * a - s * b, 2), round(self.y + s * a + c * b, 2)] for a, b in pts]
        return [[round(self.x + self.radius * math.cos(t), 2), round(self.y + self.radius * math.sin(t), 2)]
                for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)]

    def to_dict(self) -> dict[str, Any]:
        out = {"kind": self.kind, "name": self.name, "x": round(self.x, 2), "y": round(self.y, 2),
               "base": round(self.base, 2), "height": round(self.height, 2)}
        if self.is_box:
            out.update(width=round(2 * self.half_w, 2), depth=round(2 * self.half_d, 2),
                       rotation=round(math.degrees(self.rotation), 2))
        else:
            out["radius"] = round(self.radius, 2)
        return out


def _num(d: Mapping[str, Any], key: str, where: str, *, default: float | None = None, positive: bool = False) -> float:
    v = d.get(key, default)
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise SceneError(f"{where}: '{key}' must be a number")
    if positive and v <= 0:
        raise SceneError(f"{where}: '{key}' must be > 0")
    return float(v)


def _range(d: Mapping[str, Any], key: str, where: str, default: tuple[float, float]) -> tuple[float, float]:
    v = d.get(key, list(default))
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v, v]
    if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(isinstance(a, (int, float)) and a > 0 for a in v) or v[0] > v[1]:
        raise SceneError(f"{where}: '{key}' must be a positive number or [min, max]")
    return float(v[0]), float(v[1])


def parse_scene(data: Any, ground: Callable[[float, float], float]) -> list[Obstacle]:
    """Validate a scene mapping and build the obstacle list (forests expanded into trees)."""
    if data is None:
        return []
    if not isinstance(data, Mapping) or set(data) - {"obstacles"}:
        raise SceneError("scene must be a mapping with an 'obstacles' list")
    items = data.get("obstacles") or []
    if not isinstance(items, list) or len(items) > 2000:
        raise SceneError("'obstacles' must be a list of at most 2000 entries")
    out: list[Obstacle] = []
    for k, item in enumerate(items):
        where = f"obstacles[{k}]"
        if not isinstance(item, Mapping):
            raise SceneError(f"{where} must be a mapping")
        kind = item.get("type")
        if kind not in OBSTACLE_TYPES:
            raise SceneError(f"{where}: type must be one of {', '.join(OBSTACLE_TYPES)}")
        unknown = set(item) - _KEYS[kind]
        if unknown:
            raise SceneError(f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
        name = str(item.get("name") or f"{kind.capitalize()} {k + 1}")[:40]
        x, y = _num(item, "x", where), _num(item, "y", where)
        if kind == "building":
            out.append(Obstacle("building", name, x, y, ground(x, y), _num(item, "height", where, positive=True),
                                half_w=_num(item, "width", where, positive=True) / 2,
                                half_d=_num(item, "depth", where, positive=True) / 2,
                                rotation=math.radians(_num(item, "rotation", where, default=0.0))))
        elif kind in ("tower", "tree"):
            out.append(Obstacle(kind, name, x, y, ground(x, y), _num(item, "height", where, positive=True),
                                radius=_num(item, "radius", where, positive=True)))
        else:
            radius = _num(item, "radius", where, positive=True)
            count = int(_num(item, "count", where, default=20, positive=True))
            if count > 1000:
                raise SceneError(f"{where}: count must be <= 1000")
            h_lo, h_hi = _range(item, "height", where, (8.0, 15.0))
            r_lo, r_hi = _range(item, "tree_radius", where, (2.0, 4.0))
            rng = np.random.default_rng(int(_num(item, "seed", where, default=0)))
            for t in range(count):
                a, rr = rng.uniform(0, 2 * math.pi), radius * math.sqrt(rng.uniform(0, 1))
                tx, ty = x + rr * math.cos(a), y + rr * math.sin(a)
                out.append(Obstacle("tree", f"{name} #{t + 1}", tx, ty, ground(tx, ty), float(rng.uniform(h_lo, h_hi)),
                                    radius=float(rng.uniform(r_lo, r_hi))))
    return out


def load_scene(path: Path, ground: Callable[[float, float], float]) -> list[Obstacle]:
    if not path.is_file():
        raise SceneError(f"scene file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SceneError(f"invalid YAML in {path.name}: {exc}") from None
    return parse_scene(data, ground)


class ObstacleManager:
    """Signed distances, normals, segment checks and collision tests against the static obstacles."""

    def __init__(self, obstacles: list[Obstacle]) -> None:
        self.obstacles = obstacles
        self.version = 1
        n = len(obstacles)
        self.centres = np.array([[o.x, o.y] for o in obstacles]).reshape(n, 2)
        self.top = np.array([o.top for o in obstacles])
        self.is_box = np.array([o.is_box for o in obstacles], dtype=bool)
        self.radius = np.array([o.radius for o in obstacles])
        self.half = np.array([[o.half_w, o.half_d] for o in obstacles]).reshape(n, 2)
        self.cos = np.array([math.cos(o.rotation) for o in obstacles])
        self.sin = np.array([math.sin(o.rotation) for o in obstacles])
        self.max_bound = max((o.bound_radius for o in obstacles), default=0.0)
        self.index = SpatialIndex()
        self.index.rebuild(self.centres)

    def __len__(self) -> int:
        return len(self.obstacles)

    # ------------------------------------------------------------------ geometry
    def _horizontal(self, px: np.ndarray, py: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed horizontal distance to the footprints of obstacles ``k`` and the outward unit normal (2-D)."""
        dx, dy = px - self.centres[k, 0], py - self.centres[k, 1]
        d = np.empty(len(k))
        n = np.zeros((len(k), 2))
        box = self.is_box[k]
        if np.any(~box):
            kc = ~box
            r = np.hypot(dx[kc], dy[kc])
            d[kc] = r - self.radius[k[kc]]
            safe = np.where(r > 1e-9, r, 1.0)
            n[kc] = np.where(r[:, None] > 1e-9, np.stack((dx[kc], dy[kc]), axis=1) / safe[:, None], [1.0, 0.0])
        if np.any(box):
            kb = k[box]
            c, s = self.cos[kb], self.sin[kb]
            qx, qy = c * dx[box] + s * dy[box], -s * dx[box] + c * dy[box]       # into the box frame
            hw, hd = self.half[kb, 0], self.half[kb, 1]
            ox, oy = np.abs(qx) - hw, np.abs(qy) - hd
            out_x, out_y = np.maximum(ox, 0.0), np.maximum(oy, 0.0)
            outside = np.hypot(out_x, out_y)
            inside = np.minimum(np.maximum(ox, oy), 0.0)
            d[box] = outside + inside
            # local normal: towards the nearest face (outside: from the nearest point; inside: nearest face)
            lx = np.where(outside > 1e-9, np.sign(qx) * out_x, np.where(ox > oy, np.sign(qx), 0.0))
            ly = np.where(outside > 1e-9, np.sign(qy) * out_y, np.where(ox > oy, 0.0, np.sign(qy)))
            ln = np.hypot(lx, ly)
            lx, ly = np.where(ln > 1e-9, lx / np.maximum(ln, 1e-12), 1.0), np.where(ln > 1e-9, ly / np.maximum(ln, 1e-12), 0.0)
            n[box] = np.stack((c * lx - s * ly, s * lx + c * ly), axis=1)
        return d, n

    def sdf_pairs(self, points: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed 3-D distance from ``points[i]`` to obstacle ``k[i]`` and the outward unit normal."""
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        k = np.asarray(k, dtype=int)
        d_h, n_h = self._horizontal(p[:, 0], p[:, 1], k)
        d_v = p[:, 2] - self.top[k]
        oh, ov = np.maximum(d_h, 0.0), np.maximum(d_v, 0.0)
        sdf = np.hypot(oh, ov) + np.minimum(np.maximum(d_h, d_v), 0.0)
        normal = np.zeros((len(k), 3))
        above_clear = (d_v > 0) & (d_h > 0)
        # corner region: blend of horizontal and vertical; side: horizontal; roof: vertical
        normal[:, :2] = n_h * np.where(above_clear, oh / np.maximum(np.hypot(oh, ov), 1e-9), 1.0)[:, None]
        normal[:, 2] = np.where(above_clear, ov / np.maximum(np.hypot(oh, ov), 1e-9), 0.0)
        roof = (d_h <= 0) & (d_v > d_h)            # over the footprint: nearest surface is the roof
        normal[roof] = [0.0, 0.0, 1.0]
        return sdf, normal

    def candidates(self, points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """(point index, obstacle index) pairs whose centres are within ``radius + max_bound`` (KD-tree)."""
        if not self.obstacles:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        hits = self.index.query_ball(pts[:, :2], radius + self.max_bound)
        pi = [i for i, h in enumerate(hits) for _ in h]
        ok = [j for h in hits for j in h]
        return np.array(pi, dtype=int), np.array(ok, dtype=int)

    def nearest(self, points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per point: distance to the nearest obstacle within ``radius`` (inf if none), its normal and index."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        dist = np.full(len(pts), np.inf)
        normal = np.zeros((len(pts), 3))
        which = np.full(len(pts), -1)
        pi, ok = self.candidates(pts, radius)
        if len(pi):
            sdf, nrm = self.sdf_pairs(pts[pi], ok)
            order = np.lexsort((sdf, pi))                      # smallest sdf first within each point
            first = np.r_[True, pi[order][1:] != pi[order][:-1]]
            sel = order[first]
            dist[pi[sel]] = sdf[sel]
            normal[pi[sel]] = nrm[sel]
            which[pi[sel]] = ok[sel]
        return dist, normal, which

    def clearance_along(self, a: np.ndarray, b: np.ndarray, step: float = 2.0) -> float:
        """Smallest obstacle distance along the segment a -> b (sampled every ``step`` metres)."""
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        pts = a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]
        d, _, _ = self.nearest(pts, 50.0)
        return float(d.min())

    def segment_clear(self, a: np.ndarray, b: np.ndarray, clearance: float) -> bool:
        return self.clearance_along(a, b, step=min(2.0, max(clearance / 2, 0.5))) >= clearance

    def describe(self) -> dict[str, Any]:
        return {"count": len(self.obstacles), "version": self.version}

    def to_list(self) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.obstacles]
