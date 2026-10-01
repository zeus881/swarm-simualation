"""Area survey: lawnmower (boustrophedon) coverage of a polygon, split between N drones.

1. **Line spacing.** Given directly, or from the camera footprint and the side overlap::

       footprint = 2 * alt * tan(hfov / 2)          spacing = footprint * (1 - overlap)

2. **Sweep lines.** The polygon is rotated so the sweep direction is +x; horizontal lines at
   ``y = y_min + spacing/2 + k*spacing`` are intersected with every edge and the crossings are paired
   (even-odd rule), so concave polygons give several segments on one line.
3. **Cells** (boustrophedon decomposition, simplified). Consecutive lines with the same number of
   segments form a region; segment ``j`` of every line in a region belongs to cell ``j``. Each cell is
   covered completely before the next one, so a concave area (a U, an L) is crossed between cells once
   instead of on every line.
4. **Split.** The ordered sweep passes are divided into N contiguous blocks of roughly equal flight
   length, so every drone covers a compact part of the area and the drones do not cross lanes.
5. **Order.** The direction alternates pass by pass (boustrophedon); passes are rotated back to ENU and
   become WAYPOINT pairs (start, end) at the survey altitude, followed by the finish action.

The default sweep direction is along the polygon's longest edge, which minimises the number of turns
for elongated areas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import Polygon, as_polygon, polygon_area
from .model import MissionAction, Waypoint


@dataclass(slots=True)
class SurveyResult:
    tracks: list[list[Waypoint]]
    line_spacing: float
    angle_deg: float
    lines: int
    area_m2: float
    track_lengths: list[float]

    def stats(self) -> dict[str, Any]:
        return {"line_spacing": round(self.line_spacing, 2), "angle": round(self.angle_deg, 1), "lines": self.lines,
                "area_m2": round(self.area_m2, 1), "tracks": len(self.tracks),
                "track_lengths": [round(v, 1) for v in self.track_lengths]}


def spacing_from_overlap(altitude: float, hfov_deg: float, overlap_pct: float) -> float:
    """Line spacing for a camera with horizontal field of view ``hfov_deg`` at ``altitude`` [m AGL]."""
    footprint = 2.0 * altitude * math.tan(math.radians(hfov_deg) / 2.0)
    return footprint * (1.0 - overlap_pct / 100.0)


def longest_edge_heading(poly: Polygon) -> float:
    """Compass heading of the polygon's longest edge."""
    edges = np.roll(poly, -1, axis=0) - poly
    k = int(np.argmax(np.linalg.norm(edges, axis=1)))
    dx, dy = edges[k]
    return (90.0 - math.degrees(math.atan2(dy, dx))) % 180.0


def _rotation(heading_deg: float) -> np.ndarray:
    """Rotation matrix taking the sweep direction (compass heading) onto +x."""
    theta = math.radians(90.0 - heading_deg)          # ENU angle of the sweep direction
    c, s = math.cos(-theta), math.sin(-theta)
    return np.array([[c, -s], [s, c]])


def sweep_lines(poly: Polygon, spacing: float, heading_deg: float) -> list[list[tuple[float, float, float]]]:
    """Sweep lines in the rotated frame: per line, the segments ``(y, x_start, x_end)`` sorted by x."""
    rot = _rotation(heading_deg)
    p = poly @ rot.T
    y_min, y_max = float(p[:, 1].min()), float(p[:, 1].max())
    a, b = p, np.roll(p, -1, axis=0)
    lines = []
    y = y_min + spacing / 2.0
    if y > y_max:                                     # area narrower than one line: one pass through the middle
        y = 0.5 * (y_min + y_max)
    while y <= y_max + 1e-9:
        straddle = (a[:, 1] > y) != (b[:, 1] > y)
        if np.any(straddle):
            xa, ya, xb, yb = a[straddle, 0], a[straddle, 1], b[straddle, 0], b[straddle, 1]
            xs = np.sort(xa + (y - ya) * (xb - xa) / (yb - ya))
            segs = [(y, float(xs[k]), float(xs[k + 1])) for k in range(0, len(xs) - 1, 2) if xs[k + 1] - xs[k] > 0.5]
            if segs:
                lines.append(segs)
        y += spacing
    return lines


def sweep_passes(lines: list[list[tuple[float, float, float]]]) -> list[tuple[float, float, float]]:
    """Order the segments of all lines cell by cell (see module docstring, step 3).

    Returns directed passes ``(y, x_start, x_end)`` in the rotated frame. Within a region the next cell,
    the end (bottom or top) it is entered from, and the direction of every pass are chosen greedily as
    the ones nearest to where the drone currently is - which yields a boustrophedon inside each cell.
    """
    regions: list[list[list[tuple[float, float, float]]]] = []
    for segs in lines:
        if not regions or len(regions[-1][-1]) != len(segs):
            regions.append([])
        regions[-1].append(segs)

    def gap(cur: tuple[float, float] | None, seg: tuple[float, float, float]) -> float:
        if cur is None:
            return 0.0
        y, x0, x1 = seg
        return min(math.hypot(cur[0] - x0, cur[1] - y), math.hypot(cur[0] - x1, cur[1] - y))

    passes: list[tuple[float, float, float]] = []
    cur: tuple[float, float] | None = None
    for region in regions:
        cells = [[line[j] for line in region] for j in range(len(region[0]))]
        while cells:
            _, ci, rev = min((gap(cur, cell[-1] if rev else cell[0]), ci, rev)
                             for ci, cell in enumerate(cells) for rev in (False, True))
            cell = cells.pop(ci)
            for y, x0, x1 in (reversed(cell) if rev else cell):
                if cur is not None and abs(cur[0] - x1) < abs(cur[0] - x0):
                    x0, x1 = x1, x0                   # start the pass at the end nearest to the drone
                passes.append((y, x0, x1))
                cur = (x1, y)
    return passes


def _split(lengths: list[float], parts: int) -> list[tuple[int, int]]:
    """``parts`` contiguous, non-empty blocks ``[start, end)`` of ``lengths`` with roughly equal sums."""
    n = len(lengths)
    parts = max(1, min(parts, n))
    cum = np.cumsum(lengths)
    total = float(cum[-1])
    cuts, prev = [], 0
    for p in range(1, parts):
        target = total * p / parts
        k = int(np.searchsorted(cum, target))              # first line whose cumulative length reaches target
        before = float(cum[k - 1]) if k > 0 else 0.0
        cut = k + 1 if k < n and cum[k] - target < target - before else k
        cut = min(max(cut, prev + 1), n - (parts - p))     # every block keeps at least one line
        cuts.append(cut)
        prev = cut
    bounds = [0, *cuts, n]
    return list(zip(bounds[:-1], bounds[1:]))


def generate_survey(polygon: Any, *, altitude: float, speed: float | None = None, drones: int = 1,
                    line_spacing: float | None = None, overlap: float | None = None, hfov_deg: float = 70.0,
                    angle_deg: float | None = None, finish: str = "RTL") -> SurveyResult:
    """Lawnmower coverage of ``polygon`` split into at most ``drones`` tracks.

    Args:
        polygon: ``[[x, y], ...]`` in local ENU metres.
        altitude: survey altitude [m above home].
        speed: survey speed [m/s] (None = default cruise speed).
        drones: number of tracks to split the area into.
        line_spacing: distance between lines [m]; if None it is derived from ``overlap`` and ``hfov_deg``.
        overlap: side overlap between neighbouring camera footprints [%].
        hfov_deg: camera horizontal field of view [deg].
        angle_deg: sweep direction as a compass heading; None = along the longest polygon edge.
        finish: action appended to every track: RTL | LAND | HOLD (no action).
    """
    poly = as_polygon(polygon, "survey polygon")
    if altitude <= 0:
        raise ValueError("survey altitude must be > 0")
    if drones < 1:
        raise ValueError("drones must be >= 1")
    if line_spacing is None:
        if overlap is None:
            raise ValueError("give line_spacing or overlap")
        if not 0 <= overlap < 100:
            raise ValueError("overlap must be in [0, 100) %")
        line_spacing = spacing_from_overlap(altitude, hfov_deg, overlap)
    if line_spacing < 1.0:
        raise ValueError("line spacing must be >= 1 m")
    finish = finish.upper()
    if finish not in ("RTL", "LAND", "HOLD"):
        raise ValueError("finish must be RTL, LAND or HOLD")
    heading = longest_edge_heading(poly) if angle_deg is None else float(angle_deg) % 360.0

    lines = sweep_lines(poly, line_spacing, heading)
    if not lines:
        raise ValueError("no sweep line intersects the polygon - reduce the line spacing")
    passes = sweep_passes(lines)
    lengths = [abs(x1 - x0) for _, x0, x1 in passes]
    blocks = _split(lengths, min(drones, len(passes)))
    inv = _rotation(heading).T                       # rotated frame -> ENU

    tracks, track_lengths = [], []
    for start, end in blocks:
        pts: list[tuple[float, float]] = []
        for y, a, b in passes[start:end]:            # passes are already directed (boustrophedon)
            pts.append((a, y))
            pts.append((b, y))
        enu = np.asarray(pts) @ inv.T
        track = [Waypoint(float(x), float(y), float(altitude), speed) for x, y in enu]
        if finish == "RTL":
            last = track[-1]
            track.append(Waypoint(last.x, last.y, last.alt, speed, action=MissionAction.RTL))
        elif finish == "LAND":
            track[-1].action = MissionAction.LAND
        tracks.append(track)
        track_lengths.append(float(np.sum(np.linalg.norm(np.diff(enu, axis=0), axis=1))) if len(enu) > 1 else 0.0)
    return SurveyResult(tracks, float(line_spacing), heading, len(lines), abs(polygon_area(poly)), track_lengths)
