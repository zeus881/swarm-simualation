"""ORCA (Optimal Reciprocal Collision Avoidance) in 3D.

Reference: van den Berg, Guy, Lin, Manocha, "Reciprocal n-body Collision Avoidance" (ISRR 2011) and
the RVO2-3D library. Notation for agent ``i`` and neighbour ``j``::

    p = p_j - p_i        relative position
    v = v_i - v_j        relative velocity
    R                    combined safety radius
    tau                  time horizon

The velocity obstacle ``VO = { v | exists t in [0, tau]: |t v - p| < R }`` is a cone truncated by a
sphere of radius ``R / tau`` centred at ``p / tau``. ``u`` is the smallest change of relative velocity
that leaves the VO, ``n`` the outward normal at the closest boundary point. With reciprocal
responsibility each agent takes half of ``u``::

    ORCA_i|j = { v | (v - (v_i + share * u)) . n >= 0 },   share = 1/2 (or 1 if j does not manoeuvre)

The new velocity is the point closest to the preferred velocity inside all half-spaces and inside the
speed sphere, found with the incremental randomised linear program of RVO2-3D (``linear_program3``).
If the program is infeasible (dense conflicts) the caller falls back to a potential field.

Half-spaces are built for all neighbour pairs at once with NumPy; the linear programs run per agent
in plain Python floats (much faster than NumPy for 3-vectors), and only for agents whose preferred
velocity actually violates one of their constraints.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

EPSILON = 1e-7

Plane = tuple[float, float, float, float, float, float]
"""(point_x, point_y, point_z, normal_x, normal_y, normal_z); feasible side: (v - point) . normal >= 0."""
Vec = tuple[float, float, float]


# ----------------------------------------------------------------------------- half-space construction

def orca_planes(rel_pos: np.ndarray, rel_vel: np.ndarray, v_self: np.ndarray, radius: float, tau: float,
                recovery_time: float, share: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ORCA half-spaces for ``m`` ordered agent/neighbour pairs.

    Args:
        rel_pos: (m, 3) ``p_j - p_i``.
        rel_vel: (m, 3) ``v_i - v_j``.
        v_self: (m, 3) current velocity of agent ``i``.
        radius: combined safety radius ``R``.
        tau: time horizon of the velocity obstacle.
        recovery_time: when already inside ``R``, the time over which the overlap is resolved
            (RVO2 uses one time step; a larger value respects the vehicle's acceleration limits).
        share: (m,) responsibility of agent ``i`` (0.5 reciprocal, 1.0 if ``j`` cannot manoeuvre).

    Returns:
        ``(points, normals)``, both (m, 3).
    """
    m = len(rel_pos)
    normals = np.zeros((m, 3))
    u = np.zeros((m, 3))
    if m == 0:
        return normals.copy(), normals
    r2 = radius * radius
    dist2 = np.einsum("ij,ij->i", rel_pos, rel_pos)
    outside = dist2 > r2

    # --- no overlap: project on the cut-off sphere or on the cone side
    w = rel_vel - rel_pos / tau
    w2 = np.einsum("ij,ij->i", w, w)
    wdot = np.einsum("ij,ij->i", w, rel_pos)
    cutoff = outside & (wdot < 0) & (wdot * wdot > r2 * w2)
    if np.any(cutoff):
        wl = np.sqrt(np.maximum(w2[cutoff], EPSILON))
        unit = w[cutoff] / wl[:, None]
        normals[cutoff] = unit
        u[cutoff] = (radius / tau - wl)[:, None] * unit

    cone = outside & ~cutoff
    if np.any(cone):
        p, v = rel_pos[cone], rel_vel[cone]
        a = dist2[cone]
        b = np.einsum("ij,ij->i", p, v)
        cr = np.cross(p, v)
        c = np.einsum("ij,ij->i", v, v) - np.einsum("ij,ij->i", cr, cr) / (a - r2)
        t = (b + np.sqrt(np.maximum(b * b - a * c, 0.0))) / a
        ww = v - t[:, None] * p
        wwl = np.linalg.norm(ww, axis=1)
        unit = np.zeros_like(ww)
        ok = wwl > 1e-9
        unit[ok] = ww[ok] / wwl[ok][:, None]
        if np.any(~ok):
            # Exactly head-on: the relative velocity lies on the cone axis and every perpendicular is
            # equally short. Turn right of the line of sight - the rule is symmetric, so both agents
            # veer to their own right and separate.
            unit[~ok] = _right_of(p[~ok])
        normals[cone] = unit
        u[cone] = (radius * t - wwl)[:, None] * unit

    # --- already inside R: push apart over the recovery time
    inside = ~outside
    if np.any(inside):
        wi = rel_vel[inside] - rel_pos[inside] / recovery_time
        wl = np.linalg.norm(wi, axis=1)
        unit = np.zeros_like(wi)
        ok = wl > 1e-9
        unit[ok] = wi[ok] / wl[ok][:, None]
        unit[~ok] = -_safe_unit(rel_pos[inside][~ok])
        normals[inside] = unit
        u[inside] = (radius / recovery_time - wl)[:, None] * unit

    points = v_self + share[:, None] * u
    return points, normals


def _right_of(p: np.ndarray) -> np.ndarray:
    """Horizontal unit vector to the right of the direction ``p`` (ENU, z up)."""
    right = np.stack((p[:, 1], -p[:, 0], np.zeros(len(p))), axis=1)
    n = np.linalg.norm(right, axis=1)
    right[n > 1e-9] /= n[n > 1e-9][:, None]
    right[n <= 1e-9] = (1.0, 0.0, 0.0)   # vertical line of sight: step East
    return right


def _safe_unit(p: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(p, axis=1)
    out = np.tile(np.array([1.0, 0.0, 0.0]), (len(p), 1))
    out[n > 1e-9] = p[n > 1e-9] / n[n > 1e-9][:, None]
    return out


# ----------------------------------------------------------------------------- linear programs (RVO2-3D)

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Sequence[float], b: Sequence[float]) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _violates(plane: Plane, v: Sequence[float]) -> bool:
    return plane[3] * (plane[0] - v[0]) + plane[4] * (plane[1] - v[1]) + plane[5] * (plane[2] - v[2]) > 0.0


def _lp1(planes: list[Plane], plane_no: int, line_point: Vec, line_dir: Vec, radius: float, opt: Vec,
         direction_opt: bool) -> Vec | None:
    """Optimise along a line inside the speed sphere, subject to planes[0:plane_no]."""
    dot = _dot(line_point, line_dir)
    disc = dot * dot + radius * radius - _dot(line_point, line_point)
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    t_left, t_right = -dot - sq, -dot + sq
    for k in range(plane_no):
        pl = planes[k]
        numerator = (pl[0] - line_point[0]) * pl[3] + (pl[1] - line_point[1]) * pl[4] + (pl[2] - line_point[2]) * pl[5]
        denominator = line_dir[0] * pl[3] + line_dir[1] * pl[4] + line_dir[2] * pl[5]
        if denominator * denominator <= EPSILON:
            if numerator > 0.0:
                return None
            continue
        t = numerator / denominator
        if denominator >= 0.0:
            t_left = max(t_left, t)
        else:
            t_right = min(t_right, t)
        if t_left > t_right:
            return None
    if direction_opt:
        t = t_right if _dot(opt, line_dir) > 0.0 else t_left
    else:
        t = _dot(line_dir, (opt[0] - line_point[0], opt[1] - line_point[1], opt[2] - line_point[2]))
        t = min(max(t, t_left), t_right)
    return (line_point[0] + t * line_dir[0], line_point[1] + t * line_dir[1], line_point[2] + t * line_dir[2])


def _lp2(planes: list[Plane], plane_no: int, radius: float, opt: Vec, direction_opt: bool) -> Vec | None:
    """Optimise on plane ``plane_no`` inside the speed sphere, subject to planes[0:plane_no]."""
    pl = planes[plane_no]
    normal = (pl[3], pl[4], pl[5])
    plane_dist = pl[0] * pl[3] + pl[1] * pl[4] + pl[2] * pl[5]
    plane_dist2 = plane_dist * plane_dist
    r2 = radius * radius
    if plane_dist2 > r2:
        return None
    plane_r2 = r2 - plane_dist2
    center = (plane_dist * normal[0], plane_dist * normal[1], plane_dist * normal[2])
    if direction_opt:
        on = _dot(opt, normal)
        proj = (opt[0] - on * normal[0], opt[1] - on * normal[1], opt[2] - on * normal[2])
        proj2 = _dot(proj, proj)
        if proj2 <= EPSILON:
            result = center
        else:
            s = math.sqrt(plane_r2 / proj2)
            result = (center[0] + s * proj[0], center[1] + s * proj[1], center[2] + s * proj[2])
    else:
        k = (pl[0] - opt[0]) * normal[0] + (pl[1] - opt[1]) * normal[1] + (pl[2] - opt[2]) * normal[2]
        result = (opt[0] + k * normal[0], opt[1] + k * normal[1], opt[2] + k * normal[2])
        if _dot(result, result) > r2:
            rel = (result[0] - center[0], result[1] - center[1], result[2] - center[2])
            rel2 = _dot(rel, rel)
            s = math.sqrt(plane_r2 / rel2) if rel2 > EPSILON else 0.0
            result = (center[0] + s * rel[0], center[1] + s * rel[1], center[2] + s * rel[2])
    for k in range(plane_no):
        other = planes[k]
        if not _violates(other, result):
            continue
        cross = _cross((other[3], other[4], other[5]), normal)
        c2 = _dot(cross, cross)
        if c2 <= EPSILON:
            return None    # (almost) parallel planes and ``other`` rules out this one entirely
        cl = math.sqrt(c2)
        line_dir = (cross[0] / cl, cross[1] / cl, cross[2] / cl)
        line_normal = _cross(line_dir, normal)
        denom = _dot(line_normal, (other[3], other[4], other[5]))
        if abs(denom) <= EPSILON:
            return None
        s = ((other[0] - pl[0]) * other[3] + (other[1] - pl[1]) * other[4] + (other[2] - pl[2]) * other[5]) / denom
        line_point = (pl[0] + s * line_normal[0], pl[1] + s * line_normal[1], pl[2] + s * line_normal[2])
        result = _lp1(planes, k, line_point, line_dir, radius, opt, direction_opt)
        if result is None:
            return None
    return result


def linear_program3(planes: list[Plane], radius: float, opt: Vec, direction_opt: bool = False) -> tuple[Vec, int]:
    """Velocity closest to ``opt`` satisfying every plane, with ``|v| <= radius``.

    Returns ``(result, fail_index)``; ``fail_index == len(planes)`` means success, otherwise the
    constraints are infeasible from that plane on and ``result`` satisfies ``planes[:fail_index]``.
    """
    if direction_opt:
        result = (opt[0] * radius, opt[1] * radius, opt[2] * radius)
    else:
        n2 = _dot(opt, opt)
        if n2 > radius * radius:
            s = radius / math.sqrt(n2)
            result = (opt[0] * s, opt[1] * s, opt[2] * s)
        else:
            result = (float(opt[0]), float(opt[1]), float(opt[2]))
    for k, plane in enumerate(planes):
        if _violates(plane, result):
            new = _lp2(planes, k, radius, opt, direction_opt)
            if new is None:
                return result, k
            result = new
    return result, len(planes)


def solve_orca(points: np.ndarray, normals: np.ndarray, v_pref: np.ndarray, max_speed: float) -> tuple[np.ndarray, bool]:
    """Solve one agent's ORCA program. Returns ``(velocity, feasible)``."""
    planes: list[Plane] = np.hstack((points, normals)).tolist()   # plain floats: fast scalar maths
    opt = (float(v_pref[0]), float(v_pref[1]), float(v_pref[2]))
    result, fail = linear_program3(planes, max_speed, opt)
    return np.array(result), fail == len(planes)
