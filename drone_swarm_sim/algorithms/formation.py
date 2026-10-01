"""Formation control: shape generation, optimal slot assignment, synchronized transitions, slot tracking.

Frames
------
Shapes are generated in the formation *body frame*: x = forward (direction of travel), y = left,
z = up, then centred on their centroid. World slots are::

    s_k = p_ref + R_z(psi_ref) @ o_k

where ``p_ref`` / ``psi_ref`` are the formation reference position and ENU yaw. With
``altitude_layers = L > 1`` slot ``k`` is lifted to layer ``k mod L``, layers ``layer_spacing`` apart.

Reference modes
---------------
* **virtual** (virtual structure): the reference is a point the operator moves with
  ``swarm_goto``. It travels at ``formation_speed`` and slows down when members lag behind
  (formation keeping), so the shape is not torn apart on the move.
* **leader** (leader-follower): the reference is a leader drone the operator flies normally
  (goto, RTL...). Followers keep their offsets relative to the leader's position and track heading.
  If the leader fails, loses its link, lands or returns home, the follower flying closest to the lead
  slot is **promoted** automatically: it inherits the leader's goto and the others re-slot behind it
  (``swarm.leader_promotion``; when disabled the formation holds on a virtual reference instead).

Assignment
----------
Drones are matched to slots with the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
minimising the sum of *squared* distances. For squared Euclidean costs the optimal assignment has
no crossing straight-line paths.

Synchronized transitions (``formation_transition: synchronized``)
------------------------------------------------------------------
On every shape/member change each drone's slot target moves on the straight line from where the drone
*is* (``a_i``, body frame) to its new slot (``b_i``) with a common time profile::

    o_i(t) = a_i + (b_i - a_i) * s(t/T),    s(x) = 3x^2 - 2x^3   (zero velocity at both ends)

All drones start and finish together, the duration ``T`` respecting the catch-up speed and the climb /
descent limits of the drone with the longest path. This is the CAPT scheme (Turpin, Michael & Kumar,
2014): with the squared-distance optimal assignment, synchronized straight-line trajectories stay
separated whenever start and goal positions are separated, so transitions are smooth and collision free
by construction. Collision avoidance (priority 40) still guards everything.

Slot tracking
-------------
``v_i = v_ref + R(psi) do_i/dt + approach(p_i -> s_i)``: feed-forward of the reference and transition
velocities plus the same braking-profile guidance the autopilot uses, limited to ``formation_catchup_speed``.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from simulation.config import SimConfig
from simulation.drone import heading_to_yaw
from simulation.dynamics import clip_velocity, wrap_angle
from simulation.events import EventBus, EventCategory, Severity
from simulation.guidance import approach_velocity
from simulation.swarm import SwarmContext, VelocityStage
from simulation.types import CommStatus, FlightMode, Vector3

HEADING_RATE = math.radians(30.0)   # rad/s, fastest the formation reference ever turns
TURN_SPEED_SHARE = 0.6              # the outermost slot may swing at most this share of the catch-up speed
REF_ACCEL_SHARE = 0.35              # the virtual reference accelerates at this share of the drones' horizontal limit
LEADER_LOST_MODES = frozenset({FlightMode.LAND, FlightMode.RTL, FlightMode.EMERGENCY})
TRANSITION_SPEED_SHARE = 0.8        # peak transition speed as a share of the catch-up / climb limits
MIN_TRANSITION_S = 1.0


class FormationShape(StrEnum):
    LINE = "line"
    COLUMN = "column"
    V = "v"
    DIAMOND = "diamond"
    GRID = "grid"
    CIRCLE = "circle"
    WEDGE = "wedge"
    CUSTOM = "custom"


# ----------------------------------------------------------------------------- shape generators

def _line(n: int, s: float) -> np.ndarray:
    return np.array([(0.0, (k - (n - 1) / 2.0) * s, 0.0) for k in range(n)])


def _column(n: int, s: float) -> np.ndarray:
    return np.array([(-k * s, 0.0, 0.0) for k in range(n)])


def _v(n: int, s: float, half_angle: float) -> np.ndarray:
    """Apex first, then alternating right/left along two arms at ``half_angle`` from the axis."""
    out = [(0.0, 0.0, 0.0)]
    for k in range(1, n):
        rank, side = (k + 1) // 2, (1.0 if k % 2 else -1.0)
        out.append((-rank * s * math.cos(half_angle), side * rank * s * math.sin(half_angle), 0.0))
    return np.array(out)


def _wedge(n: int, s: float) -> np.ndarray:
    """Filled triangle: row r holds r + 1 drones on a hexagonal lattice (all neighbours at ``s``)."""
    out, r = [], 0
    while len(out) < n:
        for c in range(r + 1):
            if len(out) == n:
                break
            out.append((-r * s * math.sqrt(3) / 2.0, (c - r / 2.0) * s, 0.0))
        r += 1
    return np.array(out)


def _grid(n: int, s: float) -> np.ndarray:
    cols = max(1, math.ceil(math.sqrt(n)))
    return np.array([(-(k // cols) * s, ((k % cols) - (cols - 1) / 2.0) * s, 0.0) for k in range(n)])


def _diamond(n: int, s: float) -> np.ndarray:
    """Concentric diamond rings |a| + |b| = layer on a square lattice (nearest neighbours >= s)."""
    out, layer = [(0.0, 0.0, 0.0)], 1
    while len(out) < n:
        ring = [(a, b) for a in range(-layer, layer + 1) for b in range(-layer, layer + 1) if abs(a) + abs(b) == layer]
        ring.sort(key=lambda ab: (-ab[0], abs(ab[1]), -ab[1]))     # front first, then symmetric pairs
        out.extend((a * s, b * s, 0.0) for a, b in ring)
        layer += 1
    return np.array(out[:n])


def _circle(n: int, s: float) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 3))
    r = max(s / (2.0 * math.sin(math.pi / n)), s / 2.0)       # chord between neighbours = s
    return np.array([(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), 0.0) for k in range(n)])


def formation_offsets(shape: str | FormationShape, n: int, spacing: float, *,
                      v_angle_deg: float = 45.0, custom: Sequence[Sequence[float]] | None = None,
                      layers: int = 1, layer_spacing: float = 6.0) -> np.ndarray:
    """Slot offsets (n, 3) in the body frame, centred on their centroid. Slot 0 is the front/lead slot.

    Args:
        shape: one of :class:`FormationShape`.
        n: number of slots.
        spacing: distance between neighbouring slots [m].
        v_angle_deg: half-angle of the V arms.
        custom: ``[[forward, left, up], ...]`` for the custom shape; missing slots trail behind in a column.
        layers: number of altitude layers; slot ``k`` flies in layer ``k mod layers``.
        layer_spacing: vertical distance between layers [m].
    """
    if n <= 0:
        return np.zeros((0, 3))
    shape = FormationShape(str(shape).lower())
    if shape == FormationShape.LINE:
        pts = _line(n, spacing)
    elif shape == FormationShape.COLUMN:
        pts = _column(n, spacing)
    elif shape == FormationShape.V:
        pts = _v(n, spacing, math.radians(v_angle_deg))
    elif shape == FormationShape.WEDGE:
        pts = _wedge(n, spacing)
    elif shape == FormationShape.GRID:
        pts = _grid(n, spacing)
    elif shape == FormationShape.DIAMOND:
        pts = _diamond(n, spacing)
    elif shape == FormationShape.CIRCLE:
        pts = _circle(n, spacing)
    else:
        given = np.asarray(custom or [], dtype=np.float64).reshape(-1, 3)[:n]
        extra = n - len(given)
        if extra > 0:   # not enough custom slots: trail the remainder in a column behind the shape
            tail_x = (given[:, 0].min() if len(given) else 0.0) - spacing
            given = np.vstack([given, [(tail_x - k * spacing, 0.0, 0.0) for k in range(extra)]]) if len(given) \
                else _column(n, spacing)
        pts = given
    pts = np.array(pts, dtype=np.float64)
    if layers > 1:
        layer = np.arange(n) % layers
        pts[:, 2] += (layer - (layers - 1) / 2.0) * layer_spacing
    return pts - pts.mean(axis=0)


def lead_slot(offsets: np.ndarray) -> int:
    """Index of the front-most slot (ties: closest to the centre line) — where a leader flies."""
    if len(offsets) == 0:
        return 0
    return int(np.lexsort((np.abs(offsets[:, 1]), -np.round(offsets[:, 0], 6)))[0])


def rotate_z(offsets: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return offsets @ rot.T


def assign_slots(positions: np.ndarray, slots: np.ndarray) -> np.ndarray:
    """Optimal drone -> slot assignment (Hungarian algorithm) minimising the sum of squared distances.

    Returns ``idx`` with ``idx[k]`` = slot of drone ``k``. Requires ``len(slots) >= len(positions)``.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    slots = np.asarray(slots, dtype=np.float64).reshape(-1, 3)
    if len(positions) == 0:
        return np.zeros(0, dtype=int)
    if len(slots) < len(positions):
        raise ValueError("more drones than formation slots")
    diff = positions[:, None, :] - slots[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty(len(positions), dtype=int)
    out[rows] = cols
    return out


def smoothstep(x: float) -> tuple[float, float]:
    """``s(x) = 3x^2 - 2x^3`` on [0, 1] and its derivative ``ds/dx``."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x)


def paths_cross(starts: np.ndarray, goals: np.ndarray, min_distance: float) -> bool:
    """True if any two synchronized straight-line paths ``starts -> goals`` come closer than ``min_distance``.

    Uses the closest approach of the relative motion ``d(x) = d0 + (d1 - d0) x`` for ``x`` in [0, 1] (the
    common time profile does not change *where* the paths meet, only when).
    """
    n = len(starts)
    for a in range(n):
        d0 = starts[a] - starts[a + 1:]
        d1 = goals[a] - goals[a + 1:]
        dd = d1 - d0
        dd2 = np.einsum("ij,ij->i", dd, dd)
        x = np.where(dd2 > 1e-12, np.clip(-np.einsum("ij,ij->i", d0, dd) / np.where(dd2 > 1e-12, dd2, 1.0), 0, 1), 0.0)
        closest = np.linalg.norm(d0 + dd * x[:, None], axis=1)
        if np.any(closest < min_distance):
            return True
    return False


# ----------------------------------------------------------------------------- controller

class FormationController(VelocityStage):
    """Velocity-pipeline stage that keeps the member drones on their formation slots."""

    name = "formation"
    priority = 20

    def __init__(self, config: SimConfig, events: EventBus) -> None:
        self.config = config
        self.events = events
        sw = config.swarm
        self.active = False
        self.shape = FormationShape(sw.formation.lower())
        self.spacing = sw.formation_spacing
        self.layers = sw.formation_altitude_layers
        self.layer_spacing = sw.formation_layer_spacing
        self.custom: list[list[float]] = [list(p) for p in sw.custom_formation]
        self.reference_mode = sw.formation_reference
        self.speed = sw.formation_speed
        self.catchup = sw.formation_catchup_speed
        self.synchronized = sw.formation_transition == "synchronized"
        self.leader_promotion = sw.leader_promotion
        self.leader_id: int | None = None
        self.ref_pos: Vector3 = np.zeros(3)
        self.ref_vel: Vector3 = np.zeros(3)
        self.default_heading_locked = sw.formation_heading is not None
        self.ref_yaw = heading_to_yaw(sw.formation_heading) if sw.formation_heading is not None else math.pi / 2
        self.ref_target: Vector3 | None = None
        self.ref_route: list[Vector3] = []                  # pass-through points before ref_target (path planner)
        self.heading_locked = self.default_heading_locked   # True when an explicit heading is in force
        self._members: tuple[int, ...] = ()
        self._assignment: dict[int, int] = {}
        self._offsets = np.zeros((0, 3))
        self._slots_world = np.zeros((0, 3))
        self._lead_offset = np.zeros(3)
        # synchronized transition: per-member body-frame start / goal offsets and timing
        self._trans_start: dict[int, np.ndarray] = {}
        self._trans_goal: dict[int, np.ndarray] = {}
        self._trans_t0 = 0.0
        self._trans_duration = 0.0
        self._time = 0.0
        self._yaw_rate = 0.0                # current turn rate of the reference [rad/s] (slot feed-forward)
        self._ref_speed = 0.0               # current speed of the virtual reference [m/s] (acceleration limited)
        dc = config.drone
        self.ref_accel = REF_ACCEL_SHARE * min(dc.max_acceleration, 9.80665 * math.tan(math.radians(dc.max_tilt_deg)))
        self._leader_goal: tuple[np.ndarray, float] | None = None
        self.promotions = 0
        self.max_error = 0.0
        self._dirty = True

    # ------------------------------------------------------------------ configuration (engine thread)
    def configure(self, *, shape: str | None = None, spacing: float | None = None,
                  reference: str | None = None, leader_id: int | None = None,
                  heading_yaw: float | None = None, layers: int | None = None,
                  layer_spacing: float | None = None, custom: list[list[float]] | None = None) -> None:
        """Change formation parameters; the next tick re-assigns slots and starts a transition."""
        if shape is not None:
            self.shape = FormationShape(shape.lower())
        if spacing is not None:
            self.spacing = float(spacing)
        if layers is not None:
            self.layers = int(layers)
        if layer_spacing is not None:
            self.layer_spacing = float(layer_spacing)
        if custom is not None:
            self.custom = [list(map(float, p)) for p in custom]
        if reference is not None:
            self.reference_mode = reference
        if reference == "leader" or leader_id is not None:
            self.leader_id = leader_id
        if self.reference_mode != "leader":
            self.leader_id = None
        if heading_yaw is not None:
            self.ref_yaw = heading_yaw
            self.heading_locked = True
        self._dirty = True

    def start(self, positions: np.ndarray, altitude: float | None = None) -> None:
        """Activate with the reference at the members' centroid (virtual) — or at the leader."""
        self.active = True
        centroid = positions.mean(axis=0) if len(positions) else np.zeros(3)
        self.ref_pos = centroid.copy()
        if altitude is not None:
            self.ref_pos[2] = altitude
        self.ref_vel = np.zeros(3)
        self._ref_speed = 0.0
        self._yaw_rate = 0.0
        self.ref_target = None
        self._members = ()
        self._dirty = True

    def stop(self) -> None:
        self.active = False
        self.ref_target = None
        self.ref_route = []
        self._members = ()
        self._assignment = {}
        self._slots_world = np.zeros((0, 3))
        self._trans_start, self._trans_goal = {}, {}
        self._leader_goal = None

    def move_to(self, target: Vector3, speed: float | None = None, heading_yaw: float | None = None,
                route: list[Vector3] | None = None) -> None:
        """Move the virtual reference (through ``route`` pass-through points, if given). Without an explicit
        heading the formation faces its direction of travel, unless the configuration fixes the heading."""
        self.ref_target = np.asarray(target, dtype=np.float64).copy()
        self.ref_route = [np.asarray(p, dtype=np.float64).copy() for p in (route or [])]
        if speed:
            self.speed = float(speed)
        if heading_yaw is not None:
            self.ref_yaw = heading_yaw
            self.heading_locked = True
        else:
            self.heading_locked = self.default_heading_locked

    @property
    def transition_progress(self) -> float:
        if self._trans_duration <= 0:
            return 1.0
        return float(min(max((self._time - self._trans_t0) / self._trans_duration, 0.0), 1.0))

    # ------------------------------------------------------------------ pipeline stage
    @staticmethod
    def _is_member(d) -> bool:
        return d.flight_mode == FlightMode.FORMATION and d.swarm_behavior == "formation"

    @staticmethod
    def _leader_lost_reason(drone) -> str | None:
        if drone is None:
            return "removed"
        if drone.failed:
            return f"failed ({drone.failure_reason})"
        if drone.comm_status == CommStatus.LOST:
            return "communication lost"
        if not drone.in_flight:
            return "landed"
        if drone.flight_mode in LEADER_LOST_MODES:
            return f"left the flight ({drone.flight_mode})"
        return None

    def apply(self, ctx: SwarmContext, velocities: np.ndarray) -> np.ndarray:
        if not self.active:
            return velocities
        self._time = ctx.time
        leader_k = None
        if self.reference_mode == "leader" and self.leader_id is not None:
            leader_k = next((k for k, d in enumerate(ctx.drones) if d.id == self.leader_id), None)
            reason = self._leader_lost_reason(ctx.drones[leader_k] if leader_k is not None else None)
            if reason is not None:
                leader_k = self._handle_leader_loss(ctx, reason)
            else:
                leader = ctx.drones[leader_k]
                self._leader_goal = ((leader.target_position.copy(), leader.goto_speed)
                                     if leader.flight_mode == FlightMode.GOTO and leader.target_position is not None
                                     else None)
        idx = [k for k, d in enumerate(ctx.drones) if self._is_member(d)]
        if not idx:
            self.events.emit(EventCategory.MISSION, "formation_ended", "formation has no members left",
                             time=ctx.time)
            self.stop()
            return velocities

        members = tuple(ctx.drones[k].id for k in idx)
        if members != self._members or self._dirty:
            self._rebuild(ctx, idx, members, leader_k)

        self._update_reference(ctx, leader_k)
        s, ds = smoothstep(self.transition_progress) if self.synchronized else (1.0, 0.0)
        rate = ds / self._trans_duration if self._trans_duration > 0 else 0.0
        out = velocities.copy()
        errors = []
        dc = self.config.drone
        slots_world = self.ref_pos + rotate_z(self._offsets, self.ref_yaw)
        self._slots_world = slots_world
        for k in idx:
            drone = ctx.drones[k]
            goal = self._trans_goal[drone.id]
            start = self._trans_start[drone.id]
            offset = start + (goal - start) * s
            arm = rotate_z(offset[None, :], self.ref_yaw)[0]
            slot = self.ref_pos + arm
            # Feed-forward: reference velocity + transition velocity + rotation of the slot about the reference.
            feed = (self.ref_vel + rotate_z(((goal - start) * rate)[None, :], self.ref_yaw)[0]
                    + self._yaw_rate * np.array([-arm[1], arm[0], 0.0]))
            drone.swarm_target = slot
            err = slot - ctx.positions[k]
            errors.append(float(np.linalg.norm(err)))
            track = approach_velocity(ctx.positions[k], slot, cruise_speed=self.catchup, max_climb=dc.max_climb_rate,
                                      max_descent=dc.max_descent_rate, decel=0.5 * drone.params.max_horizontal_accel,
                                      gain=dc.position_gain)
            out[k] = clip_velocity(feed + track, dc.max_horizontal_speed, dc.max_climb_rate, dc.max_descent_rate)
        self.max_error = max(errors) if errors else 0.0
        return out

    def _handle_leader_loss(self, ctx: SwarmContext, reason: str) -> int | None:
        """Promote a follower to leader (or fall back to a virtual reference). Returns the new leader index."""
        old = self.leader_id
        followers = [k for k, d in enumerate(ctx.drones) if self._is_member(d)]
        if self.leader_promotion and followers:
            # Successor: the follower flying the slot closest to the lead position (offsets are relative
            # to the leader), so the shape changes least. Ties: lowest id.
            def rank(k: int) -> tuple[float, int]:
                d = ctx.drones[k]
                slot = self._assignment.get(d.id)
                dist = float(np.linalg.norm(self._offsets[slot])) if slot is not None and slot < len(self._offsets) \
                    else float(np.linalg.norm(ctx.positions[k] - self.ref_pos))
                return dist, d.id
            new_k = min(followers, key=rank)
            successor = ctx.drones[new_k]
            if self._leader_goal is not None:
                target, speed = self._leader_goal
                successor.goto(target, speed=speed)
            else:
                successor.hover()
            self.leader_id = successor.id
            self.promotions += 1
            self._dirty = True
            self.events.emit(EventCategory.MISSION, "leader_promoted",
                             f"formation leader D{old:02d} {reason} - {successor.name} promoted to leader",
                             severity=Severity.WARNING, time=ctx.time, old_leader=old, new_leader=successor.id,
                             reason=reason)
            return new_k
        self.events.emit(EventCategory.MISSION, "leader_lost",
                         f"formation leader D{old:02d} {reason} - holding formation on a virtual reference",
                         severity=Severity.WARNING, time=ctx.time, old_leader=old, reason=reason)
        self.reference_mode, self.leader_id = "virtual", None
        self._leader_goal = None
        self._dirty = True
        return None

    def _rebuild(self, ctx: SwarmContext, idx: list[int], members: tuple[int, ...], leader_k: int | None) -> None:
        sw = self.config.swarm
        n_slots = len(idx) + (1 if leader_k is not None else 0)
        offsets = formation_offsets(self.shape, n_slots, self.spacing, v_angle_deg=sw.v_angle_deg,
                                    custom=self.custom, layers=self.layers, layer_spacing=self.layer_spacing)
        if leader_k is not None:
            # Leader occupies the lead slot; express all offsets relative to it.
            lead = lead_slot(offsets)
            self._lead_offset = offsets[lead].copy()
            offsets = np.delete(offsets, lead, axis=0) - self._lead_offset
        else:
            self._lead_offset = np.zeros(3)
        self._offsets = offsets
        if leader_k is not None:
            self._update_reference(ctx, leader_k)
        world = self.ref_pos + rotate_z(offsets, self.ref_yaw)
        assignment = assign_slots(ctx.positions[idx], world)
        self._assignment = {ctx.drones[k].id: int(a) for k, a in zip(idx, assignment)}

        # Synchronized transition from where every member is now to its new slot (body frame).
        starts = rotate_z(ctx.positions[idx] - self.ref_pos, -self.ref_yaw)
        goals = offsets[assignment]
        self._trans_start = {ctx.drones[k].id: starts[r] for r, k in enumerate(idx)}
        self._trans_goal = {ctx.drones[k].id: goals[r] for r, k in enumerate(idx)}
        delta = goals - starts
        dc = self.config.drone
        need = max(
            float(np.max(np.hypot(delta[:, 0], delta[:, 1]))) / self.catchup if len(delta) else 0.0,
            float(np.max(np.maximum(delta[:, 2], 0.0))) / dc.max_climb_rate if len(delta) else 0.0,
            float(np.max(np.maximum(-delta[:, 2], 0.0))) / dc.max_descent_rate if len(delta) else 0.0,
        )
        # smoothstep peaks at 1.5x the mean speed
        self._trans_duration = max(MIN_TRANSITION_S, 1.5 * need / TRANSITION_SPEED_SHARE) if self.synchronized else 0.0
        self._trans_t0 = ctx.time

        changed = bool(self._members) and members != self._members
        self._members = members
        self._dirty = False
        self.events.emit(EventCategory.MISSION, "formation", f"formation {self.shape.upper()} with {len(idx)} drones"
                         + (f" following D{self.leader_id:02d}" if leader_k is not None else "")
                         + (f", {self.layers} altitude layers" if self.layers > 1 else "")
                         + (" (members changed)" if changed else "")
                         + (f", transition {self._trans_duration:.0f} s" if self.synchronized else ""),
                         time=ctx.time, shape=str(self.shape), members=list(members),
                         transition_s=round(self._trans_duration, 2))

    def _update_reference(self, ctx: SwarmContext, leader_k: int | None) -> None:
        """Advance the reference. The virtual reference behaves like a vehicle: its speed and turn rate
        are acceleration limited, so every slot moves in a way the drones can actually follow."""
        dt = ctx.dt
        turning = False
        if leader_k is not None:
            leader = ctx.drones[leader_k]
            self.ref_pos = ctx.positions[leader_k].copy()
            self.ref_vel = ctx.velocities[leader_k].copy()
            self._ref_speed = float(np.linalg.norm(self.ref_vel))
            if not self.heading_locked:
                fast = math.hypot(self.ref_vel[0], self.ref_vel[1]) > 1.0
                turning = self._turn_towards(math.atan2(self.ref_vel[1], self.ref_vel[0]) if fast else leader.body.yaw, dt)
            if not turning:
                self._settle_turn(dt)
            return
        if self.ref_target is None:
            self.ref_vel = np.zeros(3)
            self._ref_speed = 0.0
            self._settle_turn(dt)
            return
        passing = bool(self.ref_route)
        aim = self.ref_route[0] if passing else self.ref_target
        delta = aim - self.ref_pos
        dist = float(np.linalg.norm(delta))
        if passing and dist < max(5.0, 1.2 * self._ref_speed):
            self.ref_route.pop(0)                             # pass-through point reached: aim at the next one
            aim = self.ref_route[0] if self.ref_route else self.ref_target
            passing = bool(self.ref_route)
            delta = aim - self.ref_pos
            dist = float(np.linalg.norm(delta))
        if dist < 0.05:
            self.ref_pos = self.ref_target.copy()
            self.ref_vel = np.zeros(3)
            self._ref_speed = 0.0
            self.ref_target = None
            self._settle_turn(dt)
            return
        align = 1.0
        if math.hypot(delta[0], delta[1]) > 1.0 and not self.heading_locked:
            travel = math.atan2(delta[1], delta[0])
            turning = self._turn_towards(travel, dt)
            # Turn before moving: a wide formation re-orients on the spot (no translation while the heading
            # error exceeds 60 deg, full speed when aligned), so slot speeds never add up past the catch-up speed.
            align = float(np.clip((math.cos(wrap_angle(travel - self.ref_yaw)) - 0.5) / 0.5, 0.0, 1.0))
        if not turning:
            self._settle_turn(dt)
        # Formation keeping: slow the reference while members are far from their slots.
        keep = float(np.clip(1.0 - (self.max_error - 0.5 * self.spacing) / (2.0 * self.spacing), 0.15, 1.0))
        a = self.ref_accel
        brake = math.inf if passing else math.sqrt(2.0 * a * dist)           # brake only onto the final target
        wanted = min(self.speed * keep * align, brake)
        self._ref_speed = float(np.clip(wanted, self._ref_speed - 2.0 * a * dt, self._ref_speed + a * dt))
        speed = min(self._ref_speed, dist / dt)
        self.ref_vel = delta / dist * speed
        self.ref_pos = self.ref_pos + self.ref_vel * dt

    def max_turn_rate(self) -> float:
        """Fastest reference turn [rad/s] at which the outermost slot swings at most
        ``TURN_SPEED_SHARE`` of the catch-up speed (``omega * r_max <= share * v_catch``)."""
        r_max = self._radius()
        return HEADING_RATE if r_max < 1.0 else min(HEADING_RATE, TURN_SPEED_SHARE * self.catchup / r_max)

    def _radius(self) -> float:
        return float(np.max(np.hypot(self._offsets[:, 0], self._offsets[:, 1]))) if len(self._offsets) else 0.0

    def radius(self) -> float:
        """Horizontal radius of the current shape around the reference [m]."""
        return self._radius()

    def _turn_accel(self) -> float:
        """Angular acceleration limit [rad/s^2]: the outermost slot accelerates like the reference does."""
        return self.ref_accel / max(self._radius(), 1.0)

    def _turn_towards(self, yaw: float, dt: float) -> bool:
        """Turn towards ``yaw`` with rate and angular-acceleration limits. Returns True while turning."""
        err = wrap_angle(yaw - self.ref_yaw)
        if abs(err) < 1e-4 and abs(self._yaw_rate) < 1e-4:
            self._yaw_rate = 0.0
            return False
        alpha = self._turn_accel()
        # Rate that can still stop exactly on the target heading (braking curve), capped by the turn limit.
        wanted = math.copysign(min(self.max_turn_rate(), math.sqrt(2.0 * alpha * abs(err))), err)
        rate = float(np.clip(wanted, self._yaw_rate - alpha * dt, self._yaw_rate + alpha * dt))
        step = rate * dt
        if abs(step) > abs(err) and math.copysign(1.0, step) == math.copysign(1.0, err):
            step, rate = err, 0.0                            # land exactly on the heading
        self.ref_yaw = wrap_angle(self.ref_yaw + step)
        self._yaw_rate = rate
        return True

    def _settle_turn(self, dt: float) -> None:
        """No heading target: bleed off any residual turn rate smoothly."""
        if self._yaw_rate:
            alpha = self._turn_accel()
            rate = math.copysign(max(abs(self._yaw_rate) - alpha * dt, 0.0), self._yaw_rate)
            self.ref_yaw = wrap_angle(self.ref_yaw + rate * dt)
            self._yaw_rate = rate

    # ------------------------------------------------------------------ telemetry
    def snapshot(self) -> dict:
        return {
            "shape": str(self.shape),
            "spacing": self.spacing,
            "layers": self.layers,
            "layer_spacing": self.layer_spacing,
            "reference": self.reference_mode,
            "leader_id": self.leader_id,
            "members": list(self._members),
            "reference_position": {"x": round(float(self.ref_pos[0]), 2), "y": round(float(self.ref_pos[1]), 2),
                                   "z": round(float(self.ref_pos[2]), 2)},
            "heading": round((90.0 - math.degrees(self.ref_yaw)) % 360.0, 1),
            "heading_locked": self.heading_locked,
            "target": ({"x": round(float(self.ref_target[0]), 2), "y": round(float(self.ref_target[1]), 2),
                        "z": round(float(self.ref_target[2]), 2)} if self.ref_target is not None else None),
            "max_slot_error": round(self.max_error, 2),
            "transition_progress": round(self.transition_progress, 3),
            "transition_s": round(self._trans_duration, 2),
            "promotions": self.promotions,
            "slots": [{"x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "z": round(float(p[2]), 2)}
                      for p in self._slots_world],
        }
