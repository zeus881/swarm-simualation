"""Mission manager: named drone groups, the geofence, and mission execution.

A started mission becomes a :class:`MissionRun` made of :class:`TrackRun` executors:

* **formation track**: several drones fly one waypoint list *in formation*. The executor moves the
  formation reference (virtual) or the leader (leader-follower) from waypoint to waypoint;
  CHANGE_FORMATION items re-shape the group on the way.
* **single track**: one drone flies its own list (a single drone, or one strip of a split survey).

Executors command drones only through :class:`~simulation.drone_interface.DroneInterface` and the
swarm coordinator, exactly like the operator does. Operator authority is kept: a manual command
addressed to a drone (hover, goto, RTL, ...) or a failsafe/geofence action takes that drone out of
its mission, and a mission without drones is aborted. Everything is logged as MISSION events.

Commands registered here: ``define_group``, ``delete_group``, ``set_geofence``, ``clear_geofence``,
``mission_validate``, ``mission_start``, ``mission_pause``, ``mission_resume``, ``mission_abort``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from simulation.commands import Command, CommandError, CommandProcessor
from simulation.config import ConfigError
from simulation.drone_interface import CommandResult
from simulation.events import EventCategory, Severity, SimEvent
from simulation.types import CommStatus, FlightMode

from .geofence import Geofence, GeofenceMonitor
from .geometry import PolygonError
from .model import Mission, MissionAction, MissionError, Waypoint
from .validation import ValidationReport, validate_mission

if TYPE_CHECKING:
    from simulation.drone import Drone
    from simulation.engine import SimulationEngine

# Operator commands that take the addressed drones out of their mission.
MANUAL_COMMANDS = frozenset({
    "arm", "disarm", "takeoff", "land", "hover", "goto", "set_velocity", "set_altitude", "return_to_home",
    "emergency_stop", "set_formation", "start_flocking", "release_swarm", "swarm_goto", "remove_drone",
})
GROUP_NAME_MAX = 24


class RunState(StrEnum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"


class Phase(StrEnum):
    ENTER = "ENTER"            # issue the commands for the current waypoint
    TAKEOFF = "TAKEOFF"        # climbing to the takeoff altitude
    FORMING = "FORMING"        # formation assembling / re-shaping
    TRANSIT = "TRANSIT"        # flying to the waypoint
    HOLD = "HOLD"              # holding at the waypoint
    LOITER = "LOITER"          # orbiting the waypoint
    LANDING = "LANDING"        # LAND / RTL until everyone is on the ground
    DONE = "DONE"


@dataclass
class TrackRun:
    """Executes one waypoint list for one drone (single) or a group of drones (formation)."""

    manager: "MissionManager"
    drone_ids: list[int]
    waypoints: list[Waypoint]
    formation: bool
    index: int = 0
    phase: Phase = Phase.ENTER
    hold_until: float = 0.0
    loiter_angle: float = 0.0
    loiter_until: float = 0.0
    released: list[int] = field(default_factory=list)
    goal: np.ndarray | None = None             # target of the current leg (fixed when the leg starts)
    agl: bool = False                          # altitude_mode "agl": heights above the terrain under the drone
    _takeoff_alt: float = 0.0

    # ------------------------------------------------------------------ helpers
    @property
    def engine(self) -> "SimulationEngine":
        return self.manager.engine

    @property
    def done(self) -> bool:
        return self.phase == Phase.DONE

    @property
    def current(self) -> Waypoint | None:
        return self.waypoints[self.index] if self.index < len(self.waypoints) else None

    def drones(self) -> list["Drone"]:
        swarm = self.engine.swarm
        return [swarm.get(i) for i in self.drone_ids if i in swarm]

    def target(self, wp: Waypoint) -> np.ndarray:
        """Waypoint position: ``alt`` above home (relative mode) or above the terrain there (AGL mode)."""
        env = self.engine.environment
        ground = env.ground_height(wp.x, wp.y) if self.agl else float(env.home_position[2])
        return np.array([wp.x, wp.y, ground + wp.alt])

    def route_to(self, start: np.ndarray, goal: np.ndarray, wp: Waypoint, who: str) -> list[np.ndarray]:
        """Obstacle-free (and, in AGL mode, terrain-following) pass-through points to ``goal``."""
        agl = wp.alt if self.agl and wp.action != MissionAction.LAND else None
        return self.engine.coordinator.route(start, goal, agl=agl, who=who)

    def leg_target(self, wp: Waypoint, current_z: float) -> np.ndarray:
        """Where the leg to ``wp`` ends. A LAND point is approached at the current altitude and then
        descended to vertically (as NAV_LAND does); every other waypoint uses its own altitude."""
        t = self.target(wp)
        if wp.action == MissionAction.LAND:
            ground = self.engine.environment.ground_height(wp.x, wp.y)
            t[2] = max(current_z, ground + self.engine.config.drone.min_altitude)
        return t

    def release(self, drone_id: int, reason: str) -> None:
        if drone_id in self.drone_ids:
            self.drone_ids.remove(drone_id)
            self.released.append(drone_id)
            self.manager.log(f"D{drone_id:02d} left the mission ({reason})", Severity.WARNING, drone_id=drone_id)
        if not self.drone_ids and not self.done:
            self.phase = Phase.DONE

    def advance(self, t: float) -> None:
        self.index += 1
        self.phase = Phase.ENTER if self.index < len(self.waypoints) else Phase.DONE

    def _speed(self, wp: Waypoint) -> float:
        dc = self.engine.config.drone
        return min(wp.speed if wp.speed is not None else self.engine.config.mission.default_speed,
                   dc.max_horizontal_speed)

    # ------------------------------------------------------------------ supervision
    def supervise(self) -> None:
        """Drop drones that failed, vanished or were taken over by a failsafe or another controller."""
        swarm = self.engine.swarm
        wp = self.current
        landing_ok = wp is not None and wp.action in (MissionAction.LAND, MissionAction.RTL)
        coord = self.engine.coordinator
        for i in list(self.drone_ids):
            if i not in swarm:
                self.release(i, "removed")
                continue
            d = swarm.get(i)
            if d.failed:
                self.release(i, f"failed: {d.failure_reason}")
            elif d.comm_status == CommStatus.LOST:
                self.release(i, "link lost")
            elif d.flight_mode == FlightMode.EMERGENCY:
                self.release(i, "emergency")
            elif d.flight_mode in (FlightMode.RTL, FlightMode.LAND) and not (landing_ok and self.phase == Phase.LANDING):
                self.release(i, f"{d.flight_mode} by failsafe or operator")
            elif (self.formation and self.phase in (Phase.TRANSIT, Phase.HOLD, Phase.LOITER, Phase.FORMING)
                  and d.flight_mode != FlightMode.FORMATION and d.id != coord.formation.leader_id):
                self.release(i, f"left the formation ({d.flight_mode})")
            elif (not self.formation and self.phase == Phase.TRANSIT
                  and d.flight_mode not in (FlightMode.GOTO, FlightMode.HOVER)):
                self.release(i, f"taken over ({d.flight_mode})")

    # ------------------------------------------------------------------ execution
    def step(self, t: float) -> None:
        if self.done:
            return
        self.supervise()
        if self.done:
            return
        wp = self.current
        if wp is None:
            self.phase = Phase.DONE
            return
        if self.formation:
            self._step_formation(t, wp)
        else:
            self._step_single(t, wp)

    # --- single drone
    def _step_single(self, t: float, wp: Waypoint) -> None:
        d = self.drones()[0]
        cfg = self.engine.config
        if self.phase == Phase.ENTER:
            if wp.action == MissionAction.TAKEOFF:
                self.phase = Phase.TAKEOFF
                if d.in_flight:
                    d.goto((d.position[0], d.position[1], self.target(wp)[2]))
                else:
                    self._check(d.takeoff(wp.alt), d)
            elif wp.action == MissionAction.RTL:
                self.phase = Phase.LANDING
                self._check(d.return_to_home(), d)
            elif wp.action == MissionAction.CHANGE_FORMATION:
                self.advance(t)                           # meaningless for one drone (validation warns)
            else:
                if not d.in_flight:                       # implicit takeoff before the first leg
                    self.phase = Phase.TAKEOFF
                    self._check(d.takeoff(max(wp.alt, cfg.drone.min_altitude + 1.0)), d)
                    return
                self.phase = Phase.TRANSIT
                self.goal = self.leg_target(wp, float(d.position[2]))
                route = self.route_to(d.position, self.goal, wp, who=f"mission leg {self.index + 1} {d.name}")
                self.goal = route[-1]
                self._check(d.goto_path(route, speed=self._speed(wp)), d)
            return
        if self.phase == Phase.TAKEOFF:
            if d.flight_mode == FlightMode.HOVER and d.in_flight:
                if wp.action == MissionAction.TAKEOFF:
                    self.advance(t)
                else:
                    self.phase = Phase.ENTER              # takeoff inserted before a positional waypoint
            return
        if self.phase == Phase.TRANSIT:
            if d.flight_mode == FlightMode.HOVER and np.linalg.norm(d.position - self.goal) <= cfg.mission.arrival_radius:
                self._arrived(t, wp, [d])
            return
        if self.phase == Phase.HOLD:
            if t >= self.hold_until:
                self.advance(t)
            return
        if self.phase == Phase.LOITER:
            if t >= self.loiter_until:
                self._check(d.goto(self.target(wp), speed=self._speed(wp)), d)
                self.advance(t)
            else:
                d.goto(self._orbit_point(t, wp), speed=min(self._speed(wp), 6.0))
            return
        if self.phase == Phase.LANDING:
            if not d.in_flight:
                self.advance(t)

    # --- formation group
    def _step_formation(self, t: float, wp: Waypoint) -> None:
        coord = self.engine.coordinator
        form = coord.formation
        drones = self.drones()
        cfg = self.engine.config
        if self.phase == Phase.ENTER:
            grounded = [d for d in drones if not d.in_flight]
            if wp.action == MissionAction.TAKEOFF or (grounded and wp.action not in (MissionAction.RTL,)):
                alt = wp.alt if wp.action == MissionAction.TAKEOFF else max(wp.alt, cfg.drone.min_altitude + 1.0)
                for d in grounded:
                    self._check(d.takeoff(alt), d)
                self.phase = Phase.TAKEOFF
                self._takeoff_alt = alt
                return
            if wp.action == MissionAction.RTL:
                self._leave_formation()
                for d in drones:
                    self._check(d.return_to_home(), d)
                self.phase = Phase.LANDING
                return
            if not self._formation_ready():
                self._form()
                self.phase = Phase.FORMING
                return
            if wp.action == MissionAction.CHANGE_FORMATION:
                params = {"shape": wp.params.get("shape")}
                if "spacing" in wp.params:
                    params["spacing"] = wp.params["spacing"]
                self.manager.internal_command("set_formation", self.group_ids(), params)
                self.phase = Phase.FORMING
                return
            self.goal = self.leg_target(wp, float(form.ref_pos[2]))
            agl = wp.alt if self.agl and wp.action != MissionAction.LAND else None
            if form.reference_mode == "leader" and form.leader_id is not None and form.leader_id in self.engine.swarm:
                route = self.engine.coordinator.route(self.engine.swarm.get(form.leader_id).position, self.goal,
                                                      agl=agl, who=f"mission leg {self.index + 1} leader")
            else:
                route = self.engine.coordinator._formation_route(self.goal, agl)
            self.goal = route[-1]
            self.manager.move_group(self.goal, self._speed(wp), route=route)
            self.phase = Phase.TRANSIT
            return
        if self.phase == Phase.TAKEOFF:
            if all(d.in_flight and d.flight_mode != FlightMode.TAKEOFF for d in drones):
                if wp.action == MissionAction.TAKEOFF:
                    self._form(altitude=self._takeoff_alt)
                    self.phase = Phase.FORMING
                else:
                    self.phase = Phase.ENTER          # implicit takeoff done: now form up and fly the waypoint
            return
        if self.phase == Phase.FORMING:
            if form.active and form.transition_progress >= 1.0 and form.max_error <= cfg.mission.formation_arrival_error:
                if wp.action in (MissionAction.CHANGE_FORMATION, MissionAction.TAKEOFF):
                    self.advance(t)
                else:
                    self.phase = Phase.ENTER
            elif not form.active:
                self.phase = Phase.ENTER
            return
        if self.phase == Phase.TRANSIT:
            if self._group_arrived(self.goal):
                self._arrived(t, wp, drones)
            return
        if self.phase == Phase.HOLD:
            if t >= self.hold_until:
                self.advance(t)
            return
        if self.phase == Phase.LOITER:
            if t >= self.loiter_until:
                self.manager.move_group(self.target(wp), self._speed(wp))
                self.advance(t)
            else:
                self.manager.move_group(self._orbit_point(t, wp), min(self._speed(wp), 6.0), quiet=True)
            return
        if self.phase == Phase.LANDING:
            if all(not d.in_flight for d in drones):
                self.advance(t)

    def group_ids(self) -> list[int]:
        return list(self.drone_ids)

    def _formation_ready(self) -> bool:
        form = self.engine.coordinator.formation
        if not form.active:
            return False
        members = set(form.snapshot()["members"]) | ({form.leader_id} if form.leader_id is not None else set())
        return set(self.drone_ids) <= members

    def _form(self, altitude: float | None = None) -> None:
        coord = self.engine.coordinator
        shape = str(coord.formation.shape) if coord.formation.active else self.engine.config.mission.default_formation
        params: dict[str, Any] = {"shape": shape}
        if altitude is not None:
            params["altitude"] = altitude
        self.manager.internal_command("set_formation", self.group_ids(), params)

    def _leave_formation(self) -> None:
        if self.engine.coordinator.formation.active:
            self.manager.internal_command("release_swarm", self.group_ids(), {})

    def _group_arrived(self, target: np.ndarray) -> bool:
        form = self.engine.coordinator.formation
        tol = self.engine.config.mission.formation_arrival_error
        if not form.active:
            return True
        if form.reference_mode == "leader" and form.leader_id is not None:
            leader = self.engine.swarm.get(form.leader_id)
            near = np.linalg.norm(leader.position - target) <= self.engine.config.mission.arrival_radius
            return near and leader.flight_mode == FlightMode.HOVER and form.max_error <= tol
        return form.ref_target is None and float(np.linalg.norm(form.ref_pos - target)) < 1.0 and form.max_error <= tol

    def _arrived(self, t: float, wp: Waypoint, drones: list["Drone"]) -> None:
        self.manager.log(f"reached waypoint {self.index + 1}/{len(self.waypoints)} ({wp.action})"
                         + (f" with {len(drones)} drones" if self.formation else f" - {drones[0].name}"),
                         drone_id=drones[0].id if not self.formation and drones else None)
        if wp.action == MissionAction.LOITER:
            self.phase = Phase.LOITER
            self.loiter_until = t + max(wp.hold, 0.0)
            ref = self.engine.coordinator.formation.ref_pos if self.formation else drones[0].position
            self.loiter_angle = math.atan2(ref[1] - wp.y, ref[0] - wp.x) if np.hypot(ref[0] - wp.x, ref[1] - wp.y) > 1 else 0.0
        elif wp.action == MissionAction.LAND:
            if self.formation:
                self._leave_formation()
            for d in drones:
                self._check(d.land(), d)
            self.phase = Phase.LANDING
        else:
            self.phase = Phase.HOLD
            self.hold_until = t + wp.hold

    def _orbit_point(self, t: float, wp: Waypoint) -> np.ndarray:
        radius = float(wp.params.get("radius", self.engine.config.mission.loiter_radius))
        speed = min(self._speed(wp), 6.0)
        self.loiter_angle += speed / max(radius, 1.0) * self.engine.dt   # counter-clockwise orbit
        lead = self.loiter_angle + 0.5 * speed / max(radius, 1.0)         # aim slightly ahead on the circle
        centre = self.target(wp)
        return centre + np.array([radius * math.cos(lead), radius * math.sin(lead), 0.0])

    def _check(self, result: CommandResult, drone: "Drone") -> None:
        if not result.success:
            self.release(drone.id, result.message)

    def snapshot(self) -> dict[str, Any]:
        wp = self.current
        return {"drones": list(self.drone_ids), "formation": self.formation, "index": self.index,
                "total": len(self.waypoints), "phase": str(self.phase),
                "action": str(wp.action) if wp is not None else None,
                "target": ({"x": round(wp.x, 1), "y": round(wp.y, 1), "alt": round(wp.alt, 1)} if wp is not None else None),
                "released": list(self.released)}


@dataclass
class MissionRun:
    id: int
    mission: Mission
    tracks: list[TrackRun]
    started: float
    state: RunState = RunState.RUNNING
    finished: float | None = None

    @property
    def drone_ids(self) -> set[int]:
        return {i for tr in self.tracks for i in tr.drone_ids}

    @property
    def active(self) -> bool:
        return self.state in (RunState.RUNNING, RunState.PAUSED)

    def snapshot(self) -> dict[str, Any]:
        done = sum(min(tr.index, len(tr.waypoints)) for tr in self.tracks)
        total = sum(len(tr.waypoints) for tr in self.tracks)
        return {"id": self.id, "name": self.mission.name, "state": str(self.state),
                "progress": round(done / total, 3) if total else 1.0, "started": round(self.started, 2),
                "finished": round(self.finished, 2) if self.finished is not None else None,
                "tracks": [tr.snapshot() for tr in self.tracks]}


class MissionManager:
    def __init__(self, engine: "SimulationEngine") -> None:
        self.engine = engine
        cfg = engine.config
        gc = cfg.geofence
        try:
            fence = Geofence.from_dict({"enabled": gc.enabled, "action": gc.action,
                                        "inclusion": gc.inclusion or None,
                                        "exclusions": [{"polygon": z} for z in gc.exclusions],
                                        "max_altitude": gc.max_altitude}, lookahead_s=gc.lookahead_s)
        except PolygonError as exc:
            raise ConfigError(f"invalid geofence in configuration: {exc}") from exc
        self.geofence = GeofenceMonitor(fence, engine.events, clear_time=gc.clear_time)
        self.groups: dict[str, list[int]] = {}
        self.runs: list[MissionRun] = []
        self._ids = itertools.count(1)
        self.version = 0                       # bumped when active mission paths change (clients re-fetch)
        self._internal = 0
        engine.events.subscribe(self._on_event)
        self._register(engine.commands)

    # ------------------------------------------------------------------ helpers
    def log(self, message: str, severity: Severity = Severity.INFO, **data: Any) -> None:
        self.engine.events.emit(EventCategory.MISSION, "mission", message, severity=severity, **data)

    def geo_to_enu(self, lat: float, lon: float) -> tuple[float, float]:
        geo = self.engine.geo
        e = geo.geodetic_to_enu(lat, lon, geo.origin.altitude)
        return float(e[0]), float(e[1])

    def enu_to_geo(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        lat, lon, alt = self.engine.geo.enu_to_geodetic(np.array([x, y, z]))
        return float(lat), float(lon), float(alt)

    def parse_mission(self, data: Any) -> Mission:
        if not isinstance(data, dict):
            raise CommandError("parameter 'mission' must be a mission object")
        try:
            return Mission.from_dict(data, self.geo_to_enu)
        except MissionError as exc:
            raise CommandError(f"invalid mission: {exc}") from None

    def internal_command(self, ctype: str, drone_ids: list[int] | None, params: dict[str, Any]) -> CommandResult:
        """Run a command on behalf of a mission: validated and logged, but it does not release mission drones."""
        self._internal += 1
        try:
            return self.engine.execute({"type": ctype, "drone_ids": drone_ids, "params": params})
        finally:
            self._internal -= 1

    def move_group(self, target: np.ndarray, speed: float, quiet: bool = False,
                   route: list[np.ndarray] | None = None) -> None:
        """Move the formation (virtual reference, or the leader in leader-follower mode), optionally
        through pass-through ``route`` points (last = target)."""
        form = self.engine.coordinator.formation
        env, min_agl = self.engine.environment, self.engine.config.drone.min_altitude
        via = [env.clamp(p, min_agl=min_agl) for p in (route or [target])[:-1]]
        if form.reference_mode == "leader" and form.leader_id is not None and form.leader_id in self.engine.swarm:
            self.engine.swarm.get(form.leader_id).goto_path([*via, target], speed=speed)
        else:
            form.move_to(env.clamp(target, min_agl=min_agl), speed, route=via)

    def resolve_drones(self, cmd: Command) -> list["Drone"]:
        group = cmd.params.get("group")
        if group is not None:
            if group not in self.groups:
                raise CommandError(f"unknown group '{group}'")
            ids = [i for i in self.groups[group] if i in self.engine.swarm]
            if not ids:
                raise CommandError(f"group '{group}' has no drones left")
            return [self.engine.swarm.get(i) for i in ids]
        drones = self.engine.swarm.select(cmd.drone_ids)
        if not drones:
            raise CommandError("no drones selected")
        return drones

    def _assign(self, mission: Mission, drones: list["Drone"], formation: bool) -> list[list["Drone"]]:
        """Drones per track. Multi-track missions use the Hungarian algorithm on distance to each track start."""
        usable = [d for d in drones if not d.failed]
        if not usable:
            raise CommandError("no usable drones (all failed)")
        if len(mission.tracks) == 1:
            if len(usable) > 1 and not formation:
                raise CommandError("several drones cannot fly one waypoint list without a formation "
                                   "(enable formation, or split the mission into tracks)")
            return [usable]
        if len(usable) < len(mission.tracks):
            raise CommandError(f"mission has {len(mission.tracks)} tracks but only {len(usable)} drones were given")
        from algorithms.formation import assign_slots
        starts = []
        for track in mission.tracks:
            first = next((w for w in track if w.positional), track[0])
            starts.append([first.x, first.y, 0.0])
        pos = np.array([[d.position[0], d.position[1], 0.0] for d in usable])
        # Assign tracks to drones: every track gets exactly one drone (drones may be left over).
        slot_of = assign_slots(np.array(starts), pos)
        return [[usable[int(slot_of[t])]] for t in range(len(mission.tracks))]

    # ------------------------------------------------------------------ tick
    def step(self, t: float) -> None:
        swarm = self.engine.swarm
        drones = swarm.drones
        if drones and self.geofence.fence.enabled:
            positions = np.array([d.position for d in drones])
            velocities = np.array([d.velocity for d in drones])
            self.geofence.step(drones, positions, velocities, t, self.engine.environment.ground_level)
        for run in self.runs:
            if run.state != RunState.RUNNING:
                continue
            for tr in run.tracks:
                tr.step(t)
            if all(tr.done for tr in run.tracks):
                flown = any(tr.drone_ids for tr in run.tracks)
                run.state = RunState.COMPLETED if flown else RunState.ABORTED
                run.finished = t
                self.version += 1
                self.log(f"mission '{run.mission.name}' {'completed' if flown else 'aborted: no drones left'}",
                         Severity.INFO if flown else Severity.WARNING, run_id=run.id)
        # Keep a short history of finished runs only (bounded memory over long sessions).
        finished = [r for r in self.runs if not r.active]
        if len(finished) > 10:
            drop = {r.id for r in finished[:-10]}
            self.runs = [r for r in self.runs if r.id not in drop]

    def _on_event(self, event: SimEvent) -> None:
        """Operator commands take the addressed drones out of their missions."""
        if self._internal or event.category != EventCategory.COMMAND or event.kind not in MANUAL_COMMANDS:
            return
        if not event.data.get("success"):
            return
        ids = event.data.get("drone_ids")
        for run in self.runs:
            if not run.active:
                continue
            for tr in run.tracks:
                for i in list(tr.drone_ids):
                    if ids is None or i in ids:
                        tr.release(i, f"operator command '{event.kind}'")

    # ------------------------------------------------------------------ commands
    def _register(self, commands: CommandProcessor) -> None:
        for name, handler, doc in (
            ("define_group", self._define_group, "Save drone_ids as a named group {name}"),
            ("delete_group", self._delete_group, "Delete a named group {name}"),
            ("set_geofence", self._set_geofence,
             "Geofence {enabled?, action? RTL|LAND|HOLD, inclusion? [[x,y],..], exclusions? [{name, polygon}], max_altitude?}"),
            ("clear_geofence", self._clear_geofence, "Remove every fence polygon"),
            ("mission_validate", self._validate, "Check a mission before upload {mission, group?}"),
            ("mission_start", self._start,
             "Upload and fly a mission {mission, group?, formation?, force?, apply_geofence?}"),
            ("mission_pause", self._pause, "Pause missions {run_id?} (drones hold position)"),
            ("mission_resume", self._resume, "Resume paused missions {run_id?}"),
            ("mission_abort", self._abort, "Abort missions {run_id?} (drones hold position)"),
        ):
            commands.register(name, handler, doc)

    def _define_group(self, cmd: Command) -> CommandResult:
        name = cmd.params.get("name")
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > GROUP_NAME_MAX:
            raise CommandError(f"group name must be 1..{GROUP_NAME_MAX} characters")
        if not cmd.drone_ids:
            raise CommandError("define_group needs explicit drone_ids")
        for i in cmd.drone_ids:
            self.engine.swarm.get(i)                      # KeyError -> "unknown drone id"
        self.groups[name.strip()] = sorted(set(cmd.drone_ids))
        return CommandResult.ok(f"group {name.strip()}: {', '.join(f'D{i:02d}' for i in self.groups[name.strip()])}")

    def _delete_group(self, cmd: Command) -> CommandResult:
        name = cmd.params.get("name")
        if name not in self.groups:
            raise CommandError(f"unknown group '{name}'")
        del self.groups[name]
        return CommandResult.ok(f"group {name} deleted")

    def _set_geofence(self, cmd: Command) -> CommandResult:
        current = self.geofence.fence.to_dict()
        merged = {**current, **cmd.params}
        try:
            fence = Geofence.from_dict(merged, lookahead_s=self.engine.config.geofence.lookahead_s)
        except PolygonError as exc:
            raise CommandError(str(exc)) from None
        env = self.engine.environment
        for poly, label in ([(fence.inclusion, "inclusion")] if fence.inclusion is not None else []) \
                + [(z.polygon, z.name) for z in fence.exclusions]:
            if np.any(poly < env.bounds_min) or np.any(poly > env.bounds_max):
                raise CommandError(f"geofence {label} extends outside the world bounds")
        self.geofence.set_fence(fence)
        if fence.enabled and fence.inclusion is not None and not bool(
                np.all([self._inside_inclusion(d.home_position, fence) for d in self.engine.swarm])):
            self.log("warning: some home pads lie outside the inclusion fence", Severity.WARNING)
        parts = [f"{'ENABLED' if fence.enabled else 'DISABLED'}", f"action {fence.action}"]
        if fence.inclusion is not None:
            parts.append("inclusion polygon")
        if fence.exclusions:
            parts.append(f"{len(fence.exclusions)} no-fly zone{'s' if len(fence.exclusions) > 1 else ''}")
        if fence.max_altitude is not None:
            parts.append(f"ceiling {fence.max_altitude:.0f} m")
        return CommandResult.ok("geofence " + ", ".join(parts), version=self.geofence.version)

    @staticmethod
    def _inside_inclusion(point: np.ndarray, fence: Geofence) -> bool:
        from .geometry import point_in_polygon
        return fence.inclusion is None or point_in_polygon(point, fence.inclusion)

    def _clear_geofence(self, cmd: Command) -> CommandResult:
        fence = self.geofence.fence
        self.geofence.set_fence(Geofence(False, fence.action, None, [], None, fence.lookahead_s))
        return CommandResult.ok("geofence cleared")

    def validate(self, mission: Mission, drones: list["Drone"] | None, formation: bool = True) -> ValidationReport:
        assignment = self._assign(mission, drones, formation) if drones else None
        return validate_mission(mission, config=self.engine.config, environment=self.engine.environment,
                                fence=self.geofence.fence if self.geofence.fence.defined else None, assignment=assignment)

    def _formation_flag(self, cmd: Command, mission: Mission, drones: list["Drone"]) -> bool:
        flag = cmd.params.get("formation")
        if flag is not None and not isinstance(flag, bool):
            raise CommandError("parameter 'formation' must be true/false")
        return (len(drones) > 1 and len(mission.tracks) == 1) if flag is None else flag

    def _validate(self, cmd: Command) -> CommandResult:
        mission = self.parse_mission(cmd.params.get("mission"))
        drones = self.resolve_drones(cmd)
        formation = self._formation_flag(cmd, mission, drones)
        report = self.validate(mission, drones, formation)
        summary = "OK" if report.ok and not report.warnings else \
            f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
        return CommandResult(report.ok, f"mission '{mission.name}': {summary}", data=report.to_dict())

    def _start(self, cmd: Command) -> CommandResult:
        p = cmd.params
        mission = self.parse_mission(p.get("mission"))
        drones = self.resolve_drones(cmd)
        formation = self._formation_flag(cmd, mission, drones)
        force = p.get("force", False)
        if not isinstance(force, bool):
            raise CommandError("parameter 'force' must be true/false")
        assignment = self._assign(mission, drones, formation)
        report = validate_mission(mission, config=self.engine.config, environment=self.engine.environment,
                                  fence=self.geofence.fence if self.geofence.fence.defined else None,
                                  assignment=assignment)
        if not report.ok:
            return CommandResult(False, f"mission rejected: {report.errors[0]}", data=report.to_dict())
        if report.warnings and not force:
            return CommandResult(False, f"mission has {len(report.warnings)} warning(s) - confirm to fly anyway "
                                        f"(first: {report.warnings[0]})", data={**report.to_dict(), "needs_confirmation": True})
        busy_formation = any(tr.formation for r in self.runs if r.active for tr in r.tracks)
        if formation and busy_formation:
            raise CommandError("a formation mission is already running - abort it first")
        if p.get("apply_geofence") and mission.geofence:
            fence_result = self._set_geofence(Command("set_geofence", None, dict(mission.geofence)))
            self.log(f"mission geofence applied: {fence_result.message}")
        # A drone can fly only one mission: take the assigned drones out of any other run.
        chosen = {d.id for group in assignment for d in group}
        for run in self.runs:
            if run.active:
                for tr in run.tracks:
                    for i in list(tr.drone_ids):
                        if i in chosen:
                            tr.release(i, "reassigned to a new mission")
        run = MissionRun(next(self._ids), mission,
                         [TrackRun(self, [d.id for d in group], list(track), formation and len(group) > 1,
                                   agl=mission.altitude_mode == "agl")
                          for group, track in zip(assignment, mission.tracks)],
                         started=self.engine.sim_time)
        self.runs.append(run)
        self.version += 1
        self.log(f"mission '{mission.name}' started (run {run.id}): {len(chosen)} drone(s), "
                 f"{mission.waypoint_count} waypoints, {len(mission.tracks)} track(s)"
                 + (" in formation" if formation and len(chosen) > 1 else ""), run_id=run.id)
        return CommandResult(True, f"mission '{mission.name}' started with {len(chosen)} drone(s)",
                             data={"run_id": run.id, **report.to_dict()})

    def _select_runs(self, cmd: Command, states: Iterable[RunState]) -> list[MissionRun]:
        run_id = cmd.params.get("run_id")
        runs = [r for r in self.runs if r.state in set(states) and (run_id is None or r.id == run_id)]
        if not runs:
            raise CommandError("no matching mission" if run_id is None else f"no matching mission run {run_id}")
        return runs

    def _hold(self, run: MissionRun) -> None:
        form = self.engine.coordinator.formation
        for tr in run.tracks:
            if tr.formation and form.active:
                if form.reference_mode == "leader" and form.leader_id in self.engine.swarm:
                    self.engine.swarm.get(form.leader_id).hover()
                else:
                    form.ref_target = None
            else:
                for d in tr.drones():
                    if d.in_flight and d.flight_mode in (FlightMode.GOTO, FlightMode.HOVER):
                        d.hover()

    def _pause(self, cmd: Command) -> CommandResult:
        runs = self._select_runs(cmd, [RunState.RUNNING])
        for run in runs:
            run.state = RunState.PAUSED
            self._hold(run)
            for tr in run.tracks:
                if tr.phase in (Phase.TRANSIT, Phase.LOITER, Phase.HOLD):
                    tr.phase = Phase.ENTER                # resume re-issues the current waypoint
        return CommandResult.ok(f"paused {len(runs)} mission(s)")

    def _resume(self, cmd: Command) -> CommandResult:
        runs = self._select_runs(cmd, [RunState.PAUSED])
        for run in runs:
            run.state = RunState.RUNNING
        return CommandResult.ok(f"resumed {len(runs)} mission(s)")

    def _abort(self, cmd: Command) -> CommandResult:
        runs = self._select_runs(cmd, [RunState.RUNNING, RunState.PAUSED])
        for run in runs:
            self._hold(run)
            run.state = RunState.ABORTED
            run.finished = self.engine.sim_time
        self.version += 1
        return CommandResult.ok(f"aborted {len(runs)} mission(s); drones hold position")

    # ------------------------------------------------------------------ telemetry
    def active_paths(self) -> list[dict[str, Any]]:
        """Waypoint paths of running missions (fetched by clients when ``version`` changes)."""
        out = []
        for run in self.runs:
            if not run.active:
                continue
            for t, tr in enumerate(run.tracks):
                out.append({"run_id": run.id, "track": t, "drones": list(tr.drone_ids), "formation": tr.formation,
                            "waypoints": [w.to_dict() for w in tr.waypoints]})
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "runs": [r.snapshot() for r in self.runs[-6:]],
            "groups": {k: list(v) for k, v in self.groups.items()},
            "geofence": self.geofence.snapshot(),
        }
