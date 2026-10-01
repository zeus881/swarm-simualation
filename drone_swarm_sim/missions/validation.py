"""Pre-upload mission validation and energy estimate.

Checks every track and reports **errors** (the mission cannot run) and **warnings** (it can run, but
the operator must confirm):

* waypoints outside the world bounds (they would be clamped),
* unreachable altitudes (above the world or fence ceiling, below the minimum flight altitude),
* waypoints outside the inclusion fence or inside a no-fly zone, and legs that cross one,
* speeds above the airframe limit,
* **insufficient battery**, from the battery power model: each leg is flown at its speed (the current
  wind is added as a headwind, which is conservative), holds and loiters hover, climbs pay the
  potential-energy rate, and the return home (or landing) at the end is included. The energy must fit
  in what each assigned drone has left above the reserve (``mission.battery_reserve_percent``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from simulation.battery import BatteryModel
from simulation.config import FORMATION_SHAPES, SimConfig

from .geofence import Geofence
from .geometry import point_in_polygon, segment_crosses_polygon
from .model import Mission, MissionAction, Waypoint

if TYPE_CHECKING:
    from simulation.drone import Drone
    from simulation.environment import Environment


@dataclass
class TrackEstimate:
    distance_m: float
    duration_s: float
    energy_wh: float

    def to_dict(self) -> dict[str, float]:
        return {"distance_m": round(float(self.distance_m), 1), "duration_s": round(float(self.duration_s), 1),
                "energy_wh": round(float(self.energy_wh), 2)}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tracks: list[dict[str, Any]] = field(default_factory=list)
    battery: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "tracks": self.tracks,
                "battery": self.battery}


def estimate_track(track: Sequence[Waypoint], start: np.ndarray, home: np.ndarray, config: SimConfig,
                   wind_speed: float = 0.0) -> TrackEstimate:
    """Distance, duration and energy for one drone flying ``track`` from ``start`` [ENU m]."""
    dc, bc = config.drone, config.battery
    model = BatteryModel(bc, 100.0)
    ground = float(home[2])

    def power(air_speed: float, climb: float = 0.0) -> float:
        return model.compute_power(armed=True, motors_on=True, air_speed=air_speed, climb_rate=climb,
                                   mass_kg=dc.mass_kg, payload_kg=dc.payload_kg) * bc.drain_multiplier

    accel = 0.5 * min(dc.max_acceleration, 9.80665 * math.tan(math.radians(dc.max_tilt_deg)))
    pos = np.array(start, dtype=np.float64)
    distance = duration = energy = 0.0
    speed = dc.cruise_speed

    def fly(target: np.ndarray, v: float) -> None:
        nonlocal pos, distance, duration, energy
        d_h = float(np.hypot(*(target[:2] - pos[:2])))
        dz = float(target[2] - pos[2])
        t_h = d_h / v + (v / accel if d_h > 1.0 else 0.0)            # cruise + accelerate/brake
        t_v = dz / dc.max_climb_rate if dz > 0 else -dz / dc.max_descent_rate
        t = max(t_h, t_v, 1e-6)
        climb = max(dz, 0.0) / t
        energy += power(min(d_h / t, v) + wind_speed, climb) * t / 3600.0
        duration += t
        distance += math.hypot(d_h, dz)
        pos = target.copy()

    def hover(seconds: float, air_speed: float = 0.0) -> None:
        nonlocal duration, energy
        energy += power(air_speed + wind_speed) * seconds / 3600.0
        duration += seconds

    def land() -> None:
        nonlocal duration, energy
        t = max(pos[2] - ground, 0.0) / dc.land_speed
        energy += power(wind_speed) * t / 3600.0
        duration += t

    ended_on_ground = False
    for wp in track:
        if wp.speed is not None:
            speed = min(wp.speed, dc.max_horizontal_speed)
        target = np.array([wp.x, wp.y, ground + wp.alt])
        if wp.action == MissionAction.TAKEOFF:
            fly(np.array([pos[0], pos[1], ground + wp.alt]), speed)
        elif wp.action in (MissionAction.WAYPOINT, MissionAction.LOITER, MissionAction.LAND):
            fly(target, speed)
            if wp.action == MissionAction.LOITER:
                hover(wp.hold, min(speed, 5.0))
            elif wp.action == MissionAction.WAYPOINT:
                hover(wp.hold)
            else:
                land()
                ended_on_ground = True
                break
        elif wp.action == MissionAction.RTL:
            fly(np.array([pos[0], pos[1], max(pos[2], ground + dc.rtl_altitude)]), speed)
            fly(np.array([home[0], home[1], pos[2]]), dc.cruise_speed)
            land()
            ended_on_ground = True
            break
    if not ended_on_ground:
        # A mission that ends in the air still has to come home: budget the return and landing.
        fly(np.array([home[0], home[1], max(pos[2], ground + dc.rtl_altitude)]), dc.cruise_speed)
        land()
    return TrackEstimate(distance, duration, energy)


def validate_mission(mission: Mission, *, config: SimConfig, environment: "Environment", fence: Geofence | None,
                     assignment: list[list["Drone"]] | None = None) -> ValidationReport:
    """Validate ``mission``; ``assignment[t]`` lists the drones that will fly track ``t`` (optional)."""
    rep = ValidationReport()
    env, dc = environment, config.drone
    ground = env.ground_level
    home = np.array(env.home_position, dtype=np.float64)
    reserve_pct = config.mission.battery_reserve_percent
    wind = env.wind.speed if env.wind.enabled else 0.0
    fence_on = fence is not None and fence.defined

    obstacles = env.obstacles
    clearance = config.obstacles.clearance
    agl_mode = mission.altitude_mode == "agl"

    def world_z(wp: Waypoint) -> float:
        return (env.ground_height(wp.x, wp.y) if agl_mode else float(home[2])) + wp.alt

    for t, track in enumerate(mission.tracks):
        label = f"track {t + 1}, " if len(mission.tracks) > 1 else ""
        prev: Waypoint | None = None
        for k, wp in enumerate(track):
            where = f"{label}waypoint {k + 1} ({wp.action})"
            if wp.action == MissionAction.CHANGE_FORMATION:
                shape = str(wp.params.get("shape", "")).lower()
                if shape not in FORMATION_SHAPES:
                    rep.errors.append(f"{where}: unknown formation shape '{shape}'")
                if assignment is not None and len(assignment[t]) < 2:
                    rep.warnings.append(f"{where}: a single drone cannot change formation (ignored)")
                continue
            if wp.action == MissionAction.RTL:
                continue
            if wp.speed is not None and wp.speed > dc.max_horizontal_speed:
                rep.warnings.append(f"{where}: speed {wp.speed:.1f} m/s exceeds the airframe limit "
                                    f"{dc.max_horizontal_speed:.1f} m/s (will be limited)")
            if wp.action != MissionAction.LAND:
                if wp.alt > env.max_altitude:
                    rep.warnings.append(f"{where}: altitude {wp.alt:.0f} m is unreachable (ceiling {env.max_altitude:.0f} m)")
                elif wp.alt < dc.min_altitude:
                    rep.warnings.append(f"{where}: altitude {wp.alt:.1f} m is below the minimum flight altitude "
                                        f"{dc.min_altitude:.1f} m")
                if fence_on and fence.max_altitude is not None and wp.alt > fence.max_altitude:
                    rep.warnings.append(f"{where}: altitude {wp.alt:.0f} m is above the geofence ceiling "
                                        f"{fence.max_altitude:.0f} m")
            if wp.action == MissionAction.TAKEOFF:
                prev = wp if prev is None else prev
                continue
            if not (env.bounds_min[0] <= wp.x <= env.bounds_max[0] and env.bounds_min[1] <= wp.y <= env.bounds_max[1]):
                rep.warnings.append(f"{where}: ({wp.x:.0f}, {wp.y:.0f}) is outside the world bounds (will be clamped)")
            if fence_on:
                if fence.inclusion is not None and not point_in_polygon((wp.x, wp.y), fence.inclusion):
                    rep.warnings.append(f"{where}: outside the inclusion geofence")
                for zone in fence.exclusions:
                    if point_in_polygon((wp.x, wp.y), zone.polygon):
                        rep.warnings.append(f"{where}: inside no-fly zone '{zone.name}'")
                if prev is not None and prev.action != MissionAction.TAKEOFF:
                    for zone in fence.exclusions:
                        if not point_in_polygon((wp.x, wp.y), zone.polygon) \
                                and segment_crosses_polygon((prev.x, prev.y), (wp.x, wp.y), zone.polygon):
                            rep.warnings.append(f"{where}: leg from waypoint {k} crosses no-fly zone '{zone.name}'")
            z = world_z(wp)
            if not agl_mode and wp.action != MissionAction.LAND and env.terrain is not None:
                ground_here = env.ground_height(wp.x, wp.y)
                if z < ground_here + config.obstacles.terrain_clearance:
                    rep.warnings.append(f"{where}: {wp.alt:.0f} m above home is below the terrain there "
                                        f"({ground_here - home[2]:.0f} m) - use altitude_mode 'agl' or climb")
            if len(obstacles) and wp.action != MissionAction.LAND:
                d, _, which = obstacles.nearest(np.array([[wp.x, wp.y, z]]), clearance + 1.0)
                if which[0] >= 0 and d[0] < clearance:
                    rep.warnings.append(f"{where}: within {max(d[0], 0):.0f} m of obstacle "
                                        f"'{obstacles.obstacles[which[0]].name}' (clearance {clearance:.0f} m)")
                if prev is not None and prev.action != MissionAction.TAKEOFF and prev.positional:
                    a = np.array([prev.x, prev.y, world_z(prev)])
                    if not obstacles.segment_clear(a, np.array([wp.x, wp.y, z]), clearance):
                        rep.warnings.append(f"{where}: leg from waypoint {k} passes an obstacle"
                                            + (" - it will be routed around" if config.obstacles.plan_paths else ""))
            prev = wp

        drones = assignment[t] if assignment is not None else []
        starts = [d.position for d in drones] or [home]
        estimates = [estimate_track(track, np.asarray(s), home, config, wind) for s in starts]
        worst = max(estimates, key=lambda e: e.energy_wh)
        rep.tracks.append({"track": t, **worst.to_dict(), "waypoints": len(track)})
        for d, est in zip(drones, estimates):
            available = d.battery.energy_wh - d.battery.capacity_wh * reserve_pct / 100.0
            ok = est.energy_wh <= available
            rep.battery.append({"drone_id": d.id, "track": t, "needed_wh": round(float(est.energy_wh), 2),
                                "available_wh": round(float(max(available, 0.0)), 2),
                                "battery": round(float(d.battery.percent), 1), "ok": bool(ok)})
            if not ok:
                rep.warnings.append(
                    f"{d.name}: insufficient battery - needs {est.energy_wh:.1f} Wh, has {max(available, 0.0):.1f} Wh "
                    f"above the {reserve_pct:.0f}% reserve ({d.battery.percent:.0f}% charge)")
    return rep
