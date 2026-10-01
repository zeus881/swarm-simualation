"""Geofence: an inclusion polygon, no-fly exclusion zones and an optional ceiling, with a breach action.

Every tick the monitor checks every airborne drone that is still under normal control against:

* **inclusion**: the drone must stay inside the polygon (if one is set),
* **exclusions**: the drone must stay outside every no-fly polygon,
* **ceiling**: optional maximum altitude above home.

Breaches are *predictive*: a drone breaches when its current position **or** its position
``lookahead_s`` seconds ahead (``p + v * lookahead``) violates the fence, so with the default 2 s
lookahead a HOLD stops a drone before it crosses into a no-fly zone. The configured action
(``RTL`` | ``LAND`` | ``HOLD``) is applied once per breach episode and a CRITICAL event is logged. The
episode ends when the drone has been clear for ``clear_time`` seconds. Drones already returning,
landing or in an emergency are not commanded again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from simulation.events import EventBus, EventCategory, Severity
from simulation.types import FlightMode

from .geometry import Polygon, PolygonError, as_polygon, points_in_polygon, polygon_to_list

if TYPE_CHECKING:
    from simulation.drone import Drone

BREACH_ACTIONS = ("RTL", "LAND", "HOLD")
EXEMPT_MODES = frozenset({FlightMode.RTL, FlightMode.LAND, FlightMode.EMERGENCY, FlightMode.TAKEOFF})


@dataclass
class ExclusionZone:
    name: str
    polygon: Polygon

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "polygon": polygon_to_list(self.polygon)}


@dataclass
class Geofence:
    enabled: bool = False
    action: str = "RTL"
    inclusion: Polygon | None = None
    exclusions: list[ExclusionZone] = field(default_factory=list)
    max_altitude: float | None = None       # [m above home], None = world ceiling only
    lookahead_s: float = 2.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, lookahead_s: float = 2.0) -> "Geofence":
        """Validate a fence definition (from a command, the config or a mission file)."""
        if not isinstance(data, Mapping):
            raise PolygonError("geofence must be an object")
        unknown = set(data) - {"enabled", "action", "inclusion", "exclusions", "max_altitude"}
        if unknown:
            raise PolygonError(f"geofence: unknown key(s) {', '.join(sorted(unknown))}")
        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            raise PolygonError("geofence.enabled must be true/false")
        action = str(data.get("action", "RTL")).upper()
        if action not in BREACH_ACTIONS:
            raise PolygonError(f"geofence.action must be one of {', '.join(BREACH_ACTIONS)}")
        inclusion = as_polygon(data["inclusion"], "geofence.inclusion") if data.get("inclusion") else None
        raw = data.get("exclusions") or []
        if not isinstance(raw, list) or len(raw) > 50:
            raise PolygonError("geofence.exclusions must be a list of at most 50 zones")
        zones = []
        for k, z in enumerate(raw):
            if isinstance(z, Mapping):
                name, poly = str(z.get("name") or f"NFZ {k + 1}")[:40], z.get("polygon")
            else:
                name, poly = f"NFZ {k + 1}", z
            zones.append(ExclusionZone(name, as_polygon(poly, f"geofence.exclusions[{k}]")))
        max_alt = data.get("max_altitude")
        if max_alt is not None and (isinstance(max_alt, bool) or not isinstance(max_alt, (int, float)) or max_alt <= 0):
            raise PolygonError("geofence.max_altitude must be a positive number or null")
        return cls(enabled, action, inclusion, zones, float(max_alt) if max_alt is not None else None, lookahead_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "action": self.action,
            "inclusion": polygon_to_list(self.inclusion) if self.inclusion is not None else None,
            "exclusions": [z.to_dict() for z in self.exclusions],
            "max_altitude": self.max_altitude,
        }

    @property
    def defined(self) -> bool:
        return self.inclusion is not None or bool(self.exclusions) or self.max_altitude is not None

    def violations(self, points: np.ndarray, ground: float = 0.0) -> list[str | None]:
        """Per point: None if inside the fence, else a short reason."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        out: list[str | None] = [None] * len(pts)
        if self.max_altitude is not None:
            for k in np.flatnonzero(pts[:, 2] - ground > self.max_altitude):
                out[k] = f"above fence ceiling {self.max_altitude:.0f} m"
        for zone in self.exclusions:
            for k in np.flatnonzero(points_in_polygon(pts, zone.polygon)):
                out[k] = out[k] or f"inside no-fly zone '{zone.name}'"
        if self.inclusion is not None:
            for k in np.flatnonzero(~points_in_polygon(pts, self.inclusion)):
                out[k] = out[k] or "outside the inclusion fence"
        return out


class GeofenceMonitor:
    """Applies the breach action to drones that violate (or are about to violate) the fence."""

    def __init__(self, fence: Geofence, events: EventBus, clear_time: float = 2.0) -> None:
        self.fence = fence
        self.events = events
        self.clear_time = clear_time
        self.version = 1                              # bumped on every change so clients rebuild the 3D fence
        self.total_breaches = 0
        self._breached: dict[int, float] = {}         # drone id -> last time seen in breach

    def set_fence(self, fence: Geofence) -> None:
        self.fence = fence
        self.version += 1
        self._breached.clear()

    def step(self, drones: list["Drone"], positions: np.ndarray, velocities: np.ndarray, t: float,
             ground: float = 0.0) -> None:
        fence = self.fence
        if not fence.enabled or not fence.defined or not drones:
            return
        active = [k for k, d in enumerate(drones) if d.in_flight and d.airborne and not d.failed]
        if not active:
            return
        idx = np.array(active)
        now = fence.violations(positions[idx], ground)
        ahead = fence.violations(positions[idx] + velocities[idx] * fence.lookahead_s, ground)
        for r, k in enumerate(idx):
            drone = drones[k]
            reason = now[r] or (f"{ahead[r]} in {fence.lookahead_s:.0f} s" if ahead[r] else None)
            if reason is None:
                last = self._breached.get(drone.id)
                if last is not None and t - last > self.clear_time:
                    del self._breached[drone.id]
                continue
            first = drone.id not in self._breached
            self._breached[drone.id] = t
            if not first or drone.flight_mode in EXEMPT_MODES:
                continue
            self.total_breaches += 1
            result = self._apply(drone)
            self.events.emit(EventCategory.MISSION, "geofence_breach",
                             f"geofence breach: {reason} -> {fence.action} ({result})",
                             severity=Severity.CRITICAL, drone_id=drone.id, time=t, action=fence.action, reason=reason)

    def _apply(self, drone: "Drone") -> str:
        action = self.fence.action
        if action == "RTL":
            result = drone.return_to_home()
        elif action == "LAND":
            result = drone.land()
        else:
            result = drone.hover()
        return result.message

    @property
    def breached_ids(self) -> list[int]:
        return sorted(self._breached)

    def snapshot(self) -> dict[str, Any]:
        return {**self.fence.to_dict(), "version": self.version, "lookahead_s": self.fence.lookahead_s,
                "breached": self.breached_ids, "total_breaches": self.total_breaches}
