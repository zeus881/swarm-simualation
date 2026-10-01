"""Mission data model and strict JSON schema validation.

A mission is one or more *tracks* of waypoints. A plain mission has one track, which a single drone
or the whole formation flies. A split survey has one track per drone. The on-disk format
(``mission_<name>.json``) is::

    {
      "schema": "gandiv.mission/1",
      "name": "North patrol",
      "altitude_mode": "relative",                 # relative (to home) | agl (terrain following, Stage 4)
      "waypoints": [                               # single track ...
        {"x": 120.0, "y": 80.0, "alt": 30.0, "speed": 8.0, "hold": 0.0, "action": "WAYPOINT", "params": {}},
        {"action": "CHANGE_FORMATION", "x": 120.0, "y": 80.0, "alt": 30.0, "params": {"shape": "line"}}
      ],
      "tracks": [[...], [...]],                    # ... or several tracks (one per drone)
      "survey": {...} | null,                      # parameters the tracks were generated from (informative)
      "geofence": {...} | null                     # optional fence saved with the mission
    }

Positions are local ENU metres (x = East, y = North); ``lat``/``lon`` may be given instead of x/y on
import. ``alt`` is metres above home (or above ground in ``agl`` mode). Unknown keys are rejected so a
typo never silently drops a field.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Mapping

SCHEMA_ID = "gandiv.mission/1"
MAX_WAYPOINTS = 500
MAX_TRACKS = 200


class MissionAction(StrEnum):
    WAYPOINT = "WAYPOINT"
    LOITER = "LOITER"
    TAKEOFF = "TAKEOFF"
    LAND = "LAND"
    RTL = "RTL"
    CHANGE_FORMATION = "CHANGE_FORMATION"


# Actions that fly to the waypoint's position first.
POSITIONAL_ACTIONS = frozenset({MissionAction.WAYPOINT, MissionAction.LOITER, MissionAction.LAND})


class MissionError(ValueError):
    """Raised for a mission that does not match the schema."""


def _num(data: Mapping[str, Any], key: str, path: str, *, default: float | None = None, minimum: float | None = None,
         maximum: float | None = None, required: bool = False) -> float | None:
    value = data.get(key, default)
    if value is None:
        if required:
            raise MissionError(f"{path}.{key} is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MissionError(f"{path}.{key} must be a finite number")
    if minimum is not None and value < minimum:
        raise MissionError(f"{path}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise MissionError(f"{path}.{key} must be <= {maximum}")
    return float(value)


def _check_keys(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    if not isinstance(data, Mapping):
        raise MissionError(f"{path} must be an object")
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise MissionError(f"{path}: unknown key(s) {', '.join(unknown)}")


@dataclass(slots=True)
class Waypoint:
    x: float
    y: float
    alt: float
    speed: float | None = None          # m/s, None = default cruise speed
    hold: float = 0.0                   # s, hold (or loiter) time after arrival
    action: MissionAction = MissionAction.WAYPOINT
    params: dict[str, Any] = field(default_factory=dict)

    _KEYS = {"x", "y", "lat", "lon", "alt", "speed", "hold", "action", "params"}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], path: str = "waypoint",
                  geodetic_to_enu: Callable[[float, float], tuple[float, float]] | None = None) -> "Waypoint":
        _check_keys(data, cls._KEYS, path)
        raw_action = data.get("action", "WAYPOINT")
        try:
            action = MissionAction(str(raw_action).upper())
        except ValueError:
            raise MissionError(f"{path}.action must be one of {', '.join(a.value for a in MissionAction)}") from None
        if "x" in data or "y" in data:
            x = _num(data, "x", path, required=True, minimum=-1e6, maximum=1e6)
            y = _num(data, "y", path, required=True, minimum=-1e6, maximum=1e6)
        elif "lat" in data or "lon" in data:
            if geodetic_to_enu is None:
                raise MissionError(f"{path}: lat/lon given but no geodetic origin is available")
            lat = _num(data, "lat", path, required=True, minimum=-90, maximum=90)
            lon = _num(data, "lon", path, required=True, minimum=-180, maximum=180)
            x, y = geodetic_to_enu(lat, lon)
        elif action in (MissionAction.RTL, MissionAction.CHANGE_FORMATION, MissionAction.TAKEOFF):
            x = y = 0.0                     # position not used by these actions
        else:
            raise MissionError(f"{path} needs x/y (or lat/lon)")
        alt = _num(data, "alt", path, default=30.0, minimum=-500.0, maximum=10000.0)
        speed = _num(data, "speed", path, minimum=0.1, maximum=100.0)
        hold = _num(data, "hold", path, default=0.0, minimum=0.0, maximum=86400.0)
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            raise MissionError(f"{path}.params must be an object")
        params = dict(params)
        if action == MissionAction.CHANGE_FORMATION:
            shape = params.get("shape")
            if not isinstance(shape, str) or not shape:
                raise MissionError(f"{path}.params.shape is required for CHANGE_FORMATION")
            if "spacing" in params:
                _num(params, "spacing", f"{path}.params", minimum=1.0, maximum=500.0)
        if action == MissionAction.LOITER and "radius" in params:
            _num(params, "radius", f"{path}.params", minimum=1.0, maximum=2000.0)
        return cls(x, y, alt, speed, hold, action, params)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"x": round(self.x, 3), "y": round(self.y, 3), "alt": round(self.alt, 3),
                               "hold": round(self.hold, 3), "action": str(self.action)}
        if self.speed is not None:
            out["speed"] = round(self.speed, 3)
        if self.params:
            out["params"] = dict(self.params)
        return out

    @property
    def positional(self) -> bool:
        return self.action in POSITIONAL_ACTIONS


@dataclass(slots=True)
class Mission:
    name: str
    tracks: list[list[Waypoint]]
    altitude_mode: str = "relative"
    survey: dict[str, Any] | None = None
    geofence: dict[str, Any] | None = None

    _KEYS = {"schema", "name", "altitude_mode", "waypoints", "tracks", "survey", "geofence", "created", "notes"}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any],
                  geodetic_to_enu: Callable[[float, float], tuple[float, float]] | None = None) -> "Mission":
        """Parse and validate a mission; raises :class:`MissionError` with the offending path."""
        _check_keys(data, cls._KEYS, "mission")
        schema = data.get("schema", SCHEMA_ID)
        if schema != SCHEMA_ID:
            raise MissionError(f"mission.schema must be '{SCHEMA_ID}' (got {schema!r})")
        name = data.get("name", "mission")
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise MissionError("mission.name must be a non-empty string of at most 80 characters")
        mode = data.get("altitude_mode", "relative")
        if mode not in ("relative", "agl"):
            raise MissionError("mission.altitude_mode must be 'relative' or 'agl'")
        if ("waypoints" in data) == ("tracks" in data):
            raise MissionError("mission needs exactly one of 'waypoints' or 'tracks'")
        raw_tracks = [data["waypoints"]] if "waypoints" in data else data["tracks"]
        if not isinstance(raw_tracks, list) or not raw_tracks or len(raw_tracks) > MAX_TRACKS:
            raise MissionError(f"mission.tracks must be a list of 1..{MAX_TRACKS} waypoint lists")
        tracks = []
        for t, raw in enumerate(raw_tracks):
            path = "mission.waypoints" if "waypoints" in data else f"mission.tracks[{t}]"
            if not isinstance(raw, list) or not raw:
                raise MissionError(f"{path} must be a non-empty list")
            if len(raw) > MAX_WAYPOINTS:
                raise MissionError(f"{path} has more than {MAX_WAYPOINTS} waypoints")
            tracks.append([Waypoint.from_dict(w, f"{path}[{k}]", geodetic_to_enu) for k, w in enumerate(raw)])
        for key in ("survey", "geofence"):
            if data.get(key) is not None and not isinstance(data[key], Mapping):
                raise MissionError(f"mission.{key} must be an object or null")
        return cls(name.strip(), tracks, mode, dict(data["survey"]) if data.get("survey") else None,
                   dict(data["geofence"]) if data.get("geofence") else None)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_ID, "name": self.name, "altitude_mode": self.altitude_mode}
        if len(self.tracks) == 1:
            out["waypoints"] = [w.to_dict() for w in self.tracks[0]]
        else:
            out["tracks"] = [[w.to_dict() for w in track] for track in self.tracks]
        out["survey"] = self.survey
        out["geofence"] = self.geofence
        return out

    @property
    def waypoint_count(self) -> int:
        return sum(len(t) for t in self.tracks)


def slugify(name: str) -> str:
    """File-system safe mission name: lower-case letters, digits, '-' and '_' only."""
    slug = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    return slug[:60] or "mission"
