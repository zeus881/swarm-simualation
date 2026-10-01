"""QGroundControl / Mission Planner plain-text waypoint files (``QGC WPL 110``).

Each line is tab separated::

    <seq> <current> <frame> <command> <p1> <p2> <p3> <p4> <lat> <lon> <alt> <autocontinue>

Line 0 is the home position (frame 0 = absolute altitude AMSL). Mission items use frame 3
(``MAV_FRAME_GLOBAL_RELATIVE_ALT``, altitude above home) or frame 10 (``MAV_FRAME_GLOBAL_TERRAIN_ALT``)
for ``altitude_mode: agl``. Action mapping:

=================  ==========================================  ======================================
Gandiv action      MAVLink command                              parameters
=================  ==========================================  ======================================
WAYPOINT           16  MAV_CMD_NAV_WAYPOINT                     p1 = hold [s]
LOITER             19  MAV_CMD_NAV_LOITER_TIME                  p1 = time [s], p3 = radius [m]
TAKEOFF            22  MAV_CMD_NAV_TAKEOFF                      altitude
LAND               21  MAV_CMD_NAV_LAND                         position
RTL                20  MAV_CMD_NAV_RETURN_TO_LAUNCH             -
(speed change)     178 MAV_CMD_DO_CHANGE_SPEED                  p1 = 1 (ground speed), p2 = speed [m/s]
CHANGE_FORMATION   31010 MAV_CMD_USER_1 (no standard command)   p1 = shape index, p2 = spacing [m]
=================  ==========================================  ======================================

A speed change is emitted before the first waypoint that uses a new speed, the same way Mission
Planner writes one. Autopilots ignore the formation item: it only means something to this GCS.
"""

from __future__ import annotations

import math
from typing import Callable

from simulation.config import FORMATION_SHAPES

from .model import Mission, MissionAction, MissionError, Waypoint

HEADER = "QGC WPL 110"
FRAME_ABSOLUTE, FRAME_RELATIVE, FRAME_TERRAIN = 0, 3, 10
CMD_WAYPOINT, CMD_LOITER_UNLIM, CMD_LOITER_TURNS, CMD_LOITER_TIME = 16, 17, 18, 19
CMD_RTL, CMD_LAND, CMD_TAKEOFF, CMD_CHANGE_SPEED, CMD_FORMATION = 20, 21, 22, 178, 31010

Enu2Geo = Callable[[float, float, float], tuple[float, float, float]]
Geo2Enu = Callable[[float, float], tuple[float, float]]


def _row(seq: int, frame: int, cmd: int, p: tuple[float, float, float, float], lat: float, lon: float, alt: float,
         current: int = 0) -> str:
    fields = [str(seq), str(current), str(frame), str(cmd), *(f"{v:.6f}" for v in p), f"{lat:.8f}", f"{lon:.8f}",
              f"{alt:.6f}", "1"]
    return "\t".join(fields)


def export_qgc_wpl(mission: Mission, enu_to_geodetic: Enu2Geo, home_enu: tuple[float, float, float],
                   home_amsl: float, track: int = 0, default_loiter_radius: float = 20.0) -> str:
    """One track of ``mission`` as QGC WPL 110 text (CRLF line endings, as Mission Planner writes)."""
    if not 0 <= track < len(mission.tracks):
        raise MissionError(f"track {track} does not exist (mission has {len(mission.tracks)})")
    frame = FRAME_TERRAIN if mission.altitude_mode == "agl" else FRAME_RELATIVE
    hlat, hlon, _ = enu_to_geodetic(home_enu[0], home_enu[1], home_enu[2])
    lines = [HEADER, _row(0, FRAME_ABSOLUTE, CMD_WAYPOINT, (0, 0, 0, 0), hlat, hlon, home_amsl, current=1)]
    seq = 1
    speed: float | None = None
    for wp in mission.tracks[track]:
        if wp.speed is not None and wp.speed != speed and wp.action != MissionAction.RTL:
            lines.append(_row(seq, frame, CMD_CHANGE_SPEED, (1, wp.speed, -1, 0), 0.0, 0.0, 0.0))
            seq += 1
            speed = wp.speed
        lat, lon, _ = enu_to_geodetic(wp.x, wp.y, 0.0)
        if wp.action == MissionAction.WAYPOINT:
            lines.append(_row(seq, frame, CMD_WAYPOINT, (wp.hold, 0, 0, 0), lat, lon, wp.alt))
        elif wp.action == MissionAction.LOITER:
            radius = float(wp.params.get("radius", default_loiter_radius))
            lines.append(_row(seq, frame, CMD_LOITER_TIME, (wp.hold, 0, radius, 0), lat, lon, wp.alt))
        elif wp.action == MissionAction.TAKEOFF:
            lines.append(_row(seq, frame, CMD_TAKEOFF, (0, 0, 0, 0), lat, lon, wp.alt))
        elif wp.action == MissionAction.LAND:
            lines.append(_row(seq, frame, CMD_LAND, (0, 0, 0, 0), lat, lon, 0.0))
        elif wp.action == MissionAction.RTL:
            lines.append(_row(seq, frame, CMD_RTL, (0, 0, 0, 0), 0.0, 0.0, 0.0))
        elif wp.action == MissionAction.CHANGE_FORMATION:
            shape = str(wp.params.get("shape", "v")).lower()
            index = FORMATION_SHAPES.index(shape) if shape in FORMATION_SHAPES else 0
            lines.append(_row(seq, frame, CMD_FORMATION, (index, float(wp.params.get("spacing", 0.0)), 0, 0),
                              lat, lon, wp.alt))
        seq += 1
    return "\r\n".join(lines) + "\r\n"


def import_qgc_wpl(text: str, geodetic_to_enu: Geo2Enu, home_amsl: float, name: str = "imported") -> tuple[Mission, list[str]]:
    """Parse QGC WPL 110 text. Returns the mission and a list of warnings for skipped items."""
    raw_lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    if not raw_lines or not raw_lines[0].startswith("QGC WPL"):
        raise MissionError("not a QGC WPL file (first line must be 'QGC WPL 110')")
    if raw_lines[0].split()[-1] != "110":
        raise MissionError(f"unsupported waypoint file version: {raw_lines[0]!r}")
    warnings: list[str] = []
    waypoints: list[Waypoint] = []
    speed: float | None = None
    mode = "relative"
    for n, line in enumerate(raw_lines[1:], start=2):
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) != 12:
            raise MissionError(f"line {n}: expected 12 fields, got {len(parts)}")
        try:
            seq, _cur, frame, cmd = (int(float(v)) for v in parts[:4])
            p1, p2, p3, _p4, lat, lon, alt = (float(v) for v in parts[4:11])
        except ValueError:
            raise MissionError(f"line {n}: non-numeric field") from None
        if seq == 0:
            continue                                        # home position
        if not all(math.isfinite(v) for v in (lat, lon, alt)):
            raise MissionError(f"line {n}: non-finite coordinate")
        if frame == FRAME_TERRAIN:
            mode = "agl"
        rel_alt = alt - home_amsl if frame == FRAME_ABSOLUTE else alt
        has_pos = not (lat == 0.0 and lon == 0.0)
        x, y = geodetic_to_enu(lat, lon) if has_pos else ((waypoints[-1].x, waypoints[-1].y) if waypoints else (0.0, 0.0))
        if cmd == CMD_CHANGE_SPEED:
            speed = p2 if p2 > 0 else speed
            continue
        if cmd == CMD_WAYPOINT:
            waypoints.append(Waypoint(x, y, rel_alt, speed, max(p1, 0.0)))
        elif cmd in (CMD_LOITER_TIME, CMD_LOITER_UNLIM, CMD_LOITER_TURNS):
            hold = max(p1, 0.0) if cmd == CMD_LOITER_TIME else 60.0
            params = {"radius": abs(p3)} if abs(p3) >= 1.0 else {}
            if cmd != CMD_LOITER_TIME:
                warnings.append(f"line {n}: loiter command {cmd} converted to a 60 s timed loiter")
            waypoints.append(Waypoint(x, y, rel_alt, speed, hold, MissionAction.LOITER, params))
        elif cmd == CMD_TAKEOFF:
            waypoints.append(Waypoint(x, y, rel_alt, speed, 0.0, MissionAction.TAKEOFF))
        elif cmd == CMD_LAND:
            waypoints.append(Waypoint(x, y, 0.0, speed, 0.0, MissionAction.LAND))
        elif cmd == CMD_RTL:
            waypoints.append(Waypoint(x, y, waypoints[-1].alt if waypoints else 30.0, speed, 0.0, MissionAction.RTL))
        elif cmd == CMD_FORMATION:
            index = int(p1)
            shape = FORMATION_SHAPES[index] if 0 <= index < len(FORMATION_SHAPES) else "v"
            params = {"shape": shape, **({"spacing": p2} if p2 > 0 else {})}
            waypoints.append(Waypoint(x, y, rel_alt, speed, 0.0, MissionAction.CHANGE_FORMATION, params))
        else:
            warnings.append(f"line {n}: MAVLink command {cmd} is not supported and was skipped")
    if not waypoints:
        raise MissionError("the file contains no supported mission items")
    return Mission(name, [waypoints], mode), warnings
