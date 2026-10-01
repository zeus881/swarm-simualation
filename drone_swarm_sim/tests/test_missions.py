"""Stage 2: mission model, geometry, survey, QGC WPL 110, storage, validation, geofence, mission execution."""

import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from missions.geometry import (PolygonError, as_polygon, distance_to_polygon_edge, point_in_polygon, points_in_polygon,
                               segment_crosses_polygon)
from missions.model import SCHEMA_ID, Mission, MissionAction, MissionError, Waypoint
from missions.qgc import import_qgc_wpl, export_qgc_wpl
from missions.storage import MissionStore
from missions.survey import generate_survey, spacing_from_overlap, sweep_lines
from missions.validation import estimate_track
from simulation.engine import SimulationEngine
from simulation.geo import GeoReference
from simulation.types import FlightMode

from .conftest import make_config

SQUARE = [[0, 0], [200, 0], [200, 200], [0, 200]]
CONCAVE_U = [[0, 0], [300, 0], [300, 200], [200, 200], [200, 60], [100, 60], [100, 200], [0, 200]]


def engine(n: int = 3, **overrides) -> SimulationEngine:
    return SimulationEngine(make_config(simulation__drone_count=n, **overrides), record=False)


def run_until_done(eng: SimulationEngine, limit_s: float = 600.0) -> str:
    run = eng.missions.runs[-1]
    end = eng.sim_time + limit_s
    while run.state == "RUNNING" and eng.sim_time < end:
        eng.step()
    return str(run.state)


def patrol(**extra) -> dict:
    return {"name": "patrol", "waypoints": [
        {"action": "TAKEOFF", "alt": 25},
        {"x": 120, "y": 0, "alt": 30, "hold": 2},
        {"x": 120, "y": 120, "alt": 30},
        {"action": "RTL"},
    ], **extra}


# ----------------------------------------------------------------------------- model

def test_mission_roundtrip_and_defaults():
    m = Mission.from_dict(patrol())
    assert len(m.tracks) == 1 and m.waypoint_count == 4
    assert m.tracks[0][0].action == MissionAction.TAKEOFF and m.tracks[0][1].hold == 2
    again = Mission.from_dict(json.loads(json.dumps(m.to_dict())))
    assert again.to_dict() == m.to_dict() and m.to_dict()["schema"] == SCHEMA_ID


@pytest.mark.parametrize("data,fragment", [
    ({"name": "x", "waypoints": [{"x": 1, "y": 2, "altitude": 5}]}, "unknown key"),
    ({"name": "x", "waypoints": [{"x": 1, "y": 2, "action": "BARREL_ROLL"}]}, "action must be"),
    ({"name": "x", "waypoints": [{"x": 1, "y": 2, "action": "CHANGE_FORMATION"}]}, "shape is required"),
    ({"name": "x", "waypoints": [{"x": 1}]}, "is required"),
    ({"name": "x", "waypoints": [{"y": 2, "action": "WAYPOINT"}]}, "is required"),
    ({"name": "x", "waypoints": [{"x": "a", "y": 2}]}, "finite number"),
    ({"name": "x", "waypoints": [{"x": 1, "y": 2, "hold": -1}]}, ">= 0"),
    ({"name": "x", "waypoints": []}, "non-empty"),
    ({"name": "x", "waypoints": [{"x": 1, "y": 1}], "tracks": [[{"x": 1, "y": 1}]]}, "exactly one"),
    ({"name": "x", "schema": "other/9", "waypoints": [{"x": 1, "y": 1}]}, "schema"),
    ({"name": "", "waypoints": [{"x": 1, "y": 1}]}, "name"),
    ({"name": "x", "waypoints": [{"x": 1, "y": 1}], "extra": 1}, "unknown key"),
])
def test_mission_schema_rejects_bad_input(data, fragment):
    with pytest.raises(MissionError, match=fragment):
        Mission.from_dict(data)


def test_waypoints_may_use_lat_lon():
    geo = GeoReference(47.397742, 8.545594, 488.0)
    lat, lon, _ = geo.enu_to_geodetic([150.0, -80.0, 0.0])

    def to_enu(la, lo):
        e = geo.geodetic_to_enu(la, lo, 488.0)
        return float(e[0]), float(e[1])

    m = Mission.from_dict({"name": "geo", "waypoints": [{"lat": float(lat), "lon": float(lon), "alt": 20}]}, to_enu)
    assert m.tracks[0][0].x == pytest.approx(150.0, abs=0.01) and m.tracks[0][0].y == pytest.approx(-80.0, abs=0.01)


# ----------------------------------------------------------------------------- geometry

def test_point_in_concave_polygon_and_crossing():
    poly = as_polygon(CONCAVE_U)
    assert point_in_polygon((50, 150), poly) and not point_in_polygon((150, 150), poly)   # inside a leg / the notch
    pts = np.array([[50, 150, 0], [150, 150, 0], [250, 30, 0], [-5, 5, 0]])
    assert points_in_polygon(pts, poly).tolist() == [True, False, True, False]
    assert segment_crosses_polygon((150, 250), (150, 30), poly)          # enters through the notch floor
    assert not segment_crosses_polygon((150, 250), (150, 100), poly)     # stays in the notch


@pytest.mark.parametrize("bad", [[[0, 0], [1, 1]], [[0, 0], [10, 10], [10, 0], [0, 10]], "nope", [[0, 0], [0, 0], [0, 0]]])
def test_invalid_polygons_are_rejected(bad):
    with pytest.raises(PolygonError):
        as_polygon(bad)


# ----------------------------------------------------------------------------- survey

def test_spacing_from_overlap():
    assert spacing_from_overlap(30, 90, 50) == pytest.approx(30.0)      # footprint 60 m, 50 % overlap


def test_survey_covers_square_with_requested_spacing():
    res = generate_survey(SQUARE, altitude=30, line_spacing=20, angle_deg=90)   # sweep East-West
    assert res.lines == 10 and len(res.tracks) == 1
    ys = sorted({round(w.y, 3) for w in res.tracks[0] if w.action == MissionAction.WAYPOINT})
    assert ys[0] == pytest.approx(10) and ys[-1] == pytest.approx(190)
    assert np.allclose(np.diff(ys), 20)
    assert res.tracks[0][-1].action == MissionAction.RTL                  # default finish
    assert all(w.alt == 30 for w in res.tracks[0])


def test_survey_concave_polygon_has_split_lines_inside():
    poly = as_polygon(CONCAVE_U)
    lines = sweep_lines(poly, 20.0, 90.0)
    assert max(len(segs) for segs in lines) == 2                          # the notch splits upper lines in two
    res = generate_survey(CONCAVE_U, altitude=30, line_spacing=20, angle_deg=90, finish="HOLD")
    wps = res.tracks[0]
    for a, b in zip(wps[0::2], wps[1::2]):                                # every survey pass lies inside
        assert point_in_polygon(((a.x + b.x) / 2, (a.y + b.y) / 2), poly)
    # Cell by cell: the notch between the two legs of the U is crossed exactly once.
    transits = [(a, b) for a, b in zip(wps[1:-1:2], wps[2::2])]

    def outside(a, b):   # clearly outside somewhere along the transit (turns may run along walls: tolerance 1 m)
        for f in (0.25, 0.5, 0.75):
            p = (a.x + (b.x - a.x) * f, a.y + (b.y - a.y) * f)
            if not point_in_polygon(p, poly) and distance_to_polygon_edge(p, poly) > 1.0:
                return True
        return False
    assert sum(outside(a, b) for a, b in transits) == 1


def test_survey_split_between_drones_is_balanced_and_disjoint():
    res = generate_survey(SQUARE, altitude=30, line_spacing=10, drones=4, angle_deg=90)
    assert len(res.tracks) == 4
    lengths = np.array(res.track_lengths)
    assert lengths.max() / lengths.min() < 1.35
    bands = [sorted({round(w.y, 3) for w in t if w.action == MissionAction.WAYPOINT}) for t in res.tracks]
    for a, b in zip(bands, bands[1:]):
        assert a[-1] < b[0] or b[-1] < a[0]                               # each drone owns its own strip


@pytest.mark.parametrize("kwargs", [dict(altitude=0), dict(altitude=30), dict(altitude=30, overlap=100),
                                    dict(altitude=30, line_spacing=0.5), dict(altitude=30, line_spacing=10, finish="NOPE")])
def test_survey_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_survey(SQUARE, **kwargs)


# ----------------------------------------------------------------------------- QGC WPL 110

def _qgc_env():
    geo = GeoReference(47.397742, 8.545594, 488.0)

    def enu_to_geo(x, y, z):
        lat, lon, alt = geo.enu_to_geodetic([x, y, z])
        return float(lat), float(lon), float(alt)

    def geo_to_enu(lat, lon):
        e = geo.geodetic_to_enu(lat, lon, 488.0)
        return float(e[0]), float(e[1])
    return enu_to_geo, geo_to_enu


def test_qgc_export_format():
    enu_to_geo, _ = _qgc_env()
    m = Mission("x", [[Waypoint(0, 0, 25, None, action=MissionAction.TAKEOFF), Waypoint(100, 50, 30, 8.0, 5.0),
                       Waypoint(200, 50, 30, 8.0, 20.0, MissionAction.LOITER, {"radius": 25}),
                       Waypoint(0, 0, 30, None, 0, MissionAction.CHANGE_FORMATION, {"shape": "line"}),
                       Waypoint(0, 0, 30, None, 0, MissionAction.RTL)]])
    text = export_qgc_wpl(m, enu_to_geo, (0, 0, 0), 488.0)
    lines = text.split("\r\n")
    assert lines[0] == "QGC WPL 110"
    home = lines[1].split("\t")
    assert home[:4] == ["0", "1", "0", "16"] and float(home[8]) == pytest.approx(47.397742) and float(home[10]) == 488.0
    rows = [ln.split("\t") for ln in lines[2:] if ln]
    assert all(len(r) == 12 for r in rows)
    assert [int(r[3]) for r in rows] == [22, 178, 16, 19, 31010, 20]      # takeoff, speed, wp, loiter, formation, rtl
    assert [int(r[0]) for r in rows] == list(range(1, 7))
    assert all(r[2] == "3" for r in rows)                                 # relative altitude frame
    speed = rows[1]
    assert float(speed[4]) == 1 and float(speed[5]) == 8.0
    loiter = rows[3]
    assert float(loiter[4]) == 20.0 and float(loiter[6]) == 25.0


def test_qgc_roundtrip_preserves_mission():
    enu_to_geo, geo_to_enu = _qgc_env()
    original = Mission.from_dict({"name": "rt", "waypoints": [
        {"action": "TAKEOFF", "alt": 20}, {"x": 120.5, "y": -40.25, "alt": 30, "speed": 9, "hold": 3},
        {"x": 300, "y": 80, "alt": 45, "speed": 9, "action": "LOITER", "hold": 30, "params": {"radius": 40}},
        {"x": 50, "y": 50, "alt": 20, "action": "CHANGE_FORMATION", "params": {"shape": "circle", "spacing": 18}},
        {"x": 10, "y": 10, "alt": 0, "action": "LAND"}]})
    text = export_qgc_wpl(original, enu_to_geo, (0, 0, 0), 488.0)
    back, warnings = import_qgc_wpl(text, geo_to_enu, 488.0)
    assert not warnings
    a, b = original.tracks[0], back.tracks[0]
    assert [w.action for w in a] == [w.action for w in b]
    for wa, wb in zip(a[1:], b[1:]):
        if wa.action == MissionAction.CHANGE_FORMATION:
            assert wb.params == {"shape": "circle", "spacing": 18.0}
            continue
        assert math.hypot(wa.x - wb.x, wa.y - wb.y) < 0.01
        assert wa.hold == wb.hold
    assert b[1].speed == 9 and b[2].params["radius"] == 40 and b[1].alt == 30


def test_qgc_import_rejects_and_warns():
    _, geo_to_enu = _qgc_env()
    with pytest.raises(MissionError, match="not a QGC WPL"):
        import_qgc_wpl("hello", geo_to_enu, 488.0)
    with pytest.raises(MissionError, match="12 fields"):
        import_qgc_wpl("QGC WPL 110\r\n0\t1\t0\t16\t0", geo_to_enu, 488.0)
    text = ("QGC WPL 110\n0\t1\t0\t16\t0\t0\t0\t0\t47.397742\t8.545594\t488\t1\n"
            "1\t0\t3\t16\t0\t0\t0\t0\t47.398\t8.546\t30\t1\n"
            "2\t0\t3\t206\t0\t0\t0\t0\t0\t0\t0\t1\n")                     # DO_SET_CAM_TRIGG_DIST: unsupported
    m, warnings = import_qgc_wpl(text, geo_to_enu, 488.0)
    assert len(m.tracks[0]) == 1 and len(warnings) == 1 and "206" in warnings[0]


# ----------------------------------------------------------------------------- storage

def test_store_save_list_load_delete(tmp_path):
    store = MissionStore(tmp_path / "missions")
    name, m = store.save(patrol(name="North Patrol #1"))
    assert name == "mission_north_patrol_1.json"
    listing = store.list()
    assert listing[0]["file"] == name and listing[0]["valid"] and listing[0]["waypoints"] == 4
    assert store.load(name).to_dict()["waypoints"] == m.to_dict()["waypoints"]
    (tmp_path / "missions" / "mission_broken.json").write_text('{"name": "b", "waypoints": [{"x": 1}]}')
    assert any(not e["valid"] for e in store.list())
    with pytest.raises(MissionError):
        store.load("mission_broken.json")
    store.delete(name)
    assert all(e["file"] != name for e in store.list())


@pytest.mark.parametrize("filename", ["../secrets.json", "mission_../x.json", "notes.txt", "mission_A B.json"])
def test_store_rejects_unsafe_names(tmp_path, filename):
    with pytest.raises(MissionError):
        MissionStore(tmp_path).load(filename)


def test_store_refuses_invalid_mission(tmp_path):
    with pytest.raises(MissionError):
        MissionStore(tmp_path).save({"name": "x", "waypoints": [{"x": 1, "y": 2, "action": "JUMP"}]})
    assert not list(tmp_path.glob("*.json"))


# ----------------------------------------------------------------------------- validation

def test_validation_flags_fence_altitude_and_crossing():
    eng = engine(2)
    eng.execute({"type": "set_geofence", "params": {
        "enabled": True, "inclusion": [[-300, -300], [300, -300], [300, 300], [-300, 300]],
        "exclusions": [{"name": "Tower", "polygon": [[80, -20], [120, -20], [120, 20], [80, 20]]}]}})
    m = {"name": "bad", "waypoints": [
        {"x": 0, "y": 0, "alt": 30}, {"x": 200, "y": 0, "alt": 30},       # leg crosses the tower
        {"x": 100, "y": 0, "alt": 30},                                    # inside the tower
        {"x": 400, "y": 0, "alt": 30},                                    # outside the inclusion fence
        {"x": 0, "y": 50, "alt": 500}]}                                   # above the ceiling
    res = eng.execute({"type": "mission_validate", "drone_ids": [1], "params": {"mission": m}})
    assert res.success                                                     # warnings only
    text = " | ".join(res.data["warnings"])
    for fragment in ("crosses no-fly zone 'Tower'", "inside no-fly zone 'Tower'", "outside the inclusion geofence",
                     "unreachable"):
        assert fragment in text, fragment
    assert res.data["tracks"][0]["energy_wh"] > 0


def test_validation_warns_about_insufficient_battery():
    eng = engine(1)
    eng.swarm.get(1).battery.energy_wh = eng.swarm.get(1).battery.capacity_wh * 0.25   # 5 % above the reserve
    far = {"name": "far", "waypoints": [{"x": 900, "y": 900, "alt": 40}, {"x": -900, "y": 900, "alt": 40}]}
    res = eng.execute({"type": "mission_validate", "drone_ids": [1], "params": {"mission": far}})
    assert any("insufficient battery" in w for w in res.data["warnings"])
    assert res.data["battery"][0]["ok"] is False
    start = eng.execute({"type": "mission_start", "drone_ids": [1], "params": {"mission": far}})
    assert not start.success and start.data["needs_confirmation"]          # warnings block without force


def test_energy_estimate_matches_simulated_flight():
    eng = engine(1)
    d = eng.swarm.get(1)
    m = {"name": "leg", "waypoints": [{"action": "TAKEOFF", "alt": 30}, {"x": 400, "y": 300, "alt": 30, "hold": 10},
                                      {"action": "RTL"}]}
    est = estimate_track(Mission.from_dict(m).tracks[0], d.position.copy(), eng.environment.home_position,
                         eng.config)
    before = d.battery.energy_wh
    assert eng.execute({"type": "mission_start", "drone_ids": [1], "params": {"mission": m}}).success
    assert run_until_done(eng) == "COMPLETED"
    used = before - d.battery.energy_wh
    assert est.energy_wh == pytest.approx(used, rel=0.25)
    assert est.duration_s == pytest.approx(d.flight_time, rel=0.3)


# ----------------------------------------------------------------------------- geofence

def _fly_towards(eng: SimulationEngine, target, seconds: float = 40.0) -> float:
    eng.execute({"type": "takeoff", "params": {"altitude": 20}})
    eng.run_for(8)
    eng.execute({"type": "goto", "params": {"position": target}})
    furthest = -np.inf
    for _ in range(int(seconds * 30)):
        eng.step()
        furthest = max(furthest, float(eng.swarm.get(1).position[0]))
    return furthest


@pytest.mark.parametrize("action,mode", [("HOLD", FlightMode.HOVER), ("RTL", FlightMode.RTL), ("LAND", FlightMode.LAND)])
def test_no_fly_zone_breach_actions(action, mode):
    eng = engine(1)
    res = eng.execute({"type": "set_geofence", "params": {"enabled": True, "action": action,
                       "exclusions": [{"name": "NFZ", "polygon": [[100, -50], [150, -50], [150, 50], [100, 50]]}]}})
    assert res.success
    eng.execute({"type": "takeoff", "params": {"altitude": 20}})
    eng.run_for(8)
    eng.execute({"type": "goto", "params": {"position": [300, 0, 20]}})
    max_x = -np.inf
    seen_mode = None
    for _ in range(40 * 30):
        eng.step()
        d = eng.swarm.get(1)
        max_x = max(max_x, float(d.position[0]))
        if seen_mode is None and any(e.kind == "geofence_breach" for e in eng.events.recent(limit=20)):
            seen_mode = d.flight_mode
    assert seen_mode == mode
    assert max_x < 100.0                                                   # predictive: never entered the zone
    breach = [e for e in eng.events.recent(limit=300) if e.kind == "geofence_breach"]
    assert len(breach) == 1 and breach[0].severity == "CRITICAL" and breach[0].data["action"] == action


def test_inclusion_fence_and_ceiling():
    eng = engine(1)
    eng.execute({"type": "set_geofence", "params": {"enabled": True, "action": "HOLD", "max_altitude": 40,
                 "inclusion": [[-100, -100], [100, -100], [100, 100], [-100, 100]]}})
    assert 50.0 < _fly_towards(eng, [400, 0, 20]) < 100.0                # stopped inside the inclusion fence
    eng.execute({"type": "goto", "params": {"position": [0, 0, 80]}})
    eng.run_for(30)
    assert eng.swarm.get(1).position[2] < 45.0
    assert eng.missions.geofence.total_breaches == 2


def test_disabled_fence_does_nothing():
    eng = engine(1)
    eng.execute({"type": "set_geofence", "params": {"enabled": False, "exclusions": [[[100, -50], [150, -50], [150, 50], [100, 50]]]}})
    assert _fly_towards(eng, [300, 0, 20]) > 150
    assert eng.missions.geofence.total_breaches == 0


@pytest.mark.parametrize("params,fragment", [
    ({"inclusion": [[0, 0], [30, 30], [30, 0], [0, 10]]}, "intersects itself"),
    ({"exclusions": [[[0, 0], [5000, 0], [5000, 10]]]}, "outside the world"),
    ({"action": "EXPLODE"}, "action"),
    ({"max_altitude": -5}, "max_altitude"),
    ({"bogus": 1}, "unknown key"),
])
def test_set_geofence_validation(params, fragment):
    res = engine(1).execute({"type": "set_geofence", "params": params})
    assert not res.success and fragment in res.message


def test_clear_geofence_bumps_version():
    eng = engine(1)
    v0 = eng.missions.geofence.version
    eng.execute({"type": "set_geofence", "params": {"exclusions": [[[10, 10], [20, 10], [20, 20]]]}})
    assert eng.execute({"type": "clear_geofence"}).success
    snap = eng.snapshot()["missions"]["geofence"]
    assert snap["version"] == v0 + 2 and snap["exclusions"] == [] and not snap["enabled"]


def test_config_geofence_loaded_and_validated():
    eng = engine(1, geofence__enabled=True, geofence__exclusions=[[[50, 50], [60, 50], [60, 60]]])
    assert eng.missions.geofence.fence.enabled and len(eng.missions.geofence.fence.exclusions) == 1
    from simulation.config import ConfigError
    with pytest.raises(ConfigError):
        engine(1, geofence__inclusion=[[0, 0], [10, 10], [10, 0], [0, 10]])


# ----------------------------------------------------------------------------- execution

def test_formation_mission_end_to_end():
    eng = engine(5)
    m = patrol()
    m["waypoints"].insert(3, {"action": "CHANGE_FORMATION", "params": {"shape": "line"}})
    m["waypoints"].insert(4, {"x": 0, "y": 120, "alt": 30, "action": "LOITER", "hold": 10, "params": {"radius": 30}})
    res = eng.execute({"type": "mission_start", "params": {"mission": m}})
    assert res.success and "5 drone" in res.message
    assert run_until_done(eng) == "COMPLETED"
    assert all(not d.in_flight for d in eng.swarm)
    kinds = [e.message for e in eng.events.recent(limit=500) if e.category == "MISSION"]
    assert sum("reached waypoint" in k for k in kinds) >= 3
    assert any("formation LINE" in k for k in kinds)
    mon = eng.swarm.collision_monitor
    assert mon.total_violations == 0 and mon.total_hard_violations == 0


def test_single_drone_loiter_orbits_the_waypoint():
    eng = engine(1)
    m = {"name": "orbit", "waypoints": [{"x": 100, "y": 100, "alt": 30, "action": "LOITER", "hold": 40,
                                          "params": {"radius": 30}}]}
    assert eng.execute({"type": "mission_start", "drone_ids": [1], "params": {"mission": m}}).success
    d = eng.swarm.get(1)
    radii, angles = [], []
    while eng.missions.runs[-1].tracks[0].phase != "LOITER" and eng.sim_time < 120:
        eng.step()
    for _ in range(35 * 30):
        eng.step()
        radii.append(math.hypot(d.position[0] - 100, d.position[1] - 100))
        angles.append(math.atan2(d.position[1] - 100, d.position[0] - 100))
    swept = np.sum(np.abs(np.diff(np.unwrap(angles))))
    assert np.median(radii) == pytest.approx(30, abs=5) and swept > math.pi   # really circling
    assert run_until_done(eng) == "COMPLETED" and d.in_flight                 # no RTL: it holds at the end


@pytest.mark.parametrize("n", [1, 3])
def test_land_waypoint_lands_at_the_point(n):
    """LAND at altitude 0: approach at the current altitude, then descend vertically (regression)."""
    eng = engine(n)
    m = {"name": "hop", "waypoints": [{"x": 60, "y": 20, "alt": 15}, {"x": 60, "y": 60, "alt": 0, "action": "LAND"}]}
    assert eng.execute({"type": "mission_start", "params": {"mission": m}}).success
    assert run_until_done(eng, 200) == "COMPLETED"
    pos = eng.swarm.positions()
    assert all(not d.in_flight for d in eng.swarm)
    assert np.linalg.norm(pos[:, :2].mean(axis=0) - [60, 60]) < 3.0 and np.allclose(pos[:, 2], 0.0, atol=0.05)


def test_survey_mission_split_between_three_drones():
    eng = engine(3)
    res = generate_survey([[60, 60], [300, 60], [300, 240], [60, 240]], altitude=30, speed=10, drones=3, line_spacing=30)
    m = Mission("survey", res.tracks).to_dict()
    start = eng.execute({"type": "mission_start", "params": {"mission": m}})
    assert start.success
    run = eng.missions.runs[-1]
    assert len(run.tracks) == 3 and sorted(i for tr in run.tracks for i in tr.drone_ids) == [1, 2, 3]
    assert all(not tr.formation for tr in run.tracks)
    assert run_until_done(eng, 900) == "COMPLETED"
    assert all(not d.in_flight for d in eng.swarm)                          # finished with RTL + land
    assert eng.swarm.collision_monitor.total_hard_violations == 0


def test_pause_resume_abort():
    eng = engine(1)
    m = {"name": "long", "waypoints": [{"x": 600, "y": 0, "alt": 30}]}
    eng.execute({"type": "mission_start", "drone_ids": [1], "params": {"mission": m}})
    eng.run_for(20)
    assert eng.execute({"type": "mission_pause"}).success
    d = eng.swarm.get(1)
    assert d.flight_mode == FlightMode.HOVER
    x = d.position[0]
    eng.run_for(10)
    assert abs(d.position[0] - x) < 15.0                                    # holding (braking distance only)
    assert eng.execute({"type": "mission_resume"}).success
    eng.run_for(10)
    assert d.flight_mode == FlightMode.GOTO and d.position[0] > x + 20
    assert eng.execute({"type": "mission_abort"}).success
    assert eng.missions.runs[-1].state == "ABORTED" and d.flight_mode == FlightMode.HOVER
    assert not eng.execute({"type": "mission_resume"}).success


def test_operator_command_and_failsafe_release_drones():
    eng = engine(3)
    m = {"name": "m", "tracks": [[{"x": 300, "y": y, "alt": 30}] for y in (-100, 0, 100)]}
    eng.execute({"type": "mission_start", "params": {"mission": m}})
    eng.run_for(15)
    eng.execute({"type": "hover", "drone_ids": [2]})                       # operator takes D02
    victim = next(i for i in (1, 3) if i in eng.missions.runs[-1].drone_ids)
    eng.swarm.get(victim).battery.energy_wh = eng.swarm.get(victim).battery.capacity_wh * 0.195  # battery RTL
    eng.run_for(1)
    run = eng.missions.runs[-1]
    assert 2 not in run.drone_ids and victim not in run.drone_ids
    released = [e.message for e in eng.events.recent(limit=300) if "left the mission" in e.message]
    assert any("operator command 'hover'" in r for r in released) and any("RTL" in r for r in released)
    assert run.state == "RUNNING"                                           # the third drone carries on


def test_groups_and_mission_by_group():
    eng = engine(4)
    assert eng.execute({"type": "define_group", "drone_ids": [3, 4], "params": {"name": "Bravo"}}).success
    assert eng.snapshot()["missions"]["groups"] == {"Bravo": [3, 4]}
    res = eng.execute({"type": "mission_start", "params": {"mission": patrol(), "group": "Bravo"}})
    assert res.success and eng.missions.runs[-1].drone_ids == {3, 4}
    assert not eng.execute({"type": "mission_start", "params": {"mission": patrol(), "group": "Zulu"}}).success
    assert eng.execute({"type": "delete_group", "params": {"name": "Bravo"}}).success
    assert not eng.execute({"type": "define_group", "params": {"name": "x"}}).success      # needs drone_ids


@pytest.mark.parametrize("params,ids,fragment", [
    ({"mission": {"name": "t", "tracks": [[{"x": 1, "y": 1}]] * 3}}, [1, 2], "3 tracks but only 2"),
    ({"mission": patrol(), "formation": False}, [1, 2], "without a formation"),
    ({"mission": {"name": "x", "waypoints": [{"x": 1}]}}, [1], "invalid mission"),
    ({"mission": "nope"}, [1], "mission object"),
])
def test_mission_start_rejections(params, ids, fragment):
    res = engine(3).execute({"type": "mission_start", "drone_ids": ids, "params": params})
    assert not res.success and fragment in res.message


def test_mission_commands_are_logged_and_snapshot_is_json():
    eng = engine(2)
    eng.execute({"type": "mission_start", "params": {"mission": patrol()}})
    eng.run_for(2)
    snap = eng.snapshot()
    json.dumps(snap, allow_nan=False)
    assert snap["missions"]["runs"][0]["state"] == "RUNNING"
    assert any(e.category == "COMMAND" and e.kind == "mission_start" for e in eng.events.recent(limit=200))
    paths = eng.missions.active_paths()
    assert len(paths) == 1 and len(paths[0]["waypoints"]) == 4 and paths[0]["formation"]


# ----------------------------------------------------------------------------- REST

@pytest.fixture
def client(tmp_path):
    app = create_app(make_config(mission__directory=str(tmp_path / "missions")), record=False)
    with TestClient(app) as c:
        yield c


def test_rest_files_survey_and_qgc(client):
    r = client.post("/api/missions/files", json={"mission": patrol(name="Alpha 1")})
    assert r.status_code == 200 and r.json()["file"] == "mission_alpha_1.json"
    assert client.get("/api/missions/files").json()[0]["name"] == "Alpha 1"
    loaded = client.get("/api/missions/files/mission_alpha_1.json").json()
    assert loaded["name"] == "Alpha 1" and len(loaded["waypoints"]) == 4
    assert client.get("/api/missions/files/mission_nope.json").status_code == 404
    assert client.get("/api/missions/files/..%2Fconfig.json").status_code in (404, 422)
    assert client.post("/api/missions/files", json={"mission": {"name": "x", "waypoints": [{"x": 1}]}}).status_code == 422
    assert client.delete("/api/missions/files/mission_alpha_1.json").status_code == 200

    s = client.post("/api/missions/survey", json={"polygon": SQUARE, "altitude": 30, "drones": 2, "line_spacing": 25})
    body = s.json()
    assert s.status_code == 200 and len(body["mission"]["tracks"]) == 2 and body["stats"]["lines"] == 8
    assert client.post("/api/missions/survey", json={"polygon": [[0, 0], [1, 1]], "altitude": 30}).status_code == 422

    q = client.post("/api/missions/export/qgc?track=1", json=body["mission"])
    assert q.status_code == 200 and q.text.startswith("QGC WPL 110") and "track2.waypoints" in q.headers["content-disposition"]
    back = client.post("/api/missions/import/qgc", json={"text": q.text, "name": "back"}).json()
    assert len(back["mission"]["waypoints"]) == len(body["mission"]["tracks"][1]) and back["warnings"] == []
    assert client.post("/api/missions/import/qgc", json={"text": "garbage"}).status_code == 422
    assert client.post("/api/missions/normalize", json=patrol()).json()["schema"] == SCHEMA_ID
    assert client.post("/api/missions/normalize", json={"name": "x"}).status_code == 422
    assert client.get("/api/missions/active").json()["paths"] == []
