"""Stage 4: terrain, obstacles + path planning, communication model, sensors + Kalman filter, failure injection."""

import math
import struct
import zlib

import numpy as np
import pytest

from algorithms.path_planning import PathPlanner
from estimation.kalman_filter import BatchKalmanFilter
from simulation.config import ConfigError, config_from_dict
from simulation.engine import SimulationEngine
from simulation.obstacles import ObstacleManager, SceneError, parse_scene
from simulation.terrain import Terrain, TerrainError, load_heightmap
from simulation.types import CommStatus, FlightMode

from .conftest import make_config

WALL_SCENE = {"obstacles": [{"type": "building", "name": "Wall", "x": 150, "y": 0, "width": 12, "depth": 160, "height": 60}]}


def engine(n: int = 1, **overrides) -> SimulationEngine:
    return SimulationEngine(make_config(simulation__drone_count=n, **overrides), record=False)


def scene_file(tmp_path, data: dict) -> str:
    import yaml
    p = tmp_path / "scene.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(p)


def airborne(eng: SimulationEngine, alt: float = 30.0) -> SimulationEngine:
    eng.execute({"type": "takeoff", "params": {"altitude": alt}})
    eng.run_for(12)
    return eng


# ----------------------------------------------------------------------------- terrain

def test_terrain_bilinear_and_vectorised():
    grid = np.array([[0.0, 10.0], [20.0, 30.0]])        # row 0 = north edge
    t = Terrain(grid, -50, -50, 100, 100)
    assert t.height(-50, 50) == pytest.approx(0.0) and t.height(50, 50) == pytest.approx(10.0)
    assert t.height(-50, -50) == pytest.approx(20.0) and t.height(0, 0) == pytest.approx(15.0)
    xs, ys = np.array([-50, 0, 50, 500]), np.array([50, 0, -50, 0])
    np.testing.assert_allclose(t.heights(xs, ys), [t.height(x, y) for x, y in zip(xs, ys)])


def test_procedural_terrain_is_deterministic_and_flat_at_home():
    eng_a, eng_b = engine(terrain__enabled=True), engine(terrain__enabled=True)
    env = eng_a.environment
    assert env.terrain.max_height > 5
    assert env.ground_height(300, -200) == eng_b.environment.ground_height(300, -200)
    home = env.home_position
    for dx, dy in [(40, 0), (0, -60), (-50, 50)]:
        assert env.ground_height(home[0] + dx, home[1] + dy) == pytest.approx(home[2], abs=0.05)
    assert eng_a.swarm.get(1).position[2] == pytest.approx(home[2], abs=0.05)     # pad on the terrain


def _png_gray16(path, grid):
    """16-bit grayscale PNG with filter 1 (Sub): byte-wise difference to the byte 2 positions back."""
    rows = []
    for row in grid:
        line = b"".join(struct.pack(">H", int(v)) for v in row)
        sub = bytes((line[i] - (line[i - 2] if i >= 2 else 0)) % 256 for i in range(len(line)))
        rows.append(b"\x01" + sub)
    raw = b"".join(rows)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", len(grid[0]), len(grid), 16, 0, 0, 0, 0)) \
        + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def test_heightmap_formats(tmp_path):
    grid = np.array([[0, 100, 200], [300, 400, 500]], dtype=float)
    (tmp_path / "h.asc").write_text("ncols 3\nnrows 2\nxllcorner 0\nyllcorner 0\ncellsize 10\nNODATA_value -9999\n"
                                    "0 100 200\n300 -9999 500\n")
    asc = load_heightmap(tmp_path / "h.asc", 50)
    assert asc.shape == (2, 3) and asc[1, 1] == 0.0                       # NODATA -> minimum
    np.save(tmp_path / "h.npy", grid)
    np.testing.assert_allclose(load_heightmap(tmp_path / "h.npy", 50), grid)
    (tmp_path / "h.csv").write_text("0,100,200\n300,400,500\n")
    np.testing.assert_allclose(load_heightmap(tmp_path / "h.csv", 50), grid)
    _png_gray16(tmp_path / "h.png", [[0, 32768, 65535], [65535, 0, 0]])
    png = load_heightmap(tmp_path / "h.png", 40.0)
    np.testing.assert_allclose(png, [[0, 40 * 32768 / 65535, 40], [40, 0, 0]], atol=1e-9)
    with pytest.raises(TerrainError):
        load_heightmap(tmp_path / "missing.asc", 10)
    (tmp_path / "bad.txt").write_text("1 2\nx y\n")
    with pytest.raises(TerrainError):
        load_heightmap(tmp_path / "bad.txt", 10)


def test_terrain_from_file_config(tmp_path):
    np.save(tmp_path / "ramp.npy", np.tile(np.linspace(0, 50, 11), (11, 1)))   # rises towards the east
    eng = engine(terrain__enabled=True, terrain__source="file", terrain__file=str(tmp_path / "ramp.npy"),
                 terrain__flatten_radius=0.0)
    env = eng.environment
    assert env.ground_height(-1000, 0) == pytest.approx(0.0) and env.ground_height(1000, 0) == pytest.approx(50.0)
    with pytest.raises(ConfigError):
        engine(terrain__enabled=True, terrain__source="file", terrain__file=str(tmp_path / "nope.npy"))


def test_agl_goto_and_terrain_clearance(tmp_path):
    np.save(tmp_path / "hill.npy", np.pad(np.full((3, 3), 40.0), 4))            # a 40 m plateau in the middle
    eng = airborne(engine(terrain__enabled=True, terrain__source="file", terrain__file=str(tmp_path / "hill.npy"),
                          terrain__flatten_radius=0.0, home__position_x=-800.0, home__position_y=-800.0))
    d = eng.swarm.get(1)
    start_ground = eng.environment.ground_height(*d.position[:2])
    assert d.altitude_agl == pytest.approx(30, abs=1.0)
    # relative-altitude goto straight into the plateau: the terrain barrier keeps the drone above it
    eng.execute({"type": "goto", "params": {"position": [0, 0, start_ground + 20], "plan": False}})
    lowest = np.inf
    for _ in range(140 * 30):
        eng.step()
        lowest = min(lowest, d.altitude_agl)
    assert lowest >= eng.config.obstacles.terrain_clearance - 0.3 and not d.failed
    # AGL goto: 25 m above the plateau
    eng.execute({"type": "goto", "params": {"position": [0, 0, 25], "agl": True}})
    eng.run_for(40)
    assert d.altitude_agl == pytest.approx(25, abs=1.5)


# ----------------------------------------------------------------------------- obstacles

def test_scene_parsing_and_validation():
    obs = parse_scene({"obstacles": [
        {"type": "building", "x": 0, "y": 0, "width": 20, "depth": 10, "height": 15, "rotation": 30},
        {"type": "forest", "x": 100, "y": 100, "radius": 30, "count": 12, "seed": 2}]}, lambda x, y: 0.0)
    assert len(obs) == 13 and sum(o.kind == "tree" for o in obs) == 12
    for bad in ({"obstacles": [{"type": "ufo", "x": 0, "y": 0}]},
                {"obstacles": [{"type": "tower", "x": 0, "y": 0, "radius": 2, "height": 10, "colour": "red"}]},
                {"obstacles": [{"type": "tower", "x": 0, "y": 0, "radius": -1, "height": 10}]},
                {"trees": []}):
        with pytest.raises(SceneError):
            parse_scene(bad, lambda x, y: 0.0)


def test_signed_distance_box_and_cylinder():
    obs = parse_scene({"obstacles": [
        {"type": "building", "x": 0, "y": 0, "width": 20, "depth": 10, "height": 30},
        {"type": "tower", "x": 100, "y": 0, "radius": 5, "height": 40}]}, lambda x, y: 0.0)
    m = ObstacleManager(obs)
    pts = np.array([[20, 0, 10], [0, 0, 40], [0, 0, 10], [100, 20, 10], [100, 0, 50], [16, 0, 36]])
    sdf, normal = m.sdf_pairs(pts, np.array([0, 0, 0, 1, 1, 0]))
    np.testing.assert_allclose(sdf[:2], [10.0, 10.0])
    assert sdf[2] == pytest.approx(-5.0)                                     # inside: distance to nearest face
    np.testing.assert_allclose(sdf[3:5], [15.0, 10.0])
    assert sdf[5] == pytest.approx(math.hypot(6, 6))                         # over the edge: corner distance
    np.testing.assert_allclose(normal[0], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(normal[1], [0, 0, 1], atol=1e-9)
    # KD-tree candidates agree with brute force
    rng = np.random.default_rng(0)
    forest = ObstacleManager(parse_scene({"obstacles": [{"type": "forest", "x": 0, "y": 0, "radius": 200, "count": 150,
                                                         "seed": 1}]}, lambda x, y: 0.0))
    q = rng.uniform(-250, 250, (40, 3))
    q[:, 2] = 5
    d, _, _ = forest.nearest(q, 60.0)
    brute = np.array([min(math.hypot(p[0] - o.x, p[1] - o.y) - o.radius for o in forest.obstacles) for p in q])
    near = brute < 60
    np.testing.assert_allclose(d[near], brute[near], atol=1e-9)


def test_obstacle_avoidance_stops_short_and_crash_without_it(tmp_path):
    for avoid in (True, False):
        eng = airborne(engine(environment__scene_file=scene_file(tmp_path, WALL_SCENE), obstacles__avoidance=avoid))
        d = eng.swarm.get(1)
        eng.execute({"type": "goto", "params": {"position": [300, 0, 30], "plan": False}})
        closest = np.inf
        for _ in range(40 * 30):
            eng.step()
            dist, _, _ = eng.environment.obstacles.nearest(d.position[None, :], 60)
            closest = min(closest, dist[0])
        if avoid:
            assert closest >= eng.config.obstacles.clearance * 0.8 and not d.failed
            assert eng.swarm.obstacle_collisions == 0
        else:
            assert d.failed and eng.swarm.obstacle_collisions == 1
            assert any(e.kind == "obstacle_collision" for e in eng.events.recent(limit=200))


@pytest.mark.parametrize("method", ["astar", "rrtstar"])
def test_planner_routes_around_a_wall(tmp_path, method):
    eng = engine(environment__scene_file=scene_file(tmp_path, WALL_SCENE), obstacles__rrt_iterations=1500)
    planner = PathPlanner(eng.environment, eng.config.obstacles)
    start, goal = np.array([0.0, 0.0, 30.0]), np.array([300.0, 0.0, 30.0])
    res = planner.plan(start, goal, method=method)
    assert res.ok and res.method == method and len(res.path) >= 2
    pts = [start, *res.path]
    for a, b in zip(pts[:-1], pts[1:]):
        assert eng.environment.obstacles.segment_clear(a, b, eng.config.obstacles.clearance * 0.9)
    assert np.allclose(res.path[-1], goal)
    assert planner.plan(start, np.array([100.0, 0.0, 30.0])).method == "direct"


def test_goto_is_routed_around_obstacles(tmp_path):
    eng = airborne(engine(environment__scene_file=scene_file(tmp_path, WALL_SCENE)))
    d = eng.swarm.get(1)
    res = eng.execute({"type": "goto", "params": {"position": [300, 0, 30]}})
    assert res.success and "via" in res.message
    assert any(e.kind == "path_planned" for e in eng.events.recent(limit=50))
    eng.run_for(80)
    assert np.linalg.norm(d.position[:2] - [300, 0]) < 2.0 and d.flight_mode == FlightMode.HOVER
    assert eng.swarm.obstacle_collisions == 0 and not d.failed


def test_formation_swarm_goto_routes_around(tmp_path):
    eng = airborne(engine(4, environment__scene_file=scene_file(tmp_path, WALL_SCENE)))
    eng.execute({"type": "set_formation", "params": {"shape": "line", "spacing": 12}})
    eng.run_for(15)
    res = eng.execute({"type": "swarm_goto", "params": {"position": [320, 0, 30]}})
    assert res.success and "via" in res.message
    eng.run_for(120)
    centroid = eng.swarm.positions().mean(axis=0)
    assert np.linalg.norm(centroid[:2] - [320, 0]) < 5.0
    assert eng.swarm.obstacle_collisions == 0 and eng.swarm.collision_monitor.total_hard_violations == 0


def test_mission_validation_flags_obstacles_and_terrain(tmp_path):
    np.save(tmp_path / "hill.npy", np.pad(np.full((3, 3), 60.0), 4))
    eng = engine(environment__scene_file=scene_file(tmp_path, WALL_SCENE), terrain__enabled=True, terrain__source="file",
                 terrain__file=str(tmp_path / "hill.npy"), terrain__flatten_radius=0.0,
                 home__position_x=-800.0, home__position_y=-800.0)
    m = {"name": "m", "waypoints": [{"x": 0, "y": -50, "alt": 30}, {"x": 300, "y": -50, "alt": 30},
                                    {"x": 150, "y": 0, "alt": 30}, {"x": 0, "y": 0, "alt": 20}]}
    res = eng.execute({"type": "mission_validate", "params": {"mission": m}})
    text = " | ".join(res.data["warnings"])
    assert "passes an obstacle" in text and "within" in text and "below the terrain" in text


# ----------------------------------------------------------------------------- communication

def comm_engine(n: int = 1, **overrides) -> SimulationEngine:
    return engine(n, communication__enabled=True, communication__packet_loss=0.0, **overrides)


def test_commands_travel_the_link_with_latency_and_telemetry_is_delayed():
    eng = comm_engine(communication__latency_ms=200, communication__jitter_ms=0)
    eng.run_for(1)
    res = eng.execute({"type": "takeoff", "params": {"altitude": 20}})
    assert res.success and "sent" in res.message
    d = eng.swarm.get(1)
    assert d.flight_mode == FlightMode.DISARMED                            # not delivered yet
    eng.run_for(0.3)
    assert d.flight_mode == FlightMode.TAKEOFF
    assert any(e.kind == "command_ack" and "taking off" in e.message for e in eng.events.recent(limit=50))
    eng.run_for(3)
    view = eng.snapshot()["drones"][0]
    assert 0.15 <= view["telemetry_age_s"] <= 0.35                          # latency + heartbeat phase
    assert view["position"]["z"] < d.position[2]                           # climbing: the GCS sees it lower


def test_out_of_range_link_loss_triggers_failsafe_and_rejects_commands():
    eng = airborne(comm_engine(communication__range=300.0, communication__failsafe_timeout_s=3.0,
                               communication__failsafe_action="RTL"))
    d = eng.swarm.get(1)
    assert d.comm_status == CommStatus.ONLINE and d.link_quality > 95
    eng.execute({"type": "goto", "params": {"position": [400, 0, 30], "speed": 12}})
    for _ in range(60 * 30):
        eng.step()
        if d.flight_mode == FlightMode.RTL:
            break
    kinds = [e.kind for e in eng.events.recent(limit=300) if e.drone_id == 1]
    assert "comm_lost" in kinds and "failsafe" in kinds and d.flight_mode == FlightMode.RTL
    assert np.linalg.norm(d.position[:2]) > 250                             # it lost the link near the range edge
    lost_cmd = eng.execute({"type": "hover", "drone_ids": [1]})
    assert not lost_cmd.success and "no link" in lost_cmd.message
    eng.run_for(40)                                                          # flying home restores the link
    assert d.comm_status == CommStatus.ONLINE
    assert any(e.kind == "comm_restored" for e in eng.events.recent(limit=300))


def test_packet_loss_degrades_link_quality():
    eng = engine(1, communication__enabled=True, communication__packet_loss=0.5)
    eng.run_for(10)
    d = eng.swarm.get(1)
    assert 30 < d.link_quality < 70 and d.comm_status == CommStatus.DEGRADED


def test_mission_releases_drone_on_link_loss():
    eng = comm_engine(2)
    eng.run_for(1)
    eng.execute({"type": "mission_start", "params": {"mission": {"name": "m", "tracks": [
        [{"x": 150, "y": 0, "alt": 20}], [{"x": 150, "y": 30, "alt": 20}]]}}})
    eng.run_for(12)
    eng.execute({"type": "inject_failure", "drone_ids": [1], "params": {"type": "comm_loss", "duration": 5}})
    eng.run_for(3)
    assert 1 not in eng.missions.runs[-1].drone_ids and 2 in eng.missions.runs[-1].drone_ids


# ----------------------------------------------------------------------------- sensors + Kalman filter

def test_batch_kalman_filter_converges():
    rng = np.random.default_rng(3)
    kf = BatchKalmanFilter(np.zeros((4, 3)), np.zeros((4, 3)), pos_var=100.0, process_noise=0.02)
    truth = np.array([[10.0, -5.0, 30.0]] * 4)
    for _ in range(300):
        kf.predict(np.zeros((4, 3)), 0.1)
        kf.update_position(truth + rng.normal(0, 2.0, (4, 3)), 4.0, np.ones(4, dtype=bool))
    assert np.abs(kf.p - truth).max() < 0.8 and kf.horizontal_sigma().max() < 1.0


def test_estimate_filters_gps_noise_and_gps_loss_grows_uncertainty():
    eng = airborne(engine(sensors__enabled=True, sensors__gps_drift=0.0, sensors__baro_drift=0.0, sensors__gps_noise_h=2.0))
    d = eng.swarm.get(1)
    errors = []
    for _ in range(20 * 30):
        eng.step()
        errors.append(np.linalg.norm(d.est_position[:2] - d.position[:2]))
    assert np.mean(errors) < 1.2                                            # well below the 2 m GPS noise
    sigma_ok = d.pos_sigma
    eng.execute({"type": "inject_failure", "drone_ids": [1], "params": {"type": "gps_loss", "duration": 20}})
    eng.run_for(15)
    assert d.gps_fix == "NONE" and d.satellites == 0 and d.pos_sigma > 5 * sigma_ok
    eng.run_for(10)
    assert d.gps_fix == "3D" and d.pos_sigma < 2 * sigma_ok
    t = d.get_telemetry().to_dict()
    assert t["pos_error"] is not None and t["est_position"] is not None


def test_autopilot_flies_on_the_estimate():
    """With a large GPS bias the drone holds position on its (biased) estimate, so it really moves."""
    eng = airborne(engine(sensors__enabled=True, sensors__gps_drift=6.0, sensors__gps_drift_time_s=1e6))
    d = eng.swarm.get(1)
    eng.run_for(20)
    offset = np.linalg.norm(d.est_position[:2] - d.position[:2])
    assert offset > 2.0                                                     # it believes it is elsewhere
    truth = engine(sensors__enabled=True, sensors__gps_drift=6.0, sensors__gps_drift_time_s=1e6,
                   sensors__control_source="truth")
    airborne(truth).run_for(20)
    assert np.linalg.norm(truth.swarm.get(1).position[:2] - truth.swarm.get(1).home_position[:2]) < 0.5


# ----------------------------------------------------------------------------- failure injection

def test_motor_failures():
    eng = airborne(engine(2))
    res = eng.execute({"type": "inject_failure", "drone_ids": [1], "params": {"type": "motor"}})
    assert res.success
    d1 = eng.swarm.get(1)
    assert d1.flight_mode == FlightMode.EMERGENCY and "motor" in d1.failures
    eng.execute({"type": "inject_failure", "drone_ids": [2], "params": {"type": "motor", "severity": "total"}})
    d2 = eng.swarm.get(2)
    assert d2.failed
    eng.run_for(40)
    assert not d1.in_flight and not d1.airborne and not d2.airborne
    assert any(e.kind == "failure_injected" and e.severity == "CRITICAL" for e in eng.events.recent(limit=200))


def test_battery_sag_and_wind_gust_expire():
    eng = airborne(engine(2))
    d1, d2 = eng.swarm.get(1), eng.swarm.get(2)
    before = d1.battery.percent
    eng.execute({"type": "inject_failure", "drone_ids": [1], "params": {"type": "battery_sag", "drop": 40}})
    assert d1.battery.percent == pytest.approx(before - 40, abs=0.2)
    start = d2.position.copy()
    eng.execute({"type": "inject_failure", "drone_ids": [2], "params": {"type": "wind_gust", "speed": 20, "direction": 270,
                                                                         "duration": 4}})
    eng.run_for(2)
    assert d2.position[0] - start[0] > 1.0                                 # pushed east (wind from the west)
    eng.run_for(3)
    assert not d2.wind_disturbance.any() and "wind_gust" not in d2.failures
    assert any(e.kind == "failure_cleared" for e in eng.events.recent(limit=100))


@pytest.mark.parametrize("params,ids,fragment", [
    ({"type": "meteor"}, [1], "type must be"),
    ({"type": "motor"}, None, "explicit drone_ids"),
    ({"type": "gps_loss"}, [1], "sensor model"),
    ({"type": "comm_loss"}, [1], "communication model"),
    ({"type": "motor", "severity": "some"}, [1], "partial or total"),
])
def test_inject_failure_validation(params, ids, fragment):
    res = engine(1).execute({"type": "inject_failure", "drone_ids": ids, "params": params})
    assert not res.success and fragment in res.message


def test_clear_failure_and_snapshot():
    eng = airborne(engine(1, communication__enabled=True, communication__packet_loss=0.0))
    eng.execute({"type": "inject_failure", "drone_ids": [1], "params": {"type": "comm_loss"}})
    assert eng.snapshot()["failures"]["active"][0]["type"] == "comm_loss"
    eng.run_for(3)
    assert eng.swarm.get(1).comm_status == CommStatus.LOST
    assert eng.execute({"type": "clear_failure", "params": {"type": "comm_loss"}}).success
    eng.run_for(2)
    assert eng.swarm.get(1).comm_status == CommStatus.ONLINE and eng.snapshot()["failures"]["active"] == []


# ----------------------------------------------------------------------------- configuration

@pytest.mark.parametrize("data", [
    {"terrain": {"source": "file"}},
    {"obstacles": {"planner": "dijkstra"}},
    {"obstacles": {"clearance": 30, "influence": 20}},
    {"obstacles": {"terrain_clearance": 5.0}},                 # above drone.min_altitude (2 m)
    {"communication": {"failsafe_action": "SELF_DESTRUCT"}},
    {"communication": {"timeout_s": 0.1}},
    {"sensors": {"control_source": "guess"}},
    {"failures": {"motor_partial_thrust": 1.5}},
])
def test_stage4_config_validation(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_shipped_configuration_enables_realism():
    from simulation.config import load_config
    cfg = load_config()
    assert cfg.terrain.enabled and cfg.communication.enabled and cfg.sensors.enabled and cfg.environment.scene_file
    eng = SimulationEngine(cfg, record=False)
    assert len(eng.environment.obstacles) > 10 and eng.comms is not None and eng.navigator is not None
    eng.close()
