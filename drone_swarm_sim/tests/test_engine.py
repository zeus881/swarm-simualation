import json

import numpy as np
import pytest

from simulation.engine import SimulationEngine
from simulation.environment import WindModel
from simulation.config import WindConfig
from simulation.types import FlightMode

from .conftest import make_config


def test_snapshot_is_json_serialisable(engine):
    engine.execute({"type": "takeoff"})
    engine.run_for(2)
    snap = engine.snapshot()
    text = json.dumps(snap, allow_nan=False)
    assert len(snap["drones"]) == 3
    assert json.loads(text)["summary"]["airborne"] == 3


def test_sim_time_is_exact(engine):
    engine.step(300)
    assert engine.tick == 300
    assert engine.sim_time == pytest.approx(10.0, abs=1e-12)


def test_deterministic_with_seed(tmp_path):
    cfg = make_config(wind__enabled=True, wind__gust_strength=2.0, simulation__seed=7)

    def run():
        eng = SimulationEngine(cfg, record=False)
        eng.execute({"type": "takeoff"})
        eng.run_for(5)
        eng.execute({"type": "goto", "params": {"position": [50, 50, 30]}})
        eng.run_for(5)
        return eng.swarm.positions()

    np.testing.assert_array_equal(run(), run())


def test_group_goto_keeps_relative_offsets(engine):
    engine.execute({"type": "takeoff", "params": {"altitude": 20}})
    engine.run_for(8)
    before = engine.swarm.positions()
    res = engine.execute({"type": "goto", "params": {"position": [100, 0, 25]}})
    assert res.success
    engine.run_for(25)
    after = engine.swarm.positions()
    np.testing.assert_allclose(after[:, :2] - after[:, :2].mean(0), before[:, :2] - before[:, :2].mean(0), atol=1.5)
    assert np.allclose(after[:, 2], 25, atol=1.0)


def test_goto_by_gps(engine):
    engine.execute({"type": "takeoff"})
    engine.run_for(6)
    lat, lon, _ = engine.geo.enu_to_geodetic(np.array([60.0, -40.0, 0.0]))
    res = engine.execute({"type": "goto", "drone_ids": [1],
                          "params": {"lat": float(lat), "lon": float(lon), "alt": engine.geo.origin.altitude + 30}})
    assert res.success
    engine.run_for(20)
    np.testing.assert_allclose(engine.swarm.get(1).position, [60, -40, 30], atol=1.0)


@pytest.mark.parametrize("command,fragment", [
    ({"type": "fly_to_moon"}, "unknown command"),
    ({"type": "takeoff", "drone_ids": [42]}, "unknown drone"),
    ({"type": "goto", "params": {}}, "needs 'position'"),
    ({"type": "goto", "params": {"position": [1, 2]}}, "three finite"),
    ({"type": "set_heading", "params": {}}, "required"),
    ({"type": "remove_drone"}, "explicit drone_ids"),
    ({"drone_ids": [1]}, "type"),
    ({"type": "takeoff", "drone_ids": "all"}, "list of integers"),
])
def test_invalid_commands_fail_cleanly(engine, command, fragment):
    res = engine.execute(command)
    assert not res.success and fragment in res.message


def test_partial_acceptance_reported(engine):
    engine.execute({"type": "takeoff", "drone_ids": [1]})
    engine.run_for(1)
    res = engine.execute({"type": "takeoff"})
    assert res.success and "2/3 accepted" in res.message and "D01" in res.details


def test_add_remove_and_wind_commands(engine):
    res = engine.execute({"type": "add_drone", "params": {"count": 2}})
    assert res.success and res.data["ids"] == [4, 5]
    assert engine.execute({"type": "remove_drone", "drone_ids": [4]}).success
    assert [d.id for d in engine.swarm] == [1, 2, 3, 5]
    assert engine.execute({"type": "set_wind", "params": {"speed": 8, "direction": 270}}).success
    assert engine.environment.wind.speed == 8


def test_reset_restores_initial_state(engine):
    engine.execute({"type": "takeoff"})
    engine.execute({"type": "add_drone"})
    engine.run_for(3)
    engine.reset()
    assert engine.tick == 0 and len(engine.swarm) == 3
    assert all(d.flight_mode == FlightMode.DISARMED and not d.airborne for d in engine.swarm)


def test_recorder_writes_files(tmp_path):
    cfg = make_config(logging__record_telemetry=True, logging__telemetry_record_rate_hz=5.0)
    eng = SimulationEngine(cfg, log_dir=tmp_path)
    run_dir = eng.recorder.directory
    eng.execute({"type": "takeoff"})
    eng.run_for(2)
    eng.close()
    rows = (run_dir / "telemetry.csv").read_text().strip().splitlines()
    assert rows[0].startswith("timestamp,drone_id,x,y,z")
    assert len(rows) - 1 == 3 * 10                     # 3 drones x 5 Hz x 2 s
    events = json.loads((run_dir / "mission_events.json").read_text())
    assert any(e["kind"] == "takeoff" and e["category"] == "COMMAND" for e in events)
    assert json.loads((run_dir / "collision_events.json").read_text()) == []
    assert json.loads((run_dir / "config.json").read_text())["simulation"]["drone_count"] == 3


def test_wind_model_statistics():
    wm = WindModel(WindConfig(speed=5, direction=90, gust_strength=1.5, gust_frequency=0.5, shear=False),
                   np.random.default_rng(0))
    np.testing.assert_allclose(wm.mean_velocity, [-5, 0, 0], atol=1e-12)   # from the east -> blows west
    samples = []
    for _ in range(20000):
        wm.step(0.05)
        samples.append(wm.gust.copy())
    s = np.array(samples)
    assert abs(s[:, 0].std() - 1.5) < 0.2 and abs(s[:, 0].mean()) < 0.3


def test_wind_pushes_uncontrolled_drift_is_compensated(tmp_path):
    eng = SimulationEngine(make_config(wind__enabled=True, wind__speed=8.0, wind__gust_strength=0.0), record=False)
    eng.execute({"type": "takeoff", "drone_ids": [1], "params": {"altitude": 20}})
    eng.run_for(20)
    d = eng.swarm.get(1)
    assert np.linalg.norm(d.position[:2] - d.home_position[:2]) < 0.5
    assert d.flight_mode == FlightMode.HOVER
