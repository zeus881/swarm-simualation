import numpy as np
import pytest

from simulation.spatial import CollisionMonitor, SpatialIndex
from simulation.swarm import VelocityStage
from simulation.types import CollisionState

from .conftest import make_config


def test_kdtree_matches_brute_force():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-100, 100, size=(80, 3))
    kd, bf = SpatialIndex(True), SpatialIndex(False)
    kd.rebuild(pts)
    bf.rebuild(pts)
    for a, b in zip(kd.query_neighbors(40, 6), bf.query_neighbors(40, 6)):
        assert list(a) == list(b)
    pa, da = kd.query_pairs(30)
    pb, db = bf.query_pairs(30)
    assert {tuple(p) for p in pa} == {tuple(p) for p in pb}
    assert len(pa) > 0


def test_neighbors_exclude_self_even_with_coincident_points():
    idx = SpatialIndex()
    idx.rebuild(np.array([[0, 0, 0], [0, 0, 0], [50, 0, 0]], dtype=float))
    n = idx.query_neighbors(10, 5)
    assert list(n[0]) == [1] and list(n[1]) == [0] and len(n[2]) == 0


def test_collision_monitor_levels_and_events(engine):
    mon = CollisionMonitor(separation=5, warning=10, collision=1, events=engine.events)
    idx = SpatialIndex()
    pts = np.array([[0, 0, 20], [8, 0, 20], [100, 0, 20], [103, 0, 20], [200, 0, 20], [200.5, 0, 20]], dtype=float)
    idx.rebuild(pts)
    report = mon.update([1, 2, 3, 4, 5, 6], idx, np.ones(6, bool), 0.0)
    assert report.states == [CollisionState.WARNING] * 2 + [CollisionState.AVOIDANCE] * 2 + [CollisionState.COLLISION] * 2
    assert report.min_separation == pytest.approx(0.5)
    kinds = [e.kind for e in engine.events.recent(limit=100) if e.category == "COLLISION"]
    assert kinds.count("separation_violation") == 1 and kinds.count("collision") == 1
    # Persisting conflicts are not re-reported every tick.
    mon.update([1, 2, 3, 4, 5, 6], idx, np.ones(6, bool), 0.1)
    assert mon.total_violations == 1 and mon.total_collisions == 1
    # Grounded drones are ignored.
    report = mon.update([1, 2, 3, 4, 5, 6], idx, np.zeros(6, bool), 0.2)
    assert all(s == CollisionState.CLEAR for s in report.states)


def test_spawn_uses_distinct_pads(engine):
    homes = np.array([d.home_position for d in engine.swarm])
    assert len(homes) == 3
    d = np.linalg.norm(homes[:, None, :2] - homes[None, :, :2], axis=-1) + np.eye(len(homes)) * 1e9
    assert d.min() >= engine.config.home.pad_spacing - 1e-9


def test_add_and_remove_drones(engine):
    new = engine.swarm.add_drone()
    assert new.id == 4 and len(engine.swarm) == 4
    engine.swarm.remove_drone(2)
    assert 2 not in engine.swarm
    again = engine.swarm.add_drone()
    assert again.id == 5                                           # ids are never reused
    homes = [tuple(d.home_position) for d in engine.swarm]
    assert len(set(homes)) == len(homes)                          # freed pad reused, no overlap
    with pytest.raises(KeyError):
        engine.swarm.remove_drone(99)


def test_max_drones_enforced(tmp_path):
    from simulation.engine import SimulationEngine
    eng = SimulationEngine(make_config(simulation__drone_count=2, simulation__max_drones=2), record=False)
    with pytest.raises(ValueError):
        eng.swarm.add_drone()


def test_neighbors_and_collision_state_assigned_in_flight(engine):
    # Detection only: switch avoidance off so the drone can actually be flown into the safety radius.
    assert engine.execute({"type": "set_avoidance", "params": {"enabled": False}}).success
    engine.execute({"type": "takeoff", "params": {"altitude": 20}})
    engine.run_for(8)
    d1, d2, d3 = engine.swarm.drones
    assert d2.id in d1.neighbors                                    # pads 12 m apart < 60 m radius
    d2.goto(d1.position + np.array([3.0, 0, 0]))
    engine.run_for(8)
    assert d1.collision_state in (CollisionState.AVOIDANCE, CollisionState.COLLISION)
    assert engine.swarm.collision_monitor.total_violations >= 1


def test_avoidance_keeps_the_same_manoeuvre_outside_the_safety_radius(engine):
    engine.execute({"type": "takeoff", "params": {"altitude": 20}})
    engine.run_for(8)
    d1, d2, _ = engine.swarm.drones
    d2.goto(d1.position + np.array([3.0, 0, 0]))
    min_sep = np.inf
    for _ in range(240):
        engine.step()
        min_sep = min(min_sep, float(np.linalg.norm(d1.position - d2.position)))
    assert engine.swarm.collision_monitor.total_violations == 0
    assert min_sep > engine.config.swarm.separation_distance


def test_velocity_pipeline_stages_run_in_priority_order(engine):
    calls = []

    class Stage(VelocityStage):
        def __init__(self, name, priority):
            self.name, self.priority = name, priority

        def apply(self, ctx, v):
            calls.append(self.name)
            assert v.shape == (len(ctx.drones), 3)
            return v

    engine.swarm.add_stage(Stage("collision", 40))
    engine.swarm.add_stage(Stage("mission", 10))
    engine.swarm.add_stage(Stage("formation", 20))
    engine.step()
    assert calls == ["mission", "formation", "collision"]


def test_stage_output_drives_the_drones(engine):
    class Climb(VelocityStage):
        name, priority = "climb", 10

        def apply(self, ctx, v):
            out = v.copy()
            out[ctx.controllable, 2] = 2.0
            return out

    engine.execute({"type": "takeoff", "params": {"altitude": 10}})
    engine.run_for(6)
    engine.swarm.add_stage(Climb())
    z0 = engine.swarm.positions()[:, 2].copy()
    engine.run_for(3)
    assert np.all(engine.swarm.positions()[:, 2] > z0 + 3)
