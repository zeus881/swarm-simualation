"""Phase 2: formation control, slot assignment, leader-follower, flocking, collision avoidance."""

import itertools
import json
import math

import numpy as np
import pytest

from algorithms.coordinator import SwarmMode
from algorithms.formation import FormationShape, assign_slots, formation_offsets, lead_slot
from simulation.engine import SimulationEngine
from simulation.types import FlightMode

from .conftest import make_config

SHAPES = [s for s in FormationShape if s != FormationShape.CUSTOM]


def airborne_engine(n: int, altitude: float = 30.0, **overrides) -> SimulationEngine:
    eng = SimulationEngine(make_config(simulation__drone_count=n, **overrides), record=False)
    eng.execute({"type": "takeoff", "params": {"altitude": altitude}})
    eng.run_for(10)
    return eng


def pairwise_min(points: np.ndarray) -> float:
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    d[np.eye(len(points), dtype=bool)] = np.inf
    return float(d.min())


# ----------------------------------------------------------------------------- geometry

@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("n", [1, 2, 5, 10, 25, 50])
def test_shapes_have_n_slots_centred_and_spaced(shape, n):
    s = 15.0
    off = formation_offsets(shape, n, s)
    assert off.shape == (n, 3)
    np.testing.assert_allclose(off.mean(axis=0), 0.0, atol=1e-9)
    if n > 1:
        assert pairwise_min(off) >= s * 0.999, shape


def test_v_apex_is_the_lead_slot_and_arms_are_symmetric():
    off = formation_offsets("v", 7, 10.0)
    lead = lead_slot(off)
    assert lead == 0
    others = np.delete(off, lead, axis=0) - off[lead]
    assert np.all(others[:, 0] < 0)                               # everyone behind the apex
    assert sorted(np.round(others[:, 1], 6)) == sorted(np.round(-others[:, 1], 6))


def test_custom_shape_and_padding():
    custom = [[0, 0, 0], [-10, 10, 0], [-10, -10, 0]]
    off = formation_offsets("custom", 5, 12.0, custom=custom)
    assert off.shape == (5, 3) and pairwise_min(off) >= 10.0 - 1e-9


def test_hungarian_assignment_is_optimal():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-50, 50, (6, 3))
    slots = rng.uniform(-50, 50, (6, 3))
    got = assign_slots(pos, slots)
    cost = lambda perm: sum(np.sum((pos[i] - slots[perm[i]]) ** 2) for i in range(6))
    best = min(itertools.permutations(range(6)), key=cost)
    assert cost(got) == pytest.approx(cost(best))
    assert sorted(got) == list(range(6))


# ----------------------------------------------------------------------------- formation flight

def test_formation_converges_to_slots():
    eng = airborne_engine(10)
    res = eng.execute({"type": "set_formation", "params": {"shape": "v", "spacing": 15}})
    assert res.success and len(res.data["members"]) == 10
    eng.run_for(40)
    snap = eng.coordinator.snapshot()
    assert snap["mode"] == "FORMATION" and snap["formation"]["max_slot_error"] < 1.5
    pos = eng.swarm.positions()
    assert pairwise_min(pos) > 15.0 * 0.85
    assert all(d.flight_mode == FlightMode.FORMATION for d in eng.swarm)
    assert eng.swarm.collision_monitor.total_violations == 0
    json.dumps(eng.snapshot(), allow_nan=False)


def test_shape_changes_with_25_drones_never_violate_separation():
    """Stage 1 exit test: 25 drones switch through all 8 formations with zero separation violations."""
    eng = airborne_engine(25)
    shapes = ["v", "grid", "circle", "line", "diamond", "wedge", "column", "custom", "v"]
    assert set(shapes) == {str(s) for s in FormationShape}
    for shape in shapes:
        assert eng.execute({"type": "set_formation", "params": {"shape": shape}}).success, shape
        eng.step()
        # Long shapes (a 25-drone column is 360 m) need longer synchronized transitions: wait for it to end.
        assert eng.coordinator.formation._trans_duration < 90.0, shape
        while eng.coordinator.formation.transition_progress < 1.0:
            eng.step()
        eng.run_for(5)
        snap = eng.coordinator.snapshot()["formation"]
        assert snap["max_slot_error"] < 2.0, shape
    mon = eng.swarm.collision_monitor
    assert mon.total_collisions == 0
    assert mon.total_violations == 0                       # never inside separation_distance (5 m)
    assert mon.total_hard_violations == 0                  # never below the hard floor swarm.min_separation
    assert mon.lowest_separation >= eng.config.swarm.separation_distance


def test_formation_moves_and_turns_towards_travel():
    eng = airborne_engine(6)
    eng.execute({"type": "set_formation", "params": {"shape": "line"}})
    eng.run_for(20)
    res = eng.execute({"type": "swarm_goto", "params": {"position": [300, 0, 40]}})
    assert res.success
    eng.run_for(80)
    snap = eng.coordinator.snapshot()["formation"]
    ref = snap["reference_position"]
    assert math.hypot(ref["x"] - 300, ref["y"]) < 1.0 and abs(ref["z"] - 40) < 0.5
    assert abs(snap["heading"] - 90.0) < 2.0                        # travelling East
    centroid = eng.swarm.positions().mean(axis=0)
    assert np.linalg.norm(centroid - np.array([300, 0, 40])) < 3.0


def test_leader_follower_tracks_the_leader():
    eng = airborne_engine(5)
    res = eng.execute({"type": "set_formation", "params": {"shape": "v", "reference": "leader", "leader_id": 1}})
    assert res.success and res.data["leader"] == 1
    leader = eng.swarm.get(1)
    assert leader.flight_mode != FlightMode.FORMATION              # the leader is flown, not slotted
    eng.execute({"type": "swarm_goto", "params": {"position": [0, 250, 30]}})   # moves the leader
    eng.run_for(70)
    assert np.linalg.norm(leader.position - np.array([0, 250, 30])) < 1.5
    snap = eng.coordinator.snapshot()["formation"]
    assert snap["max_slot_error"] < 2.0 and abs(snap["heading"] - 0.0) < 3.0  # travelled North
    followers = [d for d in eng.swarm if d.id != 1]
    assert all(d.position[1] < leader.position[1] for d in followers)      # V trails behind the leader


def test_leader_loss_falls_back_to_virtual_reference_when_promotion_is_off():
    eng = airborne_engine(4, swarm__leader_promotion=False)
    eng.execute({"type": "set_formation", "params": {"shape": "line", "reference": "leader", "leader_id": 2}})
    eng.run_for(10)
    eng.execute({"type": "land", "drone_ids": [2]})
    eng.run_for(20)
    assert eng.coordinator.formation.reference_mode == "virtual"
    assert any(e.kind == "leader_lost" for e in eng.events.recent(limit=500))
    assert eng.coordinator.mode == SwarmMode.FORMATION


def test_operator_command_removes_member_and_release_ends_formation():
    eng = airborne_engine(5)
    eng.execute({"type": "set_formation", "params": {"shape": "circle"}})
    eng.run_for(15)
    eng.execute({"type": "hover", "drone_ids": [3]})
    eng.run_for(2)
    assert 3 not in eng.coordinator.snapshot()["formation"]["members"]
    res = eng.execute({"type": "release_swarm"})
    assert res.success
    eng.step()
    assert eng.coordinator.mode == SwarmMode.BASIC
    assert all(d.flight_mode == FlightMode.HOVER for d in eng.swarm)


def test_battery_failsafe_takes_member_out_of_formation():
    eng = airborne_engine(4)
    eng.execute({"type": "set_formation", "params": {"shape": "grid"}})
    eng.run_for(5)
    d = eng.swarm.get(2)
    d.battery.energy_wh = d.battery.capacity_wh * 0.195
    eng.run_for(1)
    assert d.flight_mode == FlightMode.RTL and d.swarm_behavior is None
    assert 2 not in eng.coordinator.snapshot()["formation"]["members"]


# ----------------------------------------------------------------------------- flocking

def test_flocking_reaches_goal_cohesive_and_separated():
    eng = airborne_engine(12)
    res = eng.execute({"type": "start_flocking", "params": {"goal": [200, 150, 35]}})
    assert res.success and eng.coordinator.mode == SwarmMode.FLOCKING
    min_sep = np.inf
    for _ in range(int(60 * eng.config.simulation.simulation_rate)):
        eng.step()
        min_sep = min(min_sep, pairwise_min(eng.swarm.positions()))
    pos = eng.swarm.positions()
    centroid = pos.mean(axis=0)
    assert np.linalg.norm(centroid[:2] - np.array([200, 150])) < 20
    assert np.max(np.linalg.norm(pos - centroid, axis=1)) < eng.config.flocking.perception_radius   # cohesive
    assert min_sep > eng.config.swarm.separation_distance                                         # separated
    assert eng.execute({"type": "set_flocking_weights", "params": {"cohesion": 2.0}}).success
    assert eng.coordinator.flocking.cfg.cohesion_weight == 2.0
    assert eng.config.flocking.cohesion_weight != 2.0          # runtime change does not leak into the config


# ----------------------------------------------------------------------------- collision avoidance

def _head_on(avoid: bool):
    eng = SimulationEngine(make_config(simulation__drone_count=2, home__pad_layout="line", home__pad_spacing=40.0),
                           record=False)
    eng.execute({"type": "set_avoidance", "params": {"enabled": avoid}})
    eng.execute({"type": "takeoff", "params": {"altitude": 25}})
    eng.run_for(8)
    a, b = eng.swarm.drones
    pa, pb = a.position.copy(), b.position.copy()
    a.goto(pb + (pb - pa) * 1.5, speed=10)
    b.goto(pa - (pb - pa) * 1.5, speed=10)
    min_sep = np.inf
    for _ in range(int(25 * eng.config.simulation.simulation_rate)):
        eng.step()
        min_sep = min(min_sep, float(np.linalg.norm(a.position - b.position)))
    return eng, min_sep


def test_head_on_conflict_is_resolved():
    eng, min_sep = _head_on(True)
    assert min_sep > eng.config.swarm.separation_distance
    assert eng.swarm.collision_monitor.total_collisions == 0
    a, b = eng.swarm.drones
    assert a.flight_mode == FlightMode.HOVER and b.flight_mode == FlightMode.HOVER   # both still arrived


def test_head_on_without_avoidance_collides():
    eng, min_sep = _head_on(False)
    assert min_sep < eng.config.swarm.collision_distance
    assert eng.swarm.collision_monitor.total_collisions >= 1


def test_crossing_group_goto_with_avoidance_has_no_collisions():
    eng = airborne_engine(16)
    for d in eng.swarm:                       # every drone flies through the centre to the mirrored point
        d.goto((-d.position[0] * 3, -d.position[1] * 3, 30))
    eng.run_for(45)
    assert eng.swarm.collision_monitor.total_collisions == 0


# ----------------------------------------------------------------------------- commands

@pytest.mark.parametrize("command,fragment", [
    ({"type": "set_formation", "params": {"shape": "hexagon"}}, "shape must be"),
    ({"type": "set_formation", "params": {"spacing": 2}}, "spacing"),
    ({"type": "swarm_goto", "params": {"position": [1, 2, 3]}}, "no formation"),
    ({"type": "set_avoidance", "params": {}}, "enabled"),
    ({"type": "set_flocking_weights", "params": {}}, "at least one"),
])
def test_swarm_command_validation(command, fragment):
    eng = airborne_engine(3)
    res = eng.execute(command)
    assert not res.success and fragment in res.message


def test_custom_formation_from_config():
    eng = airborne_engine(6)
    res = eng.execute({"type": "set_formation", "params": {"shape": "custom"}})
    assert res.success
    eng.run_for(30)
    assert eng.coordinator.snapshot()["formation"]["max_slot_error"] < 1.5

    none = airborne_engine(3, swarm__custom_formation=[])
    res = none.execute({"type": "set_formation", "params": {"shape": "custom"}})
    assert not res.success and "no custom formation" in res.message


def test_formation_requires_airborne_drones(engine):
    res = engine.execute({"type": "set_formation", "params": {"shape": "v"}})
    assert not res.success and "take off first" in res.message
