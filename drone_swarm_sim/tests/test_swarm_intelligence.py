"""Stage 1: ORCA avoidance, hard separation floor, synchronized transitions, altitude layers,
runtime custom formations and leader promotion."""

import numpy as np
import pytest

from algorithms.collision_avoidance import CollisionAvoidance
from algorithms.formation import assign_slots, formation_offsets, paths_cross, smoothstep
from algorithms.orca import linear_program3, orca_planes
from simulation.config import ConfigError, config_from_dict
from simulation.engine import SimulationEngine
from simulation.types import CommStatus, FlightMode

from .conftest import make_config


def airborne_engine(n: int, altitude: float = 30.0, **overrides) -> SimulationEngine:
    eng = SimulationEngine(make_config(simulation__drone_count=n, **overrides), record=False)
    eng.execute({"type": "takeoff", "params": {"altitude": altitude}})
    eng.run_for(10)
    return eng


# ----------------------------------------------------------------------------- ORCA maths

def test_orca_head_on_half_plane_points_sideways():
    # Agent at the origin flying East at 5 m/s, neighbour 20 m East flying West at 5 m/s.
    points, normals = orca_planes(np.array([[20.0, 0, 0]]), np.array([[10.0, 0, 0]]), np.array([[5.0, 0, 0]]),
                                  radius=7.0, tau=3.0, recovery_time=0.5, share=np.array([0.5]))
    n = normals[0]
    assert abs(np.linalg.norm(n) - 1.0) < 1e-9
    assert abs(n[0]) < 1e-9 and n[1] < 0            # exactly head-on: keep right (South when flying East)
    assert (np.array([5.0, 0, 0]) - points[0]) @ n < 0  # the current velocity is not allowed


def test_orca_constraint_is_inactive_when_diverging():
    points, normals = orca_planes(np.array([[20.0, 0, 0]]), np.array([[-10.0, 0, 0]]), np.array([[-5.0, 0, 0]]),
                                  radius=7.0, tau=3.0, recovery_time=0.5, share=np.array([0.5]))
    assert (np.array([-5.0, 0, 0]) - points[0]) @ normals[0] >= 0


def test_linear_program_keeps_feasible_preference_and_projects_otherwise():
    plane = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]         # v_y >= 0
    v, fail = linear_program3([plane], 15.0, (3.0, 2.0, 0.0))
    assert fail == 1 and np.allclose(v, (3.0, 2.0, 0.0))
    v, fail = linear_program3([plane], 15.0, (3.0, -2.0, 0.0))
    assert fail == 1 and np.allclose(v, (3.0, 0.0, 0.0))
    v, fail = linear_program3([plane], 2.0, (30.0, -2.0, 0.0))  # speed sphere respected
    assert np.linalg.norm(v) <= 2.0 + 1e-9 and v[1] >= -1e-9


def test_linear_program_reports_infeasibility():
    planes = [[0.0, 5.0, 0.0, 0.0, 1.0, 0.0],      # v_y >= 5
              [0.0, -5.0, 0.0, 0.0, -1.0, 0.0]]    # v_y <= -5
    _, fail = linear_program3(planes, 15.0, (0.0, 0.0, 0.0))
    assert fail == 1


# ----------------------------------------------------------------------------- hard separation floor

def _ram(eng: SimulationEngine, speed: float, seconds: float = 15.0) -> float:
    a, b = eng.swarm.drones[:2]
    lowest = np.inf
    for _ in range(int(seconds * eng.config.simulation.simulation_rate)):
        d = b.position - a.position
        u = d / np.linalg.norm(d)
        a.set_velocity(u * speed)
        b.set_velocity(-u * speed)
        eng.step()
        lowest = min(lowest, float(np.linalg.norm(a.position - b.position)))
    return lowest


@pytest.mark.parametrize("method", ["orca", "potential_field"])
def test_offboard_ramming_never_breaks_min_separation(method):
    eng = airborne_engine(2, 40.0, home__pad_layout="line", home__pad_spacing=60.0, swarm__avoidance_method=method)
    lowest = _ram(eng, 15.0)
    assert lowest >= eng.config.swarm.min_separation
    assert eng.swarm.collision_monitor.total_hard_violations == 0


def test_safety_filter_alone_guarantees_the_floor(monkeypatch):
    """With ORCA and the potential field disabled, the braking-distance filter still holds the floor."""
    monkeypatch.setattr(CollisionAvoidance, "_orca", lambda self, ctx, v, *a: v)
    monkeypatch.setattr(CollisionAvoidance, "_potential_field", lambda self, ctx, v, *a: v)
    eng = airborne_engine(2, 40.0, home__pad_layout="line", home__pad_spacing=60.0)
    lowest = _ram(eng, 15.0)
    floor = eng.config.swarm.min_separation
    assert floor <= lowest < floor + 1.0        # it brakes as late as is safe, not earlier
    # A swarm imploding onto its centroid at full speed.
    eng = airborne_engine(16)
    for _ in range(12 * 30):
        c = eng.swarm.positions().mean(axis=0)
        for d in eng.swarm:
            v = c - d.position
            v[2] = 0.0
            nv = np.linalg.norm(v)
            d.set_velocity(v / nv * 15.0 if nv > 1e-6 else [0, 0, 0])
        eng.step()
    assert eng.swarm.collision_monitor.lowest_separation >= floor
    assert eng.swarm.collision_monitor.total_hard_violations == 0


def test_orca_keeps_more_clearance_than_potential_field_in_dense_crossing():
    lowest = {}
    for method in ("orca", "potential_field"):
        eng = airborne_engine(16, swarm__avoidance_method=method)
        for d in eng.swarm:
            d.goto((-d.position[0] * 3, -d.position[1] * 3, 30))
        eng.run_for(40)
        mon = eng.swarm.collision_monitor
        assert mon.total_collisions == 0 and mon.total_hard_violations == 0
        assert all(d.flight_mode == FlightMode.HOVER for d in eng.swarm)          # everyone still arrives
        lowest[method] = mon.lowest_separation
    assert lowest["orca"] > lowest["potential_field"]
    assert lowest["orca"] > eng.config.swarm.separation_distance


def test_min_separation_breach_is_counted_and_logged():
    eng = airborne_engine(2, 40.0, home__pad_layout="line", home__pad_spacing=60.0)
    eng.execute({"type": "set_avoidance", "params": {"enabled": False}})   # operator override: no protection
    _ram(eng, 8.0, seconds=6.0)
    mon = eng.swarm.collision_monitor
    assert mon.total_hard_violations >= 1
    assert any(e.kind == "min_separation_breach" for e in eng.events.recent(limit=500))
    assert eng.swarm.summary()["min_separation_breaches"] == mon.total_hard_violations


def test_set_avoidance_switches_method_and_validates():
    eng = airborne_engine(2)
    res = eng.execute({"type": "set_avoidance", "params": {"method": "potential_field"}})
    assert res.success and eng.coordinator.avoidance.method == "potential_field"
    assert eng.coordinator.snapshot()["avoidance"]["method"] == "potential_field"
    res = eng.execute({"type": "set_avoidance", "params": {"method": "magic"}})
    assert not res.success and "method" in res.message
    assert any(e.kind == "set_avoidance" for e in eng.events.recent(limit=100))   # commands are logged


# ----------------------------------------------------------------------------- synchronized transitions

def test_hungarian_assignment_gives_non_crossing_synchronized_paths():
    rng = np.random.default_rng(3)
    start = formation_offsets("grid", 16, 15.0) + rng.normal(0, 1.0, (16, 3)) * [1, 1, 0]
    goal = formation_offsets("circle", 16, 15.0)
    best = goal[assign_slots(start, goal)]
    assert not paths_cross(start, best, 5.0)
    # A poor (reversed) assignment does cross - the check is not vacuous.
    assert paths_cross(start, goal[::-1], 5.0)


def test_smoothstep_profile():
    assert smoothstep(0.0) == (0.0, 0.0) and smoothstep(1.0) == (1.0, 0.0)
    s, ds = smoothstep(0.5)
    assert s == pytest.approx(0.5) and ds == pytest.approx(1.5)


def test_transition_starts_where_drones_are_and_finishes_together():
    eng = airborne_engine(9)
    eng.execute({"type": "set_formation", "params": {"shape": "line"}})
    eng.run_for(30)
    assert eng.execute({"type": "set_formation", "params": {"shape": "circle"}}).success
    eng.step()
    snap = eng.coordinator.snapshot()["formation"]
    assert snap["transition_s"] > 1.0 and snap["transition_progress"] < 0.1
    assert snap["max_slot_error"] < 0.5                 # no jump: targets start at the drones
    errors = []
    while eng.coordinator.formation.transition_progress < 1.0:
        eng.step()
        errors.append(eng.coordinator.formation.max_error)
    assert max(errors) < 3.0                            # drones track the moving slots closely
    eng.run_for(5)
    assert eng.coordinator.formation.max_error < 0.5


def test_direct_transition_mode_still_available():
    eng = airborne_engine(6, swarm__formation_transition="direct")
    eng.execute({"type": "set_formation", "params": {"shape": "grid"}})
    eng.run_for(30)
    snap = eng.coordinator.snapshot()["formation"]
    assert snap["transition_s"] == 0.0 and snap["max_slot_error"] < 1.5


# ----------------------------------------------------------------------------- altitude layers, heading, custom

def test_altitude_layers_offsets():
    off = formation_offsets("line", 6, 15.0, layers=3, layer_spacing=8.0)
    assert sorted(set(np.round(off[:, 2], 6))) == [-8.0, 0.0, 8.0]
    np.testing.assert_allclose(off.mean(axis=0), 0.0, atol=1e-9)


def test_formation_flies_in_altitude_layers():
    eng = airborne_engine(6)
    res = eng.execute({"type": "set_formation", "params": {"shape": "circle", "layers": 2, "layer_spacing": 10}})
    assert res.success and "2 layers" in res.message
    eng.run_for(30)
    z = np.sort(eng.swarm.positions()[:, 2])
    assert z[-1] - z[0] == pytest.approx(10.0, abs=1.0)
    assert eng.coordinator.snapshot()["formation"]["layers"] == 2
    assert not eng.execute({"type": "set_formation", "params": {"layers": 2.5}}).success
    assert not eng.execute({"type": "set_formation", "params": {"layer_spacing": 1}}).success


def test_configured_formation_heading_is_kept_while_moving():
    eng = airborne_engine(5, swarm__formation_heading=90.0)
    eng.execute({"type": "set_formation", "params": {"shape": "line"}})
    eng.run_for(10)
    eng.execute({"type": "swarm_goto", "params": {"position": [0, 200, 30]}})     # travel North
    eng.run_for(50)
    snap = eng.coordinator.snapshot()["formation"]
    assert snap["heading_locked"] and abs(snap["heading"] - 90.0) < 1.0           # still facing East
    eng.execute({"type": "swarm_goto", "params": {"position": [0, 250, 30], "heading": 180}})
    eng.run_for(10)
    assert abs(eng.coordinator.snapshot()["formation"]["heading"] - 180.0) < 1.0


def test_custom_formation_from_runtime_offsets():
    eng = airborne_engine(4)
    offsets = [[0, 0, 0], [-12, 12, 0], [-12, -12, 0], [-24, 0, 5]]
    res = eng.execute({"type": "set_formation", "params": {"shape": "custom", "offsets": offsets}})
    assert res.success
    eng.run_for(30)
    assert eng.coordinator.snapshot()["custom_offsets"] == [[float(x) for x in p] for p in offsets]
    rel = eng.swarm.positions() - eng.swarm.positions().mean(axis=0)
    expected = formation_offsets("custom", 4, 15.0, custom=offsets)
    got = np.sort(np.round(np.linalg.norm(rel, axis=1), 0))
    assert np.allclose(got, np.sort(np.round(np.linalg.norm(expected, axis=1), 0)), atol=1.5)


@pytest.mark.parametrize("params,fragment", [
    ({"shape": "custom", "offsets": [[0, 0, 0], [1, 0, 0]]}, "apart"),
    ({"shape": "custom", "offsets": "nope"}, "non-empty list"),
    ({"shape": "custom", "offsets": [[0, 0]]}, "three finite numbers"),
    ({"shape": "v", "offsets": [[0, 0, 0], [20, 0, 0]]}, "only be used with shape 'custom'"),
])
def test_custom_offsets_are_validated(params, fragment):
    eng = airborne_engine(2)
    res = eng.execute({"type": "set_formation", "params": params})
    assert not res.success and fragment in res.message


# ----------------------------------------------------------------------------- leader promotion

def test_leader_failure_promotes_next_drone_and_formation_continues():
    eng = airborne_engine(5)
    eng.execute({"type": "set_formation", "params": {"shape": "v", "reference": "leader", "leader_id": 1}})
    eng.execute({"type": "swarm_goto", "params": {"position": [0, 300, 30]}})
    eng.run_for(15)
    eng.execute({"type": "land", "drone_ids": [1]})          # the leader leaves the flight
    eng.run_for(1)
    form = eng.coordinator.formation
    assert form.reference_mode == "leader" and form.leader_id not in (None, 1)
    new_leader = eng.swarm.get(form.leader_id)
    assert new_leader.flight_mode == FlightMode.GOTO           # inherited the old leader's goto
    assert np.allclose(new_leader.target_position, [0, 300, 30])
    events = [e for e in eng.events.recent(limit=500) if e.kind == "leader_promoted"]
    assert len(events) == 1 and events[0].data["old_leader"] == 1 and events[0].data["new_leader"] == form.leader_id
    eng.run_for(60)
    assert np.linalg.norm(new_leader.position - np.array([0, 300, 30])) < 2.0
    assert eng.coordinator.snapshot()["formation"]["max_slot_error"] < 2.0
    assert eng.swarm.collision_monitor.total_hard_violations == 0


def test_leader_comm_loss_promotes_a_follower():
    eng = airborne_engine(4)
    eng.execute({"type": "set_formation", "params": {"shape": "line", "reference": "leader", "leader_id": 2}})
    eng.run_for(10)
    eng.swarm.get(2).comm_status = CommStatus.LOST
    eng.step()
    assert eng.coordinator.formation.leader_id != 2
    assert any(e.kind == "leader_promoted" and "communication lost" in e.message for e in eng.events.recent(limit=200))


def test_successive_leader_losses_until_one_drone_left():
    eng = airborne_engine(3)
    eng.execute({"type": "set_formation", "params": {"shape": "column", "reference": "leader", "leader_id": 1}})
    eng.run_for(10)
    eng.execute({"type": "land", "drone_ids": [1]})
    eng.run_for(1)
    second = eng.coordinator.formation.leader_id
    eng.execute({"type": "land", "drone_ids": [second]})
    eng.run_for(1)
    # The last drone was promoted too; with no followers left the formation ends cleanly.
    eng.run_for(2)
    assert eng.coordinator.formation.promotions == 2
    assert not eng.coordinator.formation.active


# ----------------------------------------------------------------------------- configuration

@pytest.mark.parametrize("swarm", [
    {"min_separation": 0.5},                                  # below collision distance
    {"min_separation": 6.0},                                  # above separation distance
    {"avoidance_method": "rvo9"},
    {"formation_altitude_layers": 0},
    {"formation_layer_spacing": 1.0},
    {"formation_transition": "teleport"},
    {"orca_time_horizon": 0.1},
    {"custom_formation": [[0, 0, 0], [1, 0, 0]]},             # slots closer than separation_distance
])
def test_swarm_config_validation(swarm):
    with pytest.raises(ConfigError):
        config_from_dict({"swarm": swarm})


def test_default_config_has_stage1_keys():
    cfg = make_config()
    sw = cfg.swarm
    assert sw.collision_distance < sw.min_separation <= sw.separation_distance
    assert sw.avoidance_method == "orca" and sw.leader_promotion
