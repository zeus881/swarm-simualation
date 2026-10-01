"""Stage 3: operator-awareness telemetry (link / GPS / endurance), swarm health summary and alerts."""

import json

import numpy as np
import pytest

from simulation.alerts import AlertManager, AlertPriority, classify
from simulation.config import AlertsConfig, ConfigError, config_from_dict
from simulation.engine import SimulationEngine
from simulation.events import EventBus, EventCategory, Severity, SimEvent
from simulation.spatial import nearest_neighbor_distances

from .conftest import make_config


def engine(n: int = 2, **overrides) -> SimulationEngine:
    return SimulationEngine(make_config(simulation__drone_count=n, **overrides), record=False)


def airborne(n: int = 2, altitude: float = 30.0, **overrides) -> SimulationEngine:
    eng = engine(n, **overrides)
    eng.execute({"type": "takeoff", "params": {"altitude": altitude}})
    eng.run_for(10)
    return eng


# ----------------------------------------------------------------------------- telemetry

def test_hud_telemetry_fields_present():
    eng = airborne(1)
    t = eng.snapshot()["drones"][0]
    for key in ("link_quality", "gps_fix", "satellites", "hdop", "time_left_s", "battery_voltage", "roll", "pitch"):
        assert key in t, key
    assert t["gps_fix"] == "3D" and 0 < t["link_quality"] <= 100 and t["time_left_s"] > 60


def test_link_quality_falls_with_range():
    eng = airborne(1, communication__range=500.0)
    near = eng.swarm.get(1).link_quality
    eng.execute({"type": "goto", "params": {"position": [400, 0, 30], "speed": 15}})
    eng.run_for(40)
    far = eng.swarm.get(1).link_quality
    assert near > 95 and far < 40
    assert far == pytest.approx(100 * (1 - (np.linalg.norm(eng.swarm.get(1).position) / 500) ** 2), abs=2)


def test_flight_time_left_uses_actual_power():
    eng = airborne(1)
    d = eng.swarm.get(1)
    b = d.battery
    usable = b.energy_wh - b.capacity_wh * b.config.emergency / 100
    assert d.get_telemetry().time_left_s == pytest.approx(3600 * usable / b.power_avg_w, rel=0.01)
    hover_left = d.get_telemetry().time_left_s
    eng.execute({"type": "goto", "params": {"position": [800, 0, 30], "speed": 15}})   # fast flight costs more
    eng.run_for(15)
    assert d.get_telemetry().time_left_s < hover_left * 0.97


def test_swarm_health_summary():
    eng = engine(3)
    s = eng.swarm.summary()
    assert s["flight_time_left_s"] is None and s["min_separation"] is None           # nothing airborne
    assert s["min_battery"] == pytest.approx(min(d.battery.percent for d in eng.swarm), abs=0.1)
    eng.execute({"type": "takeoff", "params": {"altitude": 30}})
    eng.run_for(10)
    s = eng.swarm.summary()
    lefts = [d.get_telemetry().time_left_s for d in eng.swarm]
    assert s["flight_time_left_s"] == pytest.approx(min(lefts), abs=2)
    pos = eng.swarm.positions()
    true_min = min(np.linalg.norm(pos[i] - pos[j]) for i in range(3) for j in range(i + 1, 3))
    assert s["min_separation"] == pytest.approx(true_min, abs=0.05)                 # beyond the warning radius too
    assert all(d.nearest_distance is not None for d in eng.swarm)


def test_nearest_neighbor_distances_matches_brute_force():
    rng = np.random.default_rng(4)
    pts = rng.uniform(-100, 100, (40, 3))
    brute = np.array([min(np.linalg.norm(p - q) for k, q in enumerate(pts) if k != i) for i, p in enumerate(pts)])
    np.testing.assert_allclose(nearest_neighbor_distances(pts), brute)
    assert np.isinf(nearest_neighbor_distances(pts[:1])).all()


# ----------------------------------------------------------------------------- alerts

def ev(kind, severity=Severity.WARNING, category=EventCategory.DRONE, time=0.0, drone_id=1, message="m", **data):
    return SimEvent(time, category, kind, message, severity, drone_id, data)


def test_classification():
    assert classify(ev("collision", Severity.CRITICAL, EventCategory.COLLISION)) == AlertPriority.CRITICAL
    assert classify(ev("battery_state", Severity.WARNING)) == AlertPriority.WARNING
    assert classify(ev("mission", Severity.INFO, EventCategory.MISSION, message="mission 'x' completed")) == AlertPriority.INFO
    assert classify(ev("mission", Severity.INFO, EventCategory.MISSION, message="reached waypoint 2/4")) is None
    assert classify(ev("mode_change", Severity.INFO)) is None
    assert classify(ev("takeoff", Severity.WARNING, EventCategory.COMMAND)) is None       # command results are toasts


def _manager(**cfg) -> tuple[AlertManager, EventBus]:
    bus = EventBus()
    return AlertManager(AlertsConfig(**cfg), bus), bus


def test_dedup_and_expiry():
    am, bus = _manager(dedup_window_s=10, warning_ttl_s=60, info_ttl_s=20)
    for t in (0.0, 3.0, 6.0):
        bus.publish(ev("separation_violation", Severity.WARNING, EventCategory.COLLISION, time=t, drone_id=None,
                       drone_a=1, drone_b=2))
    assert len(am.active) == 1 and am.active[0].count == 3 and am.active[0].drone_ids == [1, 2]
    bus.publish(ev("separation_violation", Severity.WARNING, EventCategory.COLLISION, time=30.0, drone_id=None,
                   drone_a=1, drone_b=2))
    assert len(am.active) == 2                                              # outside the dedup window
    am.step(70.0)
    assert len(am.active) == 1                                              # first one expired (TTL from last repeat)
    am.step(95.0)
    assert am.active == [] and am.total[AlertPriority.WARNING] == 2 and len(am.history) == 2


def test_critical_waits_for_acknowledgement():
    am, bus = _manager(acknowledged_ttl_s=30)
    bus.publish(ev("emergency", Severity.CRITICAL, time=1.0))
    am.step(10_000.0)
    assert am.unacked_critical == 1 and am.active[0].requires_ack
    assert am.acknowledge([am.active[0].id], by="op1") == 1
    assert am.unacked_critical == 0 and am.active[0].ack_by == "op1"
    am.step(10_020.0)
    assert len(am.active) == 1                                              # acknowledged: still shown for a while
    am.step(10_031.0)
    assert am.active == []


def test_trimming_keeps_unacknowledged_criticals():
    am, bus = _manager(max_active=5, max_alerts=100)
    bus.publish(ev("failure", Severity.CRITICAL, drone_id=9))
    for k in range(10):
        bus.publish(ev("battery_state", Severity.WARNING, drone_id=k + 20))
    assert len(am.active) == 5 and any(a.kind == "failure" for a in am.active)


def test_geofence_breach_raises_critical_alert_and_ack_command():
    eng = airborne(1)
    p = eng.swarm.get(1).position
    eng.execute({"type": "set_geofence", "params": {"enabled": True, "action": "HOLD", "exclusions": [
        [[p[0] - 5, p[1] - 5], [p[0] + 5, p[1] - 5], [p[0] + 5, p[1] + 5], [p[0] - 5, p[1] + 5]]]}})
    eng.run_for(0.5)
    snap = eng.snapshot()["alerts"]
    crit = [a for a in snap["active"] if a["priority"] == "CRITICAL"]
    assert snap["unacked_critical"] == 1 and crit[0]["kind"] == "geofence_breach" and crit[0]["drone_ids"] == [1]
    assert not eng.execute({"type": "ack_alert", "params": {"ids": [999]}}).success
    assert not eng.execute({"type": "ack_alert", "params": {}}).success
    res = eng.execute({"type": "ack_alert", "params": {"ids": [crit[0]["id"]], "user": "alice"}})
    assert res.success and res.data["acknowledged"] == 1
    snap = eng.snapshot()["alerts"]
    assert snap["unacked_critical"] == 0 and snap["active"][0]["ack_by"] == "alice"
    assert any(e.kind == "ack_alert" for e in eng.events.recent(limit=50))   # acknowledgement is logged
    json.dumps(eng.snapshot(), allow_nan=False)


def test_ack_all_and_mission_info_alert():
    eng = engine(1)
    m = {"name": "hop", "waypoints": [{"x": 30, "y": 0, "alt": 15}, {"action": "LAND", "x": 30, "y": 0, "alt": 0}]}
    eng.execute({"type": "mission_start", "drone_ids": [1], "params": {"mission": m}})
    eng.run_for(80)
    infos = [a for a in eng.alerts.history if a["priority"] == "INFO"]      # INFO alerts expire; history keeps them
    assert any("started" in a["message"] for a in infos) and any("completed" in a["message"] for a in infos)
    eng.events.emit(EventCategory.DRONE, "failure", "motor 2 failed", severity=Severity.CRITICAL, drone_id=1)
    eng.events.emit(EventCategory.DRONE, "emergency", "descending", severity=Severity.CRITICAL, drone_id=1)
    assert eng.alerts.unacked_critical == 2
    res = eng.execute({"type": "ack_alert", "params": {"all": True}})
    assert res.success and res.data["acknowledged"] == 2 and eng.alerts.unacked_critical == 0


@pytest.mark.parametrize("alerts", [{"max_active": 1}, {"warning_ttl_s": 0}, {"max_alerts": 10, "max_active": 50},
                                    {"critical_repeat_s": 0.2}, {"dedup_window_s": -1}])
def test_alert_config_validation(alerts):
    with pytest.raises(ConfigError):
        config_from_dict({"alerts": alerts})


def test_world_info_carries_alert_repeat():
    assert engine(1).world_info()["alerts"]["critical_repeat_s"] == make_config().alerts.critical_repeat_s
