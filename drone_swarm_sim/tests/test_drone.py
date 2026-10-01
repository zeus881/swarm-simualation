import math

import numpy as np
import pytest

from simulation.drone import Drone, heading_to_yaw, yaw_to_heading
from simulation.environment import Environment
from simulation.events import EventBus
from simulation.geo import GeoReference
from simulation.types import BatteryState, FlightMode, HealthStatus

from .conftest import make_config

DT = 1 / 30


class Harness:
    def __init__(self, battery=100.0, **overrides):
        self.cfg = make_config(**overrides)
        self.env = Environment(self.cfg, np.random.default_rng(0))
        self.events = EventBus()
        self.t = 0.0
        self.drone = Drone(1, self.cfg, self.env, GeoReference(47.0, 8.0, 400.0), home=(0, 0, 0),
                           initial_battery=battery, events=self.events)

    def run(self, seconds, check=None):
        for _ in range(int(round(seconds / DT))):
            before = self.drone.position.copy()
            self.drone.update(DT, self.t, self.env.wind.velocity_at(self.drone.position))
            self.t += DT
            if check:
                check(before, self.drone.position)

    def kinds(self):
        return [e.kind for e in self.events.recent(limit=10_000)]


def test_takeoff_reaches_altitude_and_hovers():
    h = Harness()
    assert h.drone.takeoff(15).success
    assert h.drone.armed and h.drone.flight_mode == FlightMode.TAKEOFF
    h.run(10)
    assert h.drone.flight_mode == FlightMode.HOVER
    assert abs(h.drone.position[2] - 15) < 0.5
    assert "takeoff_complete" in h.kinds()


def test_goto_moves_smoothly_without_teleporting():
    h = Harness()
    h.drone.takeoff(20)
    h.run(8)
    assert h.drone.goto((80, 40, 30)).success
    vmax = h.cfg.drone.max_horizontal_speed + h.cfg.drone.max_climb_rate

    def no_teleport(before, after):
        assert np.linalg.norm(after - before) <= vmax * DT + 1e-6

    h.run(25, no_teleport)
    assert np.linalg.norm(h.drone.position - np.array([80, 40, 30])) < h.cfg.drone.acceptance_radius
    assert h.drone.flight_mode == FlightMode.HOVER
    assert "arrived" in h.kinds()
    # Heading followed the direction of travel (compass bearing of (80, 40) is ~63 deg).
    assert abs(h.drone.heading - math.degrees(math.atan2(80, 40))) < 5


def test_goto_is_clamped_to_geofence():
    h = Harness()
    h.drone.takeoff(20)
    h.run(6)
    result = h.drone.goto((5000, 0, 1000))
    assert result.success and "clamped" in result.message
    assert h.drone.target_position[0] == pytest.approx(1000) and h.drone.target_position[2] == pytest.approx(300)


def test_commands_rejected_on_ground():
    h = Harness()
    assert not h.drone.goto((10, 0, 10)).success
    assert not h.drone.set_velocity((1, 0, 0)).success
    assert not h.drone.return_to_home().success
    assert not h.drone.goto((float("nan"), 0, 10)).success


def test_land_touches_down_softly_and_disarms():
    h = Harness()
    h.drone.takeoff(10)
    h.run(6)
    assert h.drone.land().success
    h.run(15)
    assert not h.drone.airborne and not h.drone.armed
    assert h.drone.flight_mode == FlightMode.DISARMED
    assert h.drone.body.last_touchdown_speed < h.cfg.drone.hard_landing_speed
    assert not h.drone.failed


def test_return_to_home_climbs_returns_and_lands():
    h = Harness()
    h.drone.takeoff(10)
    h.run(4)
    h.drone.goto((120, -60, 10))
    h.run(20)
    assert h.drone.return_to_home().success
    max_z = 0.0

    def track(_, after):
        nonlocal max_z
        max_z = max(max_z, after[2])

    h.run(60, track)
    assert max_z >= h.cfg.drone.rtl_altitude - 1.0                # climbed to RTL altitude first
    assert np.linalg.norm(h.drone.position[:2]) < 1.0            # landed on its home pad
    assert h.drone.flight_mode == FlightMode.DISARMED


def test_offboard_velocity_and_timeout_failsafe():
    h = Harness()
    h.drone.takeoff(20)
    h.run(8)
    for _ in range(90):   # 3 s of setpoints at 30 Hz
        h.drone.set_velocity((5, 0, 0))
        h.run(DT)
    assert h.drone.flight_mode == FlightMode.OFFBOARD
    assert h.drone.velocity[0] > 4.0
    h.run(1.0)   # stream stops
    assert h.drone.flight_mode == FlightMode.HOVER
    assert "offboard_timeout" in h.kinds()


def test_low_battery_triggers_rtl_then_emergency():
    # ~0.5-0.8 %/s drain: the RTL threshold (20 %) is crossed during the outbound leg.
    h = Harness(battery=60.0, battery__drain_multiplier=10.0)
    h.drone.takeoff(20)
    h.run(6)
    h.drone.battery.energy_wh = h.drone.battery.capacity_wh * 0.215
    assert h.drone.goto((150, 0, 20)).success
    h.run(5)
    assert h.drone.battery.state == BatteryState.RETURN_HOME
    assert h.drone.flight_mode == FlightMode.RTL
    assert "battery_failsafe" in h.kinds()
    h.run(25)
    assert h.drone.battery.state in (BatteryState.EMERGENCY, BatteryState.DEPLETED)
    assert h.drone.flight_mode in (FlightMode.EMERGENCY, FlightMode.DISARMED)
    assert h.drone.health in (HealthStatus.CRITICAL, HealthStatus.FAILED)


def test_takeoff_rejected_with_low_battery():
    h = Harness(battery=15.0)
    assert not h.drone.takeoff().success


def test_emergency_stop_land_and_kill():
    h = Harness()
    h.drone.takeoff(20)
    h.run(8)
    h.drone.emergency_stop()
    assert h.drone.flight_mode == FlightMode.EMERGENCY
    assert not h.drone.goto((0, 0, 30)).success
    h.run(15)
    assert h.drone.flight_mode == FlightMode.DISARMED and not h.drone.failed

    k = Harness(drone__emergency_stop_behavior="kill")
    k.drone.takeoff(20)
    k.run(8)
    k.drone.emergency_stop()
    k.run(5)
    assert not k.drone.airborne and k.drone.failed        # free fall from 20 m is a hard landing
    assert "hard landing" in k.drone.failure_reason


def test_auto_disarm_on_ground():
    h = Harness()
    h.drone.arm()
    h.run(h.cfg.drone.auto_disarm_delay + 0.5)
    assert not h.drone.armed and h.drone.flight_mode == FlightMode.DISARMED


def test_telemetry_schema():
    h = Harness()
    t = h.drone.get_telemetry().to_dict()
    for key in ("drone_id", "position", "velocity", "battery", "heading", "mode", "task", "communication", "gps",
                "roll", "pitch", "armed", "health", "target", "neighbors", "collision_state"):
        assert key in t
    assert t["gps"]["lat"] == pytest.approx(47.0) and t["gps"]["alt"] == pytest.approx(400.0, abs=1e-3)


def test_heading_conversions():
    assert yaw_to_heading(0.0) == pytest.approx(90.0)           # East
    assert yaw_to_heading(math.pi / 2) == pytest.approx(0.0)    # North
    assert yaw_to_heading(heading_to_yaw(225.0)) == pytest.approx(225.0)
