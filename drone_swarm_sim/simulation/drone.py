"""Simulated vehicle: autopilot mode state machine, guidance and failsafes.

A :class:`Drone` is deliberately *not* aware of formations or swarm logic.
It behaves like a PX4/ArduPilot vehicle: it accepts commands through
:class:`~simulation.drone_interface.DroneInterface`, turns its current mode
into a desired velocity (:meth:`Drone.compute_guidance`), and flies whatever
final velocity the swarm layer hands back (:meth:`Drone.integrate`).
Self-protection (battery failsafes, offboard timeout, geofence, hard-landing
detection) lives here because a real autopilot owns it too.

Mode state machine::

    DISARMED --arm--> ARMED --takeoff--> TAKEOFF --reached--> HOVER
                          ^                                   |  ^
                          |auto-disarm        goto/set_velocity |  | arrived / offboard timeout
                          |                                   v  |
    DISARMED <--touchdown-- LAND <--land-- {HOVER, GOTO, OFFBOARD, FORMATION}
    DISARMED <--touchdown-- RTL (CLIMB -> RETURN -> LAND)  <--rtl / battery failsafe--
    DISARMED <--touchdown-- EMERGENCY  <--emergency_stop / battery emergency--
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Sequence

import numpy as np

from .battery import BatteryModel
from .config import SimConfig
from .drone_interface import CommandResult, DroneInterface, DroneTelemetry
from .dynamics import DynamicsParams, QuadrotorDynamics, RigidBodyState, VelocityController, clip_velocity
from .environment import Environment
from .events import EventBus, EventCategory, Severity
from .geo import GeoReference
from .guidance import approach_velocity, hold_xy_velocity
from .types import (
    AIRBORNE_MODES,
    BatteryState,
    CollisionState,
    CommStatus,
    FlightMode,
    HealthStatus,
    Vector3,
    vec3,
)


class RtlPhase(StrEnum):
    CLIMB = "CLIMB"
    RETURN = "RETURN"
    LAND = "LAND"


def yaw_to_heading(yaw_rad: float) -> float:
    """ENU yaw (CCW from East, rad) -> compass heading (CW from North, deg)."""
    return (90.0 - math.degrees(yaw_rad)) % 360.0


def heading_to_yaw(heading_deg: float) -> float:
    """Compass heading (deg) -> ENU yaw (rad)."""
    return math.radians(90.0 - heading_deg)


_ZERO = np.zeros(3)


class Drone(DroneInterface):
    """Simulated multirotor implementing :class:`DroneInterface`."""

    def __init__(
        self,
        drone_id: int,
        config: SimConfig,
        environment: Environment,
        geo: GeoReference,
        *,
        home: Sequence[float],
        initial_battery: float = 100.0,
        events: EventBus | None = None,
    ) -> None:
        self.id = int(drone_id)
        self.name = f"D{self.id:02d}"
        self.cfg = config.drone
        self._env = environment
        self._geo = geo
        self._events = events

        self.params = DynamicsParams.from_config(self.cfg)
        self._dynamics = QuadrotorDynamics(self.params)
        self._vel_ctrl = VelocityController(self.params)
        self._decel = 0.5 * self.params.max_horizontal_accel  # braking used by guidance (margin for lag)

        self.home_position = np.array(home, dtype=np.float64)
        self.body = RigidBodyState(position=self.home_position.copy(), on_ground=True)
        self.battery = BatteryModel(config.battery, initial_battery)

        self.flight_mode = FlightMode.DISARMED
        self.armed = False
        self.comm_status = CommStatus.ONLINE
        self.collision_state = CollisionState.CLEAR
        self.nearest_distance: float | None = None
        self.neighbors: list[int] = []
        self.current_task = "IDLE"
        self.target_position: Vector3 | None = None
        # Swarm control (Phase 2): which swarm behaviour owns this drone while in FORMATION mode
        # ("formation" | "flocking"), and the point that behaviour is steering it to (telemetry only).
        self.swarm_behavior: str | None = None
        self.swarm_target: Vector3 | None = None

        self._hold: Vector3 | None = None
        self._route: list[Vector3] = []          # pass-through waypoints before target_position (goto_path)
        self._cruise_speed = self.cfg.cruise_speed
        self._heading_sp: float | None = None
        self._velocity_sp = np.zeros(3)
        self._velocity_sp_time = -math.inf
        self._rtl_phase: RtlPhase | None = None
        self._rtl_altitude = 0.0
        self._time = 0.0
        self._armed_since = 0.0
        self._motors_killed = False
        self._failed = False
        self.failure_reason = ""
        self._battery_state = self.battery.state
        self._battery_failsafes_done: set[BatteryState] = set()
        self.flight_time = 0.0
        self.distance_travelled = 0.0
        self.last_guidance = np.zeros(3)
        # Link / GNSS quality shown on the HUD. The swarm manager estimates link quality from the range
        # to the GCS; the communication and sensor models (Stage 4) drive all four.
        self.link_quality = 100.0
        self.comm_managed = False           # True once a communication model owns comm_status / link_quality
        self.wind_disturbance = np.zeros(3)  # extra local wind (injected gusts), added to the ambient wind
        # State estimate from the navigation filter (Stage 4); None without the sensor model.
        self.est_position: Vector3 | None = None
        self.est_velocity: Vector3 | None = None
        self.est_heading: float | None = None
        self.pos_sigma: float | None = None
        self.nav_from_estimate = config.sensors.enabled and config.sensors.control_source == "estimate"
        self.failures: set[str] = set()      # injected failures in force (instructor mode)
        self._thrust_scale = 1.0             # < 1 after a partial motor failure
        self.gps_fix = "3D"
        self.satellites = 14
        self.hdop = 0.7

    # ================================================================ properties
    @property
    def drone_id(self) -> int:
        return self.id

    @property
    def position(self) -> Vector3:
        return self.body.position

    @property
    def velocity(self) -> Vector3:
        return self.body.velocity

    @property
    def acceleration(self) -> Vector3:
        return self.body.acceleration

    @property
    def orientation(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) in radians; yaw is ENU (CCW from East)."""
        return self.body.roll, self.body.pitch, self.body.yaw

    @property
    def heading(self) -> float:
        return yaw_to_heading(self.body.yaw)

    @property
    def airborne(self) -> bool:
        return not self.body.on_ground

    @property
    def in_flight(self) -> bool:
        """Armed and in a flying mode (includes the ground roll of a takeoff)."""
        return self.armed and self.flight_mode in AIRBORNE_MODES

    @property
    def motors_on(self) -> bool:
        return self.in_flight and not self._motors_killed

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def nav_position(self) -> Vector3:
        """Position the autopilot believes it is at (Kalman estimate, or the truth without sensors)."""
        if self.nav_from_estimate and self.est_position is not None:
            return self.est_position
        return self.body.position

    @property
    def rtl_landing(self) -> bool:
        """True during the final descent of a return-to-home."""
        return self.flight_mode == FlightMode.RTL and self._rtl_phase == RtlPhase.LAND

    def inject_motor_failure(self, total: bool, thrust_left: float) -> str:
        """Instructor failure: a partial failure leaves ``thrust_left`` of the thrust and the autopilot
        starts an emergency descent; a total failure is a crash (motors off)."""
        self.failures.add("motor")
        if total:
            self.crash("total motor failure")
            return "total motor failure - vehicle falling"
        self._thrust_scale = thrust_left
        self._dynamics.set_thrust_scale(thrust_left)
        if self.in_flight and self.flight_mode != FlightMode.EMERGENCY:
            self._begin_emergency("motor failure", kill=False)
        return f"motor failure: {thrust_left * 100:.0f} % thrust left - emergency descent"

    def clear_motor_failure(self) -> None:
        self.failures.discard("motor")
        self._thrust_scale = 1.0
        self._dynamics.set_thrust_scale(1.0)

    def crash(self, reason: str) -> None:
        """Structural collision (e.g. with an obstacle): the vehicle fails and falls with motors off."""
        if self._failed:
            return
        self._fail(reason, category=EventCategory.COLLISION)
        if self.in_flight:
            self._begin_emergency(reason, kill=True)

    @property
    def goto_speed(self) -> float:
        """Cruise speed of the current/last goto [m/s]."""
        return self._cruise_speed

    @property
    def altitude_agl(self) -> float:
        p = self.body.position
        return float(p[2] - self._env.ground_height(p[0], p[1]))

    @property
    def health(self) -> HealthStatus:
        battery_state = self.battery.state
        if self._failed:
            return HealthStatus.FAILED
        if (
            self.flight_mode == FlightMode.EMERGENCY
            or battery_state in (BatteryState.EMERGENCY, BatteryState.DEPLETED)
            or self.collision_state == CollisionState.COLLISION
            or self.comm_status == CommStatus.LOST
        ):
            return HealthStatus.CRITICAL
        if (
            battery_state in (BatteryState.WARNING, BatteryState.RETURN_HOME)
            or self.collision_state in (CollisionState.WARNING, CollisionState.AVOIDANCE)
            or self.comm_status == CommStatus.DEGRADED
        ):
            return HealthStatus.WARNING
        return HealthStatus.OK

    # ================================================================ helpers
    def _emit(self, kind: str, message: str, severity: Severity = Severity.INFO,
              category: EventCategory = EventCategory.DRONE, **data) -> None:
        if self._events is not None:
            self._events.emit(category, kind, message, severity=severity, drone_id=self.id, time=self._time, **data)

    def _set_mode(self, mode: FlightMode, task: str | None = None, reason: str = "") -> None:
        if task is not None:
            self.current_task = task
        if mode == self.flight_mode:
            return
        old = self.flight_mode
        self.flight_mode = mode
        if mode != FlightMode.RTL:
            self._rtl_phase = None
        if old == FlightMode.FORMATION:
            # Any other command or failsafe takes the drone out of swarm control.
            self.swarm_behavior = None
            self.swarm_target = None
        self._emit("mode_change", f"{old} -> {mode}" + (f" ({reason})" if reason else ""), old=str(old), new=str(mode))

    def _ground(self, p: Vector3 | None = None) -> float:
        p = self.body.position if p is None else p
        return self._env.ground_height(p[0], p[1])

    def _gate_in_flight(self, action: str) -> CommandResult | None:
        if self._failed:
            return CommandResult.fail(f"{action} rejected: vehicle failed ({self.failure_reason})")
        if not self.in_flight:
            return CommandResult.fail(f"{action} rejected: not airborne")
        if self.flight_mode == FlightMode.EMERGENCY:
            return CommandResult.fail(f"{action} rejected: emergency active")
        return None

    @staticmethod
    def _as_vec3(value: Sequence[float], name: str) -> Vector3:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must be three finite numbers")
        return arr

    # ================================================================ commands (DroneInterface)
    def arm(self) -> CommandResult:
        if self.armed:
            return CommandResult.ok("already armed")
        if self._failed:
            return CommandResult.fail(f"arm rejected: vehicle failed ({self.failure_reason})")
        if not self.body.on_ground:
            return CommandResult.fail("arm rejected: vehicle is not on the ground")
        if self.battery.state in (BatteryState.EMERGENCY, BatteryState.DEPLETED):
            return CommandResult.fail("arm rejected: battery too low")
        self.armed = True
        self._armed_since = self._time
        self._motors_killed = False
        self._vel_ctrl.reset()
        self._set_mode(FlightMode.ARMED, task="IDLE")
        return CommandResult.ok("armed")

    def disarm(self) -> CommandResult:
        if not self.armed:
            return CommandResult.ok("already disarmed")
        if self.airborne:
            return CommandResult.fail("disarm rejected: vehicle is airborne (use emergency_stop)")
        self.armed = False
        self.target_position = None
        self._hold = None
        self._set_mode(FlightMode.DISARMED, task="IDLE")
        return CommandResult.ok("disarmed")

    def takeoff(self, altitude: float | None = None) -> CommandResult:
        alt = self.cfg.takeoff_altitude if altitude is None else float(altitude)
        if not self.cfg.min_altitude <= alt <= self._env.max_altitude:
            return CommandResult.fail(
                f"takeoff rejected: altitude must be in [{self.cfg.min_altitude}, {self._env.max_altitude}] m")
        if self.in_flight:
            return CommandResult.fail("takeoff rejected: already airborne")
        if self.battery.state in (BatteryState.RETURN_HOME, BatteryState.EMERGENCY, BatteryState.DEPLETED):
            return CommandResult.fail(f"takeoff rejected: battery {self.battery.percent:.0f}% below return threshold")
        if not self.armed:
            result = self.arm()
            if not result.success:
                return result
        p = self.body.position
        self.target_position = vec3(p[0], p[1], self._ground() + alt)
        self._heading_sp = None
        self._set_mode(FlightMode.TAKEOFF, task="TAKEOFF")
        return CommandResult.ok(f"taking off to {alt:.1f} m")

    def land(self) -> CommandResult:
        if not self.in_flight:
            if self.armed:
                self.disarm()
            return CommandResult.ok("already on the ground")
        if self.flight_mode == FlightMode.EMERGENCY:
            return CommandResult.fail("land rejected: emergency active")
        self._hold = self.body.position.copy()
        self._set_mode(FlightMode.LAND, task="LAND")
        return CommandResult.ok("landing")

    def hover(self) -> CommandResult:
        if (rejected := self._gate_in_flight("hover")) is not None:
            return rejected
        p = self.body.position.copy()
        p[2] = max(p[2], self._ground() + self.cfg.min_altitude)
        self._hold = p
        self._set_mode(FlightMode.HOVER, task="HOVER")
        return CommandResult.ok("holding position")

    def goto(self, position: Sequence[float], speed: float | None = None,
             heading: float | None = None) -> CommandResult:
        if (rejected := self._gate_in_flight("goto")) is not None:
            return rejected
        try:
            requested = self._as_vec3(position, "position")
        except ValueError as exc:
            return CommandResult.fail(f"goto rejected: {exc}")
        if speed is not None and not speed > 0:
            return CommandResult.fail("goto rejected: speed must be > 0")
        target = self._env.clamp(requested, min_agl=self.cfg.min_altitude)
        clamped = not np.allclose(target, requested, atol=1e-6)
        self._cruise_speed = min(float(speed), self.cfg.max_horizontal_speed) if speed else self.cfg.cruise_speed
        self._heading_sp = heading_to_yaw(heading) if heading is not None else None
        self.target_position = target
        self._route = []
        self._set_mode(FlightMode.GOTO, task="GOTO")
        note = " (clamped to geofence)" if clamped else ""
        return CommandResult.ok(f"goto ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}){note}")

    def goto_path(self, route: Sequence[Sequence[float]], speed: float | None = None,
                  heading: float | None = None) -> CommandResult:
        """Fly through intermediate waypoints without stopping, then stop on the last one (GOTO mode).

        Used by the obstacle path planner; an autopilot mission with pass-through waypoints does the same.
        """
        if not route:
            return CommandResult.fail("goto_path rejected: empty route")
        try:
            points = [self._env.clamp(self._as_vec3(p, "route point"), min_agl=self.cfg.min_altitude) for p in route]
        except ValueError as exc:
            return CommandResult.fail(f"goto_path rejected: {exc}")
        result = self.goto(points[-1], speed=speed, heading=heading)
        if result.success:
            self._route = points[:-1]
            if self._route:
                result = CommandResult.ok(f"{result.message} via {len(self._route)} waypoint(s)")
        return result

    @property
    def route(self) -> list[Vector3]:
        """Remaining pass-through waypoints of the current goto (empty for a direct goto)."""
        return list(self._route) if self.flight_mode == FlightMode.GOTO else []

    def set_velocity(self, velocity: Sequence[float], heading: float | None = None) -> CommandResult:
        if (rejected := self._gate_in_flight("set_velocity")) is not None:
            return rejected
        try:
            v = self._as_vec3(velocity, "velocity")
        except ValueError as exc:
            return CommandResult.fail(f"set_velocity rejected: {exc}")
        self._velocity_sp = clip_velocity(v, self.cfg.max_horizontal_speed, self.cfg.max_climb_rate,
                                          self.cfg.max_descent_rate)
        self._velocity_sp_time = self._time
        if heading is not None:
            self._heading_sp = heading_to_yaw(heading)
        self._set_mode(FlightMode.OFFBOARD, task="OFFBOARD" if self.flight_mode != FlightMode.OFFBOARD else None)
        return CommandResult.ok("velocity setpoint accepted")

    def set_heading(self, heading: float) -> CommandResult:
        if not self.armed:
            return CommandResult.fail("set_heading rejected: not armed")
        if not math.isfinite(heading):
            return CommandResult.fail("set_heading rejected: heading must be finite")
        self._heading_sp = heading_to_yaw(heading)
        return CommandResult.ok(f"heading {heading % 360:.0f} deg")

    def enter_swarm_control(self, behavior: str) -> CommandResult:
        """Hand velocity control to a swarm behaviour (formation / flocking).

        The drone holds its position until the behaviour's velocity pipeline stage takes over;
        any later operator command or failsafe leaves swarm control automatically.
        """
        if (rejected := self._gate_in_flight(behavior)) is not None:
            return rejected
        self._hold = self.body.position.copy()
        self._heading_sp = None
        self.swarm_behavior = behavior
        self.swarm_target = None
        self._set_mode(FlightMode.FORMATION, task=behavior.upper())
        self.current_task = behavior.upper()
        return CommandResult.ok(f"joined {behavior}")

    def return_to_home(self) -> CommandResult:
        if (rejected := self._gate_in_flight("return_to_home")) is not None:
            return rejected
        self._start_rtl("operator request")
        return CommandResult.ok("returning to home")

    def emergency_stop(self) -> CommandResult:
        if not self.armed:
            return CommandResult.ok("already disarmed")
        if not self.airborne:
            self.armed = False
            self._set_mode(FlightMode.DISARMED, task="IDLE", reason="emergency stop")
            return CommandResult.ok("disarmed on ground")
        kill = self.cfg.emergency_stop_behavior == "kill"
        self._begin_emergency("operator emergency stop", kill=kill)
        return CommandResult.ok("motors killed" if kill else "emergency descent")

    # ================================================================ failsafe helpers
    def _start_rtl(self, reason: str) -> None:
        p = self.body.position
        ground_home = self._ground(self.home_position)
        self._rtl_altitude = max(p[2], ground_home + self.cfg.rtl_altitude)
        horizontal = math.hypot(*(self.home_position[:2] - p[:2]))
        if horizontal <= self.cfg.acceptance_radius * 2:
            self._rtl_phase = RtlPhase.LAND
            self._hold = vec3(self.home_position[0], self.home_position[1], p[2])
        elif p[2] < self._rtl_altitude - 1.0:
            self._rtl_phase = RtlPhase.CLIMB
            self._hold = p.copy()
        else:
            self._rtl_phase = RtlPhase.RETURN
        self._heading_sp = None
        self._set_mode(FlightMode.RTL, task="RTL", reason=reason)

    def _begin_emergency(self, reason: str, *, kill: bool) -> None:
        self._motors_killed = kill
        self._hold = self.body.position.copy()
        self._set_mode(FlightMode.EMERGENCY, task="EMERGENCY", reason=reason)
        self._emit("emergency", reason + (" - motors killed" if kill else " - emergency descent"),
                   Severity.CRITICAL)

    def _fail(self, reason: str, category: EventCategory = EventCategory.DRONE) -> None:
        if self._failed:
            return
        self._failed = True
        self.failure_reason = reason
        self._emit("failure", reason, Severity.CRITICAL, category=category)

    # ================================================================ simulation loop
    def compute_guidance(self, t: float) -> Vector3:
        """Outer loop: turn the current mode into a desired velocity (ENU, m/s)."""
        self._time = t
        mode = self.flight_mode
        p = self.nav_position               # the autopilot navigates on its estimate (truth without sensors)
        cfg = self.cfg
        v = _ZERO

        if mode in (FlightMode.DISARMED, FlightMode.ARMED):
            v = _ZERO
        elif mode == FlightMode.TAKEOFF:
            target = self.target_position
            v = approach_velocity(p, target, cruise_speed=cfg.max_climb_rate, max_climb=cfg.max_climb_rate,
                                  max_descent=cfg.max_descent_rate, decel=self._decel, gain=cfg.position_gain)
            if abs(target[2] - p[2]) < 0.5:
                self._hold = target.copy()
                self._set_mode(FlightMode.HOVER, task="HOVER", reason="takeoff complete")
                self._emit("takeoff_complete", f"reached {target[2] - self._ground():.1f} m")
        elif mode == FlightMode.HOVER:
            v = self._approach(self._hold, cfg.cruise_speed)
        elif mode == FlightMode.GOTO and self._route:
            # Pass-through waypoint: full cruise speed towards it, switch early enough to turn smoothly.
            wp = self._route[0]
            delta = wp - p
            dist = math.sqrt(delta @ delta)
            switch = max(3.0 * cfg.acceptance_radius, 0.8 * self._cruise_speed)
            if dist < switch:
                self._route.pop(0)
                v = self._approach(self._route[0] if self._route else self.target_position, self._cruise_speed)
            else:
                v = clip_velocity(delta / dist * self._cruise_speed, cfg.max_horizontal_speed, cfg.max_climb_rate,
                                  cfg.max_descent_rate)
        elif mode == FlightMode.GOTO:
            target = self.target_position
            v = self._approach(target, self._cruise_speed)
            if np.linalg.norm(target - p) < cfg.acceptance_radius and np.linalg.norm(self.body.velocity) < 1.0:
                self._hold = target.copy()
                self._set_mode(FlightMode.HOVER, task="HOLD", reason="arrived")
                self._emit("arrived", f"arrived at ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
        elif mode == FlightMode.FORMATION:
            # The swarm pipeline stage overrides this; on its own the drone holds its slot/position.
            v = self._approach(self.swarm_target if self.swarm_target is not None else self._hold, cfg.cruise_speed)
        elif mode == FlightMode.OFFBOARD:
            if t - self._velocity_sp_time > cfg.offboard_timeout:
                self._hold = p.copy()
                self._set_mode(FlightMode.HOVER, task="HOVER", reason="setpoint timeout")
                self._emit("offboard_timeout", "velocity setpoint stream lost - holding position", Severity.WARNING)
                v = _ZERO
            else:
                v = self._velocity_sp
        elif mode == FlightMode.LAND:
            v = self._land_velocity(self._hold)
        elif mode == FlightMode.RTL:
            v = self._rtl_velocity()
        elif mode == FlightMode.EMERGENCY:
            if self._motors_killed:
                v = _ZERO
            else:
                v = vec3(0.0, 0.0, -cfg.emergency_descent_rate)

        if mode in (FlightMode.HOVER, FlightMode.GOTO, FlightMode.OFFBOARD, FlightMode.FORMATION):
            v = self._apply_geofence(v)
        self.last_guidance = np.array(v, dtype=np.float64)
        return self.last_guidance

    def _approach(self, target: Vector3, cruise: float) -> Vector3:
        c = self.cfg
        return approach_velocity(self.nav_position, target, cruise_speed=cruise, max_climb=c.max_climb_rate,
                                 max_descent=c.max_descent_rate, decel=self._decel, gain=c.position_gain)

    def _land_velocity(self, hold: Vector3) -> Vector3:
        p = self.body.position
        v = hold_xy_velocity(self.nav_position, hold, self.cfg.position_gain, 2.0)
        height = p[2] - self._ground()      # height above ground from the (range-finder) truth
        # Slow down during the final metres for a soft touchdown.
        v[2] = -min(self.cfg.land_speed, max(0.6, 0.5 * height))
        return v

    def _rtl_velocity(self) -> Vector3:
        p = self.nav_position
        home = self.home_position
        cruise_target = vec3(home[0], home[1], self._rtl_altitude)
        if self._rtl_phase == RtlPhase.CLIMB:
            target = vec3(self._hold[0], self._hold[1], self._rtl_altitude)
            if abs(p[2] - self._rtl_altitude) < 1.0:
                self._rtl_phase = RtlPhase.RETURN
                self._emit("rtl_phase", "RTL: returning at cruise altitude")
            return self._approach(target, self.cfg.cruise_speed)
        if self._rtl_phase == RtlPhase.RETURN:
            if math.hypot(home[0] - p[0], home[1] - p[1]) < self.cfg.acceptance_radius:
                self._rtl_phase = RtlPhase.LAND
                self._hold = vec3(home[0], home[1], p[2])
                self._emit("rtl_phase", "RTL: over home, landing")
            return self._approach(cruise_target, self.cfg.cruise_speed)
        return self._land_velocity(self._hold if self._hold is not None else home)

    def _apply_geofence(self, v: Vector3) -> Vector3:
        """Remove velocity components that would leave the flyable volume."""
        p = self.body.position
        env = self._env
        out = np.array(v, dtype=np.float64)
        ground = self._ground()
        if p[2] >= ground + env.max_altitude and out[2] > 0:
            out[2] = 0.0
        if self.flight_mode in (FlightMode.OFFBOARD, FlightMode.FORMATION) \
                and p[2] <= ground + self.cfg.min_altitude and out[2] < 0:
            out[2] = 0.0
        for axis in (0, 1):
            if (p[axis] <= env.bounds_min[axis] and out[axis] < 0) or (p[axis] >= env.bounds_max[axis] and out[axis] > 0):
                out[axis] = 0.0
        return out

    def _yaw_setpoint(self, v_cmd: Vector3) -> float | None:
        if self._heading_sp is not None:
            return self._heading_sp
        if math.hypot(v_cmd[0], v_cmd[1]) > 1.0:
            return math.atan2(v_cmd[1], v_cmd[0])   # face the direction of travel
        return None

    def integrate(self, v_cmd: Vector3, wind: Vector3, dt: float, ground_z: float | None = None) -> None:
        """Inner loop + physics + battery for one physics sub-step (``ground_z``: terrain height under the
        drone if the caller already knows it)."""
        body = self.body
        motors = self.motors_on
        if motors:
            accel_cmd = self._vel_ctrl.compute(v_cmd, body.velocity, dt)
        else:
            accel_cmd = _ZERO
            self._vel_ctrl.reset()
        was_on_ground = body.on_ground
        px, py, pz = body.position.tolist()
        self._dynamics.step(body, accel_cmd, self._yaw_setpoint(v_cmd), wind, dt, motors_on=motors,
                            ground_z=self._env.ground_height(px, py) if ground_z is None else ground_z)
        nx, ny, nz = body.position.tolist()
        self.distance_travelled += math.sqrt((nx - px) ** 2 + (ny - py) ** 2 + (nz - pz) ** 2)
        vx, vy, vz = body.velocity.tolist()
        wx, wy, wz = (float(w) for w in wind)
        grounded = body.on_ground
        ax, ay, az = body.acceleration.tolist()
        self.battery.update(
            dt,
            armed=self.armed,
            motors_on=motors,
            air_speed=0.0 if grounded else math.sqrt((vx - wx) ** 2 + (vy - wy) ** 2 + (vz - wz) ** 2),
            climb_rate=vz,
            accel_magnitude=0.0 if grounded else math.sqrt(ax * ax + ay * ay + az * az),
            mass_kg=self.cfg.mass_kg,
            payload_kg=self.cfg.payload_kg,
        )
        if not body.on_ground:
            self.flight_time += dt
        elif not was_on_ground:
            self._on_touchdown()

    def _on_touchdown(self) -> None:
        speed = self.body.last_touchdown_speed
        if speed > self.cfg.hard_landing_speed:
            self._fail(f"hard landing at {speed:.1f} m/s", category=EventCategory.COLLISION)
        expected = self.flight_mode in (FlightMode.LAND, FlightMode.RTL, FlightMode.EMERGENCY)
        self.armed = False
        self.target_position = None
        self._hold = None
        self._vel_ctrl.reset()
        self._set_mode(FlightMode.DISARMED, task="LANDED", reason="touchdown")
        p = self.body.position
        self._emit("landed", f"landed at ({p[0]:.1f}, {p[1]:.1f}) touchdown {speed:.1f} m/s",
                   Severity.INFO if expected else Severity.WARNING, touchdown_speed=round(speed, 2))

    def post_step(self, t: float) -> None:
        """Checks run once per engine tick after physics: failsafes and housekeeping."""
        self._time = t
        self._check_battery()
        if (self.flight_mode == FlightMode.ARMED and self.body.on_ground
                and t - self._armed_since > self.cfg.auto_disarm_delay):
            self.armed = False
            self._set_mode(FlightMode.DISARMED, task="IDLE", reason="auto-disarm")

    def _check_battery(self) -> None:
        state = self.battery.state
        if state != self._battery_state:
            self._battery_state = state
            severity = {
                BatteryState.NORMAL: Severity.INFO,
                BatteryState.WARNING: Severity.WARNING,
                BatteryState.RETURN_HOME: Severity.WARNING,
            }.get(state, Severity.CRITICAL)
            self._emit("battery_state", f"battery {state} ({self.battery.percent:.0f}%)", severity)
        if not self.in_flight:
            return
        mode = self.flight_mode
        done = self._battery_failsafes_done
        if state == BatteryState.DEPLETED and not self._motors_killed:
            self._fail("battery depleted in flight")
            self._begin_emergency("battery depleted", kill=True)
        elif state == BatteryState.EMERGENCY and BatteryState.EMERGENCY not in done \
                and mode not in (FlightMode.EMERGENCY, FlightMode.LAND):
            done.add(BatteryState.EMERGENCY)
            self._begin_emergency("battery emergency threshold", kill=False)
        elif state == BatteryState.RETURN_HOME and BatteryState.RETURN_HOME not in done \
                and mode not in (FlightMode.RTL, FlightMode.LAND, FlightMode.EMERGENCY):
            done.add(BatteryState.RETURN_HOME)
            self._emit("battery_failsafe", f"battery {self.battery.percent:.0f}% - returning home", Severity.WARNING)
            self._start_rtl("battery failsafe")

    def update(self, dt: float, t: float, wind: Vector3 | None = None) -> None:
        """Stand-alone update (guidance + physics + checks) for single-drone use/tests."""
        v = self.compute_guidance(t)
        self.integrate(v, np.zeros(3) if wind is None else wind, dt)
        self.post_step(t + dt)

    # ================================================================ telemetry
    def current_target(self) -> Vector3 | None:
        mode = self.flight_mode
        if mode == FlightMode.GOTO and self._route:
            return self._route[0]
        if mode in (FlightMode.TAKEOFF, FlightMode.GOTO):
            return self.target_position
        if mode == FlightMode.HOVER:
            return self._hold
        if mode == FlightMode.FORMATION:
            return self.swarm_target if self.swarm_target is not None else self._hold
        if mode == FlightMode.LAND and self._hold is not None:
            return vec3(self._hold[0], self._hold[1], self._ground(self._hold))
        if mode == FlightMode.RTL:
            h = self.home_position
            return vec3(h[0], h[1], self._ground(h) if self._rtl_phase == RtlPhase.LAND else self._rtl_altitude)
        return None

    def get_telemetry(self, gps: tuple[float, float, float] | None = None) -> DroneTelemetry:
        b = self.body
        if gps is None:
            lat, lon, alt = self._geo.enu_to_geodetic(b.position)
            gps = (float(lat), float(lon), float(alt))
        target = self.current_target()
        return DroneTelemetry(
            drone_id=self.id,
            name=self.name,
            position=tuple(float(x) for x in b.position),
            gps=gps,
            altitude_agl=self.altitude_agl,
            velocity=tuple(float(x) for x in b.velocity),
            acceleration=tuple(float(x) for x in b.acceleration),
            heading=self.heading,
            roll=math.degrees(b.roll),
            pitch=math.degrees(b.pitch),
            yaw_rate=-math.degrees(b.yaw_rate),  # compass convention: clockwise positive
            battery=self.battery.percent,
            battery_state=self.battery.state,
            battery_voltage=self.battery.voltage,
            power_w=self.battery.power_w,
            mode=self.flight_mode,
            armed=self.armed,
            airborne=self.airborne,
            health=self.health,
            communication=self.comm_status,
            task=self.current_task,
            target=tuple(float(x) for x in target) if target is not None else None,
            home=tuple(float(x) for x in self.home_position),
            neighbors=list(self.neighbors),
            collision_state=self.collision_state,
            nearest_distance=self.nearest_distance,
            flight_time=self.flight_time,
            distance_travelled=self.distance_travelled,
            link_quality=self.link_quality,
            gps_fix=self.gps_fix,
            satellites=self.satellites,
            hdop=self.hdop,
            time_left_s=self.battery.flight_time_left_s(self.airborne, self.cfg.mass_kg, self.cfg.payload_kg),
            est_position=tuple(float(x) for x in self.est_position) if self.est_position is not None else None,
            est_heading=yaw_to_heading(self.est_heading) if self.est_heading is not None else None,
            pos_sigma=self.pos_sigma,
            failures=list(self.failures),
        )


SimulatedDrone = Drone
"""Alias matching the naming used by the integration layer (SimulatedDrone / PX4Drone / MAVLinkDrone)."""
