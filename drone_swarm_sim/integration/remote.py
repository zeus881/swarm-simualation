"""Remote vehicles (real drones, PX4 / ArduPilot SITL) behind the same interface as simulated drones.

:class:`RemoteDrone` subclasses :class:`~simulation.drone.Drone`, so the swarm manager, the velocity
pipeline, missions, alerts and the GCS treat simulated and real vehicles identically. What changes:

* **No physics.** Every tick, :meth:`post_step` copies the latest telemetry from the link (a background
  thread) into the drone state: position / velocity converted from geodetic + NED to local ENU,
  attitude, armed, landed state, battery, GPS and link health. A missing heartbeat for
  ``hardware.heartbeat_timeout_s`` makes the vehicle ``COMM LOST``.
* **Commands become autopilot requests** sent through the link (arm, GUIDED + takeoff, LAND, RTL,
  position / velocity targets, yaw). They return immediately ("requested"); acknowledgements and
  autopilot messages come back as events, exactly like the simulated link model.
* **Flight mode.** The autopilot mode is mapped onto the platform's modes: LAND / RTL follow the
  autopilot, disarmed is DISARMED, and while the autopilot is in GUIDED the GCS-side mode (TAKEOFF,
  GOTO, HOVER, OFFBOARD, FORMATION) is kept, with arrival detection done here. Any other autopilot
  mode (LOITER, ALT_HOLD, AUTO, a pilot on the sticks...) shows as HOVER with the mode name as the task.
* **Swarm control.** In FORMATION / OFFBOARD the pipeline's velocity for this drone is sent as a
  velocity setpoint at ``hardware.setpoint_rate_hz`` (so formations, flocking and collision avoidance
  drive real vehicles too). Pass-through routes (obstacle planner) are sequenced here, one position
  target at a time.

Failure injection and battery failsafes are left to the real autopilot.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from simulation.drone import Drone, heading_to_yaw
from simulation.drone_interface import CommandResult
from simulation.events import EventBus, Severity
from simulation.types import CommStatus, FlightMode, Vector3, vec3

if TYPE_CHECKING:
    from simulation.config import SimConfig
    from simulation.environment import Environment
    from simulation.geo import GeoReference

GUIDED_MODES = frozenset({"GUIDED", "OFFBOARD", "HOLD", "POSCTL", "GUIDED_NOGPS"})


@dataclass
class RemoteState:
    """Latest telemetry from a vehicle, written by the link thread, read by the engine thread."""

    connected: bool = False
    last_heartbeat: float = -math.inf        # time.monotonic() of the last vehicle heartbeat
    mode: str = ""                           # autopilot mode name (GUIDED, LAND, RTL, AUTO, ...)
    armed: bool = False
    in_air: bool = False
    has_position: bool = False
    lat: float = 0.0
    lon: float = 0.0
    alt_amsl: float = 0.0
    rel_alt: float = 0.0
    vel_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)
    roll: float = 0.0                        # rad
    pitch: float = 0.0
    yaw_ned: float = 0.0                     # rad, clockwise from North
    yaw_rate: float = 0.0
    battery_pct: float | None = None
    voltage: float | None = None
    gps_fix: int = 0                         # MAVLink GPS_FIX_TYPE (3 = 3D)
    satellites: int = 0
    hdop: float = 99.9
    link_quality: float = 100.0
    home: tuple[float, float, float] | None = None   # lat, lon, alt_amsl
    ready: bool = False                      # pre-arm checks pass / EKF good enough to arm
    autopilot: str = ""


class RemoteLink:
    """Base class of a vehicle link. Subclasses run a background thread and implement :meth:`_send`."""

    kind = "remote"

    def __init__(self, url: str) -> None:
        self.url = url
        self._lock = threading.Lock()
        self._state = RemoteState()
        self._events: deque[tuple[str, str, Severity]] = deque(maxlen=200)

    # --- engine thread API (non-blocking)
    def snapshot(self) -> RemoteState:
        with self._lock:
            return replace(self._state)

    def events(self) -> list[tuple[str, str, Severity]]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def send(self, command: str, *args: Any) -> None:
        """Queue a command for the link thread."""
        self._send(command, args)

    def start(self) -> None:  # pragma: no cover - overridden
        pass

    def close(self) -> None:  # pragma: no cover - overridden
        pass

    # --- link thread helpers
    def _update(self, **fields: Any) -> None:
        with self._lock:
            for k, v in fields.items():
                setattr(self._state, k, v)

    def _event(self, kind: str, message: str, severity: Severity = Severity.INFO) -> None:
        self._events.append((kind, message, severity))

    def _send(self, command: str, args: tuple) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class RemoteDrone(Drone):
    """A real / SITL vehicle presented through the simulated drone's interface."""

    is_remote = True

    def __init__(self, drone_id: int, config: "SimConfig", environment: "Environment", geo: "GeoReference", *,
                 link: RemoteLink, events: EventBus | None = None, name: str | None = None) -> None:
        super().__init__(drone_id, config, environment, geo, home=environment.home_position, initial_battery=100.0,
                         events=events)
        self.link = link
        self.source = link.kind
        if name:
            self.name = name
        self.hw = config.hardware
        self.comm_managed = True
        self.nav_from_estimate = False
        self.comm_status = CommStatus.LOST
        self._last_setpoint = -math.inf
        self._setpoint_v = np.zeros(3)
        self._home_set = False
        self._autopilot_mode = ""
        self._was_connected = False
        self.remote_ready = False

    # ------------------------------------------------------------------ geometry
    def _enu(self, lat: float, lon: float, alt: float) -> Vector3:
        return np.asarray(self._geo.geodetic_to_enu(lat, lon, alt), dtype=np.float64)

    def _geodetic(self, p: Sequence[float]) -> tuple[float, float, float]:
        lat, lon, alt = self._geo.enu_to_geodetic(np.asarray(p, dtype=np.float64))
        return float(lat), float(lon), float(alt)

    # ------------------------------------------------------------------ state sync (engine thread)
    def post_step(self, t: float) -> None:
        import time as _time
        self._time = t
        s = self.link.snapshot()
        alive = s.connected and _time.monotonic() - s.last_heartbeat <= self.hw.heartbeat_timeout_s
        if alive != self._was_connected:
            self._was_connected = alive
            if alive:
                self._emit("comm_restored", f"{self.source} link up ({self.link.url})")
            else:
                self._emit("comm_lost", f"{self.source} link lost ({self.link.url})", Severity.CRITICAL)
        self.comm_status = CommStatus.ONLINE if alive else CommStatus.LOST
        self.link_quality = s.link_quality if alive else 0.0
        if s.has_position:
            if s.home is not None and not self._home_set:
                self.home_position = self._enu(*s.home)
                self._home_set = True
            self.body.position = self._enu(s.lat, s.lon, s.alt_amsl)
            vn, ve, vd = s.vel_ned
            self.body.velocity = vec3(ve, vn, -vd)
            self.body.roll, self.body.pitch = s.roll, s.pitch
            self.body.yaw = math.atan2(math.cos(s.yaw_ned), math.sin(s.yaw_ned))   # NED yaw -> ENU yaw
            self.body.yaw_rate = -s.yaw_rate
        self.body.on_ground = not s.in_air
        self.armed = s.armed
        self.remote_ready = s.ready
        if s.battery_pct is not None:
            self.battery.energy_wh = self.battery.capacity_wh * max(0.0, min(100.0, s.battery_pct)) / 100.0
        self.gps_fix = {0: "NONE", 1: "NONE", 2: "2D"}.get(s.gps_fix, "3D")
        self.satellites, self.hdop = s.satellites, s.hdop
        self._sync_mode(s)
        for kind, message, severity in self.link.events():
            self._emit(kind, message, severity)
            if kind == "command_ack" and message.startswith("takeoff failed") and self.flight_mode == FlightMode.TAKEOFF:
                # the autopilot never took off: fall back to what it really is doing
                self._set_mode(FlightMode.ARMED if s.armed else FlightMode.DISARMED, task="IDLE", reason="takeoff failed")
        self._check_battery_state()

    def _check_battery_state(self) -> None:
        state = self.battery.state
        if state != self._battery_state:
            self._battery_state = state
            self._emit("battery_state", f"battery {state} ({self.battery.percent:.0f}%)",
                       Severity.INFO if state.value == "NORMAL" else Severity.WARNING)

    def _sync_mode(self, s: RemoteState) -> None:
        mode = s.mode.upper()
        self._autopilot_mode = mode
        if not s.armed:
            if self.flight_mode != FlightMode.DISARMED and not (self.flight_mode == FlightMode.TAKEOFF and not s.in_air):
                self._set_mode(FlightMode.DISARMED, task="IDLE", reason=f"autopilot {mode or 'disarmed'}")
            return
        if mode == "LAND":
            self._set_mode(FlightMode.LAND, task="LAND")
        elif mode in ("RTL", "SMART_RTL", "RETURN"):
            self._set_mode(FlightMode.RTL, task="RTL")
        elif mode in GUIDED_MODES:
            if not s.in_air and self.flight_mode not in (FlightMode.TAKEOFF,):
                self._set_mode(FlightMode.ARMED, task="IDLE")
            elif self.flight_mode in (FlightMode.DISARMED, FlightMode.ARMED, FlightMode.LAND, FlightMode.RTL) and s.in_air:
                self._set_mode(FlightMode.HOVER, task="HOVER")
            self._advance_guided()
        else:
            self._set_mode(FlightMode.HOVER, task=mode or "MANUAL")

    def _advance_guided(self) -> None:
        p = self.body.position
        if self.flight_mode == FlightMode.TAKEOFF and self.target_position is not None:
            if abs(self.target_position[2] - p[2]) < 1.0:
                self._hold = self.target_position.copy()
                self._set_mode(FlightMode.HOVER, task="HOVER", reason="takeoff complete")
                self._emit("takeoff_complete", f"reached {self.altitude_agl:.1f} m")
        elif self.flight_mode == FlightMode.GOTO and self._route:
            if np.linalg.norm(self._route[0] - p) < max(3.0 * self.cfg.acceptance_radius, 0.8 * self._cruise_speed):
                self._route.pop(0)
                nxt = self._route[0] if self._route else self.target_position
                self.link.send("goto", *self._geodetic(nxt), self._cruise_speed)
        elif self.flight_mode == FlightMode.GOTO and self.target_position is not None:
            if np.linalg.norm(self.target_position - p) < 2.0 * self.cfg.acceptance_radius:
                self._hold = self.target_position.copy()
                self._set_mode(FlightMode.HOVER, task="HOLD", reason="arrived")
                self._emit("arrived", f"arrived at ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})")

    # ------------------------------------------------------------------ swarm pipeline hooks
    def compute_guidance(self, t: float) -> Vector3:
        """The autopilot flies itself; in swarm control the pipeline supplies the velocity."""
        self._time = t
        if self.flight_mode == FlightMode.OFFBOARD:
            v = self._velocity_sp.copy()
        else:
            v = self.body.velocity.copy()
        self.last_guidance = v
        return v

    def integrate(self, v_cmd: Vector3, wind: Vector3, dt: float, ground_z: float | None = None) -> None:
        """No physics: in FORMATION / OFFBOARD, stream the commanded velocity to the autopilot."""
        if self.flight_mode not in (FlightMode.FORMATION, FlightMode.OFFBOARD) or not self.in_flight:
            return
        if self._time - self._last_setpoint >= 1.0 / self.hw.setpoint_rate_hz:
            self._last_setpoint = self._time
            v = np.asarray(v_cmd, dtype=np.float64)
            self.link.send("velocity", float(v[1]), float(v[0]), float(-v[2]))   # ENU -> NED

    # ------------------------------------------------------------------ commands
    def _require_link(self, action: str) -> CommandResult | None:
        if self.comm_status == CommStatus.LOST:
            return CommandResult.fail(f"{action} rejected: no link to {self.name}")
        return None

    def arm(self) -> CommandResult:
        if (r := self._require_link("arm")) is not None:
            return r
        self.link.send("arm")
        return CommandResult.ok("arm requested")

    def disarm(self) -> CommandResult:
        if (r := self._require_link("disarm")) is not None:
            return r
        if self.airborne:
            return CommandResult.fail("disarm rejected: vehicle is airborne (use emergency_stop)")
        self.link.send("disarm")
        return CommandResult.ok("disarm requested")

    def takeoff(self, altitude: float | None = None) -> CommandResult:
        if (r := self._require_link("takeoff")) is not None:
            return r
        alt = self.cfg.takeoff_altitude if altitude is None else float(altitude)
        if not self.cfg.min_altitude <= alt <= self._env.max_altitude:
            return CommandResult.fail(f"takeoff rejected: altitude must be in [{self.cfg.min_altitude}, {self._env.max_altitude}] m")
        if self.in_flight and self.airborne:
            return CommandResult.fail("takeoff rejected: already airborne")
        p = self.body.position
        self.target_position = vec3(p[0], p[1], self._ground() + alt)
        self.link.send("takeoff", alt)
        self._set_mode(FlightMode.TAKEOFF, task="TAKEOFF")
        return CommandResult.ok(f"takeoff to {alt:.1f} m requested (GUIDED, arm, takeoff)")

    def land(self) -> CommandResult:
        if (r := self._require_link("land")) is not None:
            return r
        self.link.send("mode", "LAND")
        self._set_mode(FlightMode.LAND, task="LAND")
        return CommandResult.ok("LAND requested")

    def return_to_home(self) -> CommandResult:
        if (r := self._require_link("return_to_home")) is not None:
            return r
        self.link.send("mode", "RTL")
        self._set_mode(FlightMode.RTL, task="RTL")
        return CommandResult.ok("RTL requested")

    def hover(self) -> CommandResult:
        if (rejected := self._gate_in_flight("hover")) is not None:
            return rejected
        self._hold = self.body.position.copy()
        self._route = []
        self.link.send("goto", *self._geodetic(self._hold), None)
        self._set_mode(FlightMode.HOVER, task="HOVER")
        return CommandResult.ok("holding position")

    def goto(self, position: Sequence[float], speed: float | None = None, heading: float | None = None) -> CommandResult:
        if (rejected := self._gate_in_flight("goto")) is not None:
            return rejected
        try:
            requested = self._as_vec3(position, "position")
        except ValueError as exc:
            return CommandResult.fail(f"goto rejected: {exc}")
        target = self._env.clamp(requested, min_agl=self.cfg.min_altitude)
        self._cruise_speed = min(float(speed), self.cfg.max_horizontal_speed) if speed else self.cfg.cruise_speed
        self.target_position = target
        self._route = []
        self.link.send("goto", *self._geodetic(target), self._cruise_speed)
        if heading is not None:
            self.link.send("yaw", float(heading))
        self._set_mode(FlightMode.GOTO, task="GOTO")
        return CommandResult.ok(f"goto ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) sent")

    def goto_path(self, route: Sequence[Sequence[float]], speed: float | None = None,
                  heading: float | None = None) -> CommandResult:
        if not route:
            return CommandResult.fail("goto_path rejected: empty route")
        points = [self._env.clamp(self._as_vec3(p, "route point"), min_agl=self.cfg.min_altitude) for p in route]
        result = self.goto(points[0] if len(points) > 1 else points[-1], speed=speed, heading=heading)
        if result.success and len(points) > 1:
            self.target_position = points[-1]
            self._route = points[:-1]
            result = CommandResult.ok(f"{result.message} via {len(self._route)} waypoint(s)")
        return result

    def set_velocity(self, velocity: Sequence[float], heading: float | None = None) -> CommandResult:
        if (rejected := self._gate_in_flight("set_velocity")) is not None:
            return rejected
        result = super().set_velocity(velocity, heading)
        if result.success:
            v = self._velocity_sp
            self.link.send("velocity", float(v[1]), float(v[0]), float(-v[2]))
            self._last_setpoint = self._time
        return result

    def set_heading(self, heading: float) -> CommandResult:
        if (r := self._require_link("set_heading")) is not None:
            return r
        self.link.send("yaw", float(heading))
        self._heading_sp = heading_to_yaw(heading)
        return CommandResult.ok(f"heading {heading % 360:.0f} deg requested")

    def emergency_stop(self) -> CommandResult:
        if (r := self._require_link("emergency_stop")) is not None:
            return r
        if self.cfg.emergency_stop_behavior == "kill" or not self.airborne:
            self.link.send("disarm", True)
            self._set_mode(FlightMode.EMERGENCY if self.airborne else FlightMode.DISARMED, task="EMERGENCY")
            return CommandResult.ok("force disarm requested (motors off)")
        self.link.send("mode", "LAND")
        self._set_mode(FlightMode.EMERGENCY, task="EMERGENCY")
        return CommandResult.ok("emergency LAND requested")

    def enter_swarm_control(self, behavior: str) -> CommandResult:
        result = super().enter_swarm_control(behavior)
        if result.success:
            self.link.send("mode", "GUIDED")
        return result

    def inject_motor_failure(self, total: bool, thrust_left: float) -> str:
        raise ValueError("failure injection is not available on real vehicles")

    # ------------------------------------------------------------------ telemetry
    def get_telemetry(self, gps: tuple[float, float, float] | None = None):
        t = super().get_telemetry(gps)
        t.task = self.current_task if self._autopilot_mode in GUIDED_MODES or not self._autopilot_mode \
            else f"{self.current_task} [{self._autopilot_mode}]"
        t.source = self.source
        t.time_left_s = None                    # the autopilot's own estimate is not modelled here
        return t

    def describe(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "source": self.source, "url": self.link.url,
                "connected": self.comm_status != CommStatus.LOST, "autopilot_mode": self._autopilot_mode,
                "ready": self.remote_ready}

    def close(self) -> None:
        self.link.close()
