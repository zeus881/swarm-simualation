"""MAVLink vehicle link over pymavlink (ArduPilot, PX4, any MAVLink autopilot or SITL).

Connection strings follow pymavlink: ``tcp:127.0.0.1:5760`` (ArduPilot SITL), ``udpin:0.0.0.0:14550``,
``udpout:192.168.1.20:14550``, ``COM4`` / ``/dev/ttyUSB0`` with ``,57600`` baud...

The link runs in a daemon thread: it receives messages into a :class:`RemoteState`, sends a GCS
heartbeat at 1 Hz, requests telemetry streams, and executes queued commands. Multi-step commands run
as small state machines with timeouts so nothing ever blocks the simulation tick:

* ``takeoff``: GUIDED -> arm (re-sent every 3 s until armed; pre-arm failures are reported from the
  autopilot's STATUSTEXT) -> ``MAV_CMD_NAV_TAKEOFF`` -> wait until airborne.
* ``goto``: GUIDED if needed, optional ``MAV_CMD_DO_CHANGE_SPEED``, then ``SET_POSITION_TARGET_GLOBAL_INT``
  (``MAV_FRAME_GLOBAL_INT``, altitude AMSL, position-only type mask).
* ``velocity``: ``SET_POSITION_TARGET_LOCAL_NED`` with a velocity-only type mask (streamed by the swarm
  layer at ``hardware.setpoint_rate_hz``).
* ``mode``: ``set_mode`` (LAND, RTL, GUIDED, ...), ``arm`` / ``disarm`` (force = 21196), ``yaw``
  (``MAV_CMD_CONDITION_YAW``).

Link quality comes from MAVLink sequence-number loss over the last seconds.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from simulation.events import Severity

from .remote import GUIDED_MODES, RemoteLink

log = logging.getLogger(__name__)

try:
    from pymavlink import mavutil
except ImportError:  # pragma: no cover - optional dependency
    mavutil = None

POS_ONLY = 0b110111111000        # SET_POSITION_TARGET_*: use x/y/z only
VEL_ONLY = 0b110111000111        # use vx/vy/vz only
PREARM_BIT = 1 << 28             # MAV_SYS_STATUS_PREARM_CHECK
FORCE_DISARM = 21196
EKF_POS_ABS = 16 | 32            # EKF_STATUS_REPORT: absolute horizontal + vertical position
EKF_CONST_POS = 128              # EKF in constant-position mode (no GPS yet)


@dataclass
class _Step:
    action: Callable[[], None] | None
    done: Callable[[], bool]
    timeout: float
    label: str
    retry: float = 0.0            # re-run the action every ``retry`` s while waiting (0 = once)
    started: float = 0.0
    last_try: float = 0.0


@dataclass
class _Procedure:
    name: str
    steps: list[_Step] = field(default_factory=list)


class MAVLinkLink(RemoteLink):
    kind = "mavlink"

    def __init__(self, url: str, *, source_system: int = 255, target_system: int | None = None) -> None:
        if mavutil is None:
            raise RuntimeError("pymavlink is not installed (pip install pymavlink)")
        super().__init__(url)
        self.source_system = source_system
        self.wanted_system = target_system
        self._queue: queue.SimpleQueue[tuple[str, tuple]] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conn = None
        self._sys: int | None = None
        self._comp = 1
        self._procedures: list[_Procedure] = []
        self._streams_requested = False
        self._last_hb_sent = 0.0
        self._last_status = ""
        self._ext_state = False
        self._prearm_ok = False
        self._ekf_ok: bool | None = None       # None until an EKF_STATUS_REPORT arrives (autopilots without one)
        self._loss_mark = (0, 0, time.monotonic())

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"mavlink {self.url}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - best effort
                pass

    def _send(self, command: str, args: tuple) -> None:
        self._queue.put((command, args))

    # ------------------------------------------------------------------ thread
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._conn = mavutil.mavlink_connection(self.url, source_system=self.source_system,
                                                        source_component=190, autoreconnect=True)
            except Exception as exc:
                self._event("link_error", f"cannot open {self.url}: {exc}", Severity.WARNING)
                self._stop.wait(2.0)
                continue
            try:
                self._loop()
            except Exception as exc:          # keep the thread alive; reconnect
                log.exception("MAVLink link %s failed", self.url)
                self._event("link_error", f"{self.url}: {exc.__class__.__name__}: {exc}", Severity.WARNING)
                self._update(connected=False)
                self._stop.wait(1.0)

    def _loop(self) -> None:
        conn = self._conn
        while not self._stop.is_set():
            msg = conn.recv_match(blocking=True, timeout=0.05)
            if msg is not None:
                self._handle(msg)
            now = time.monotonic()
            if now - self._last_hb_sent >= 1.0:
                self._last_hb_sent = now
                conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                self._update_quality(now)
            if self._sys is not None and not self._streams_requested:
                self._streams_requested = True
                conn.mav.request_data_stream_send(self._sys, self._comp, mavutil.mavlink.MAV_DATA_STREAM_ALL, 5, 1)
                self._set_interval(mavutil.mavlink.MAVLINK_MSG_ID_EXTENDED_SYS_STATE, 2.0)
                self._set_interval(mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION, 0.5)
            while True:
                try:
                    command, args = self._queue.get_nowait()
                except queue.Empty:
                    break
                if self._sys is None:
                    self._event("command_rejected", f"{command}: vehicle not connected yet", Severity.WARNING)
                    continue
                self._execute(command, args)
            self._run_procedures(now)

    def _set_interval(self, msg_id: int, hz: float) -> None:
        self._command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)

    def _update_quality(self, now: float) -> None:
        conn = self._conn
        loss, count = getattr(conn, "mav_loss", 0), getattr(conn, "mav_count", 0)
        l0, c0, t0 = self._loss_mark
        if now - t0 >= 2.0:
            d_loss, d_count = loss - l0, count - c0
            if d_count + d_loss > 0:
                self._update(link_quality=100.0 * d_count / (d_count + d_loss))
            self._loss_mark = (loss, count, now)

    # ------------------------------------------------------------------ receive
    def _handle(self, msg: Any) -> None:
        kind = msg.get_type()
        if kind == "HEARTBEAT":
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS or msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                return
            src = msg.get_srcSystem()
            if self.wanted_system is not None and src != self.wanted_system:
                return
            if self._sys is None:
                self._sys, self._comp = src, msg.get_srcComponent()
                self._conn.target_system, self._conn.target_component = self._sys, self._comp
                self._event("comm_info", f"vehicle {src} connected ({self.url})")
            armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            autopilot = {mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA: "ardupilot",
                         mavutil.mavlink.MAV_AUTOPILOT_PX4: "px4"}.get(msg.autopilot, str(msg.autopilot))
            fields: dict[str, Any] = dict(connected=True, last_heartbeat=time.monotonic(),
                                          mode=mavutil.mode_string_v10(msg), armed=armed, autopilot=autopilot)
            if not self._ext_state:
                s = self._state
                fields["in_air"] = armed and s.rel_alt > 0.5
            self._update(**fields)
        elif self._sys is not None and msg.get_srcSystem() != self._sys:
            return
        elif kind == "GLOBAL_POSITION_INT":
            has = msg.lat != 0 or msg.lon != 0
            self._update(has_position=has, lat=msg.lat / 1e7, lon=msg.lon / 1e7, alt_amsl=msg.alt / 1000.0,
                         rel_alt=msg.relative_alt / 1000.0, vel_ned=(msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0))
        elif kind == "ATTITUDE":
            self._update(roll=msg.roll, pitch=msg.pitch, yaw_ned=msg.yaw, yaw_rate=msg.yawspeed)
        elif kind == "SYS_STATUS":
            self._prearm_ok = bool(msg.onboard_control_sensors_health & PREARM_BIT) \
                if msg.onboard_control_sensors_present & PREARM_BIT else True
            self._update(battery_pct=None if msg.battery_remaining < 0 else float(msg.battery_remaining),
                         voltage=msg.voltage_battery / 1000.0 if msg.voltage_battery not in (0, 65535) else None,
                         ready=self._prearm_ok and self._ekf_ok is not False)
        elif kind == "EKF_STATUS_REPORT":
            # Ready to arm in GUIDED only with an absolute position estimate (not constant-position mode).
            self._ekf_ok = (msg.flags & EKF_POS_ABS) == EKF_POS_ABS and not msg.flags & EKF_CONST_POS
            self._update(ready=self._prearm_ok and self._ekf_ok)
        elif kind == "GPS_RAW_INT":
            self._update(gps_fix=int(msg.fix_type), satellites=int(msg.satellites_visible),
                         hdop=msg.eph / 100.0 if msg.eph not in (0, 65535) else 99.9)
        elif kind == "EXTENDED_SYS_STATE":
            self._ext_state = True
            self._update(in_air=msg.landed_state in (2, 3, 4))   # IN_AIR, TAKEOFF, LANDING
        elif kind == "HOME_POSITION":
            self._update(home=(msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1000.0))
        elif kind == "COMMAND_ACK":
            name = mavutil.mavlink.enums["MAV_CMD"].get(msg.command)
            name = name.name if name else str(msg.command)
            if msg.result != mavutil.mavlink.MAV_RESULT_ACCEPTED and msg.command != mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
                result = mavutil.mavlink.enums["MAV_RESULT"].get(msg.result)
                self._event("command_ack", f"{name} rejected: {result.name if result else msg.result}"
                            + (f" ({self._last_status})" if self._last_status else ""), Severity.WARNING)
        elif kind == "STATUSTEXT":
            text = msg.text if isinstance(msg.text, str) else msg.text.decode(errors="replace")
            text = text.strip("\x00").strip()
            self._last_status = text
            sev = Severity.CRITICAL if msg.severity <= 2 else Severity.WARNING if msg.severity <= 4 else Severity.INFO
            if sev != Severity.INFO or text.lower().startswith(("prearm", "arm", "disarm", "takeoff", "land")):
                self._event("autopilot", text, sev)

    # ------------------------------------------------------------------ send
    def _command(self, cmd: int, *params: float) -> None:
        p = list(params) + [0.0] * (7 - len(params))
        self._conn.mav.command_long_send(self._sys, self._comp, cmd, 0, *p[:7])

    def _set_mode(self, name: str) -> None:
        try:
            self._conn.set_mode(name)
        except Exception as exc:                  # unknown mode for this autopilot
            self._event("command_rejected", f"mode {name}: {exc}", Severity.WARNING)

    def _goto(self, lat: float, lon: float, alt: float) -> None:
        self._conn.mav.set_position_target_global_int_send(
            0, self._sys, self._comp, mavutil.mavlink.MAV_FRAME_GLOBAL_INT, POS_ONLY,
            int(round(lat * 1e7)), int(round(lon * 1e7)), float(alt), 0, 0, 0, 0, 0, 0, 0, 0)

    def _execute(self, command: str, args: tuple) -> None:
        mav = mavutil.mavlink
        if command == "arm":
            self._command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        elif command == "disarm":
            force = bool(args[0]) if args else False
            self._command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 0, FORCE_DISARM if force else 0)
        elif command == "mode":
            self._procedures = [p for p in self._procedures if p.name != "takeoff"]   # a mode change cancels takeoff
            self._set_mode(args[0])
        elif command == "takeoff":
            self._start_takeoff(float(args[0]))
        elif command == "goto":
            lat, lon, alt, speed = args
            if self._state.mode.upper() not in GUIDED_MODES:
                self._set_mode("GUIDED")
            if speed:
                self._command(mav.MAV_CMD_DO_CHANGE_SPEED, 1, float(speed), -1)
            self._goto(lat, lon, alt)
        elif command == "velocity":
            vn, ve, vd = args
            self._conn.mav.set_position_target_local_ned_send(0, self._sys, self._comp, mav.MAV_FRAME_LOCAL_NED, VEL_ONLY,
                                                              0, 0, 0, vn, ve, vd, 0, 0, 0, 0, 0)
        elif command == "yaw":
            self._command(mav.MAV_CMD_CONDITION_YAW, float(args[0]) % 360.0, 30.0, 0, 0)
        else:
            self._event("command_rejected", f"unknown link command {command}", Severity.WARNING)

    # ------------------------------------------------------------------ procedures
    def _start_takeoff(self, alt: float) -> None:
        s = lambda: self._state                                         # noqa: E731
        mav = mavutil.mavlink
        steps = [
            _Step(lambda: self._set_mode("GUIDED"), lambda: s().mode.upper() == "GUIDED", 8.0, "GUIDED mode", retry=2.0),
            _Step(lambda: self._command(mav.MAV_CMD_COMPONENT_ARM_DISARM, 1), lambda: s().armed, 30.0, "arm", retry=3.0),
            _Step(lambda: self._command(mav.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, math.nan, 0, 0, alt),
                  lambda: s().rel_alt > min(1.0, 0.5 * alt), 15.0, "takeoff", retry=4.0),
        ]
        self._procedures = [p for p in self._procedures if p.name != "takeoff"] + [_Procedure("takeoff", steps)]

    def _run_procedures(self, now: float) -> None:
        for proc in list(self._procedures):
            if not proc.steps:
                self._procedures.remove(proc)
                continue
            step = proc.steps[0]
            if step.started == 0.0:
                step.started = step.last_try = now
                if step.action:
                    step.action()
            if step.done():
                proc.steps.pop(0)
                if not proc.steps:
                    self._event("command_ack", f"{proc.name} complete")
                    self._procedures.remove(proc)
                continue
            if now - step.started > step.timeout:
                reason = f" ({self._last_status})" if self._last_status else ""
                self._event("command_ack", f"{proc.name} failed: {step.label} timed out{reason}", Severity.WARNING)
                self._procedures.remove(proc)
            elif step.retry and now - step.last_try >= step.retry and step.action:
                step.last_try = now
                step.action()
