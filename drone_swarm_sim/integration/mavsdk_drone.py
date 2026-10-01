"""MAVSDK vehicle link (native ``mavsdk`` >= 4 binding; PX4 first, ArduPilot for the basic actions).

MAVSDK 4 wraps the C++ core directly: telemetry arrives through callbacks on MAVSDK's own threads and
actions are blocking calls. The link therefore keeps one worker thread that connects, subscribes the
telemetry callbacks (they only write the shared :class:`RemoteState`) and executes queued commands,
so a slow ``arm()`` or ``takeoff()`` never blocks the simulation tick. Results and errors come back as
events.

Connection strings are MAVSDK's: ``udpin://0.0.0.0:14540`` (PX4 SITL), ``tcpout://127.0.0.1:5762``
(ArduPilot SITL serial port 1), ``serial:///dev/ttyACM0:57600``.

Command mapping: ``takeoff`` -> ``set_takeoff_altitude``, ``arm``, ``takeoff``; ``goto`` ->
``set_current_speed`` + ``goto_location`` (altitude AMSL); ``velocity`` -> offboard
``set_velocity_ned`` (offboard is started after the first setpoint, as PX4 requires); ``mode LAND /
RTL / GUIDED`` -> ``land`` / ``return_to_launch`` / ``hold``; ``disarm(force)`` -> ``kill``.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any

from simulation.events import Severity

from .remote import RemoteLink

log = logging.getLogger(__name__)

try:
    import mavsdk
    from mavsdk import ComponentType, Configuration, Mavsdk
    from mavsdk.plugins.action import Action, ActionError
    from mavsdk.plugins.offboard import Offboard, VelocityNedYaw
    from mavsdk.plugins.telemetry import Telemetry
except Exception:  # pragma: no cover - optional dependency (or the legacy gRPC package)
    mavsdk = None

_FIX = {"NO_GPS": 0, "NO_FIX": 1, "FIX_2D": 2}


class MAVSDKLink(RemoteLink):
    kind = "mavsdk"

    def __init__(self, url: str, *, connect_timeout: float = 30.0) -> None:
        if mavsdk is None:
            raise RuntimeError("the native mavsdk binding (mavsdk>=4) is not installed (pip install mavsdk)")
        super().__init__(url)
        self.connect_timeout = connect_timeout
        self._queue: queue.SimpleQueue[tuple[str, tuple]] = queue.SimpleQueue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._mavsdk = None
        self._system = None
        self._action = self._telemetry = self._offboard = None
        self._offboard_on = False

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"mavsdk {self.url}", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        for plugin in (self._offboard, self._action, self._telemetry, self._system, self._mavsdk):
            try:
                if plugin is not None:
                    plugin.destroy()
            except Exception:  # pragma: no cover - best effort
                pass

    def _send(self, command: str, args: tuple) -> None:
        self._queue.put((command, args))

    # ------------------------------------------------------------------ worker thread
    def _run(self) -> None:
        try:
            self._mavsdk = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
            result = self._mavsdk.add_any_connection(self.url)
            if int(result) != 0:                              # ConnectionResult.SUCCESS == 0
                self._event("link_error", f"MAVSDK cannot open {self.url}: {result}", Severity.WARNING)
                return
            deadline = time.monotonic() + self.connect_timeout
            while self._system is None and not self._stop.is_set() and time.monotonic() < deadline:
                self._system = self._mavsdk.first_autopilot(1.0)
            if self._system is None:
                self._event("link_error", f"MAVSDK: no autopilot on {self.url}", Severity.WARNING)
                return
            self._action, self._telemetry, self._offboard = Action(self._system), Telemetry(self._system), Offboard(self._system)
            self._subscribe()
            self._update(connected=True, last_heartbeat=time.monotonic())
            self._event("comm_info", f"MAVSDK connected ({self.url})")
        except Exception as exc:
            log.exception("MAVSDK link %s failed", self.url)
            self._event("link_error", f"MAVSDK {self.url}: {exc.__class__.__name__}: {exc}", Severity.WARNING)
            return
        while not self._stop.is_set():
            try:
                command, args = self._queue.get(timeout=0.1)
            except queue.Empty:
                self._update(connected=bool(self._system.is_connected()))
                continue
            self._do(command, args)

    def _subscribe(self) -> None:
        tel = self._telemetry
        # ArduPilot only streams what is requested: ask for the telemetry the GCS uses (PX4 ignores / accepts).
        for setter, hz in (("set_rate_position", 5.0), ("set_rate_velocity_ned", 5.0), ("set_rate_attitude_euler", 10.0),
                           ("set_rate_gps_info", 2.0), ("set_rate_battery", 1.0), ("set_rate_health", 1.0),
                           ("set_rate_home", 0.5), ("set_rate_in_air", 2.0)):
            try:
                getattr(tel, setter)(hz)
            except Exception as exc:                          # not supported by this autopilot: keep defaults
                log.debug("MAVSDK %s failed: %s", setter, exc)
        tel.subscribe_position(lambda p, _: self._update(
            has_position=bool(p.latitude_deg or p.longitude_deg), lat=p.latitude_deg, lon=p.longitude_deg,
            alt_amsl=p.absolute_altitude_m, rel_alt=p.relative_altitude_m, last_heartbeat=time.monotonic()))
        tel.subscribe_velocity_ned(lambda v, _: self._update(vel_ned=(v.north_m_s, v.east_m_s, v.down_m_s)))
        tel.subscribe_attitude_euler(lambda a, _: self._update(
            roll=math.radians(a.roll_deg), pitch=math.radians(a.pitch_deg), yaw_ned=math.radians(a.yaw_deg)))
        tel.subscribe_battery(lambda b, _: self._update(
            battery_pct=None if b.remaining_percent is None or b.remaining_percent != b.remaining_percent
            else float(b.remaining_percent if b.remaining_percent > 1.0 else 100.0 * b.remaining_percent),
            voltage=b.voltage_v))
        tel.subscribe_flight_mode(lambda m, _: self._update(mode=self._mode_name(m)))
        tel.subscribe_armed(lambda armed, _: self._update(armed=bool(armed)))
        tel.subscribe_in_air(lambda in_air, _: self._update(in_air=bool(in_air)))
        tel.subscribe_gps_info(lambda g, _: self._update(
            gps_fix=_FIX.get(getattr(g.fix_type, "name", str(g.fix_type).split(".")[-1]), 3),
            satellites=int(g.num_satellites or 0)))
        # "armable" alone is optimistic on ArduPilot: also wait for a home position (AHRS origin).
        tel.subscribe_health(lambda h, _: self._update(ready=bool(h.is_armable and h.is_home_position_ok)))
        tel.subscribe_home(lambda h, _: self._update(home=(h.latitude_deg, h.longitude_deg, h.absolute_altitude_m))
                           if (h.latitude_deg or h.longitude_deg) else None)

    @staticmethod
    def _mode_name(mode: Any) -> str:
        name = getattr(mode, "name", None) or str(mode).split(".")[-1]      # IntEnum in mavsdk 4
        return {"HOLD": "GUIDED", "OFFBOARD": "GUIDED", "POSCTL": "GUIDED", "RETURN_TO_LAUNCH": "RTL"}.get(name, name)

    # ------------------------------------------------------------------ commands (worker thread, blocking OK)
    def _do(self, command: str, args: tuple) -> None:
        a = self._action
        try:
            if command == "arm":
                a.arm()
            elif command == "disarm":
                a.kill() if args and args[0] else a.disarm()
            elif command == "takeoff":
                a.set_takeoff_altitude(float(args[0]))
                try:
                    a.hold()                                  # ArduPilot takes off only from a guided mode
                except Exception:
                    pass
                deadline = time.monotonic() + 30.0            # pre-arm checks (EKF, home) may still be settling
                while True:
                    try:
                        a.arm()
                        break
                    except ActionError:
                        if time.monotonic() > deadline or self._stop.is_set():
                            raise
                        time.sleep(3.0)
                for attempt in range(4):                      # the first takeoff right after arming may be refused
                    try:
                        a.takeoff()
                        break
                    except ActionError:
                        if attempt == 3:
                            raise
                        time.sleep(1.5)
                self._event("command_ack", "takeoff accepted")
            elif command == "mode":
                self._stop_offboard()
                name = str(args[0]).upper()
                (a.land if name == "LAND" else a.return_to_launch if name == "RTL" else a.hold)()
            elif command == "goto":
                lat, lon, alt, speed = args
                self._stop_offboard()
                if speed:
                    try:
                        a.set_current_speed(float(speed))
                    except ActionError:
                        pass                                  # not every autopilot supports it
                a.goto_location(lat, lon, alt, float("nan"))
            elif command == "velocity":
                vn, ve, vd = args
                yaw = math.degrees(math.atan2(ve, vn)) if math.hypot(vn, ve) > 0.5 else 0.0
                self._offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, yaw))
                if not self._offboard_on:
                    self._offboard.start()
                    self._offboard_on = True
            elif command == "yaw":
                self._event("command_rejected", "yaw is not exposed by MAVSDK actions (use goto heading)", Severity.INFO)
            else:
                self._event("command_rejected", f"unknown link command {command}", Severity.WARNING)
        except Exception as exc:
            self._event("command_ack", f"{command} rejected: {exc}", Severity.WARNING)

    def _stop_offboard(self) -> None:
        if self._offboard_on:
            try:
                self._offboard.stop()
            except Exception:  # pragma: no cover - already stopped
                pass
            self._offboard_on = False
