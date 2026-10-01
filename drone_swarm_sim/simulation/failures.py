"""Failure injection for training (instructor mode).

``inject_failure {type, duration?, ...}`` applies to the addressed drones at runtime:

=============  =====================================================================================
type           effect
=============  =====================================================================================
motor          ``severity: partial`` (default) leaves ``failures.motor_partial_thrust`` of the thrust and
               the autopilot starts an emergency descent; ``severity: total`` is a crash (motors off)
gps_loss       no GPS fix (``duration`` s, or until cleared): the navigation filter dead-reckons and the
               estimate drifts; with ``control_source: estimate`` the drone drifts with it
comm_loss      no packet gets through (``duration`` s, or until cleared): COMM LOST, then the failsafe
battery_sag    the pack loses ``drop`` % of its capacity at once (may trigger battery failsafes)
wind_gust      an extra local wind of ``speed`` m/s from ``direction`` deg for ``duration`` s
=============  =====================================================================================

Timed failures expire on their own; ``clear_failure {type?}`` ends them early. Every injection, expiry
and clear is a logged event (injections are CRITICAL/WARNING, so they also raise operator alerts).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .commands import Command, CommandError, param_number
from .drone_interface import CommandResult
from .events import EventCategory, Severity

if TYPE_CHECKING:
    from .drone import Drone
    from .engine import SimulationEngine

FAILURE_TYPES = ("motor", "gps_loss", "comm_loss", "battery_sag", "wind_gust")


@dataclass
class ActiveFailure:
    id: int
    kind: str
    drone_id: int
    since: float
    until: float | None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.kind, "drone_id": self.drone_id, "since": round(self.since, 2),
                "until": round(self.until, 2) if self.until is not None else None, "params": self.params}


class FailureInjector:
    def __init__(self, engine: "SimulationEngine") -> None:
        self.engine = engine
        self.cfg = engine.config.failures
        self.active: list[ActiveFailure] = []
        self._ids = itertools.count(1)
        self.injected = 0
        engine.commands.register("inject_failure", self._inject,
                                 "Instructor: inject a failure {type: motor|gps_loss|comm_loss|battery_sag|wind_gust, "
                                 "duration?, severity? partial|total, drop? %, speed? m/s, direction? deg}")
        engine.commands.register("clear_failure", self._clear, "Instructor: end injected failures {type?}")

    # ------------------------------------------------------------------ commands
    def _inject(self, cmd: Command) -> CommandResult:
        if not self.cfg.enabled:
            raise CommandError("failure injection is disabled (failures.enabled)")
        p = cmd.params
        kind = p.get("type")
        if kind not in FAILURE_TYPES:
            raise CommandError(f"type must be one of {', '.join(FAILURE_TYPES)}")
        if not cmd.drone_ids:
            raise CommandError("select the drones to fail (explicit drone_ids)")
        duration = param_number(p, "duration", minimum=0.5, maximum=3600.0)
        drones = self.engine.swarm.select(cmd.drone_ids)
        t = self.engine.sim_time
        messages = []
        for d in drones:
            msg, params, until = self._apply(d, kind, p, duration, t)
            if until is not None or kind in ("gps_loss", "comm_loss", "motor"):
                self.active = [a for a in self.active if not (a.kind == kind and a.drone_id == d.id)]
                self.active.append(ActiveFailure(next(self._ids), kind, d.id, t, until, params))
            self.injected += 1
            severity = Severity.CRITICAL if kind in ("motor", "comm_loss", "gps_loss") else Severity.WARNING
            self.engine.events.emit(EventCategory.DRONE, "failure_injected", f"instructor: {msg}", severity=severity,
                                    drone_id=d.id, failure=kind, params=params, until=until)
            messages.append(f"{d.name}: {msg}")
        return CommandResult.ok(f"{kind} injected on {len(drones)} drone(s)", messages=messages)

    def _apply(self, d: "Drone", kind: str, p: dict[str, Any], duration: float | None,
               t: float) -> tuple[str, dict[str, Any], float | None]:
        until = t + duration if duration is not None else None
        eng = self.engine
        if kind == "motor":
            severity = p.get("severity", "partial")
            if severity not in ("partial", "total"):
                raise CommandError("severity must be partial or total")
            return d.inject_motor_failure(severity == "total", self.cfg.motor_partial_thrust), {"severity": severity}, None
        if kind == "gps_loss":
            if eng.sensors is None:
                raise CommandError("gps_loss needs the sensor model (sensors.enabled: true)")
            eng.sensors.set_gps_loss(d.id, until if until is not None else math.inf)
            d.failures.add("gps_loss")
            return f"GPS lost{f' for {duration:.0f} s' if duration else ''}", {}, until
        if kind == "comm_loss":
            if eng.comms is None:
                raise CommandError("comm_loss needs the communication model (communication.enabled: true)")
            eng.comms.force_down(d.id, until if until is not None else math.inf)
            d.failures.add("comm_loss")
            return f"link jammed{f' for {duration:.0f} s' if duration else ''}", {}, until
        if kind == "battery_sag":
            drop = param_number(p, "drop", default=self.cfg.default_battery_drop, minimum=1.0, maximum=100.0)
            b = d.battery
            b.energy_wh = max(0.0, b.energy_wh - b.capacity_wh * drop / 100.0)
            return f"battery sag -{drop:.0f} % (now {b.percent:.0f} %)", {"drop": drop}, None
        speed = param_number(p, "speed", default=self.cfg.default_gust_speed, minimum=0.5, maximum=60.0)
        direction = param_number(p, "direction", default=float(eng.environment.wind.direction))
        gust_for = duration or self.cfg.default_gust_duration_s
        rad = math.radians(direction)
        d.wind_disturbance = -speed * np.array([math.sin(rad), math.cos(rad), 0.0])   # wind comes FROM direction
        d.failures.add("wind_gust")
        return f"wind gust {speed:.0f} m/s from {direction % 360:.0f} deg for {gust_for:.0f} s", \
            {"speed": speed, "direction": direction % 360}, t + gust_for

    def _clear(self, cmd: Command) -> CommandResult:
        kind = cmd.params.get("type")
        if kind is not None and kind not in FAILURE_TYPES:
            raise CommandError(f"type must be one of {', '.join(FAILURE_TYPES)}")
        ids = set(cmd.drone_ids) if cmd.drone_ids else None
        ended = [a for a in self.active if (kind is None or a.kind == kind) and (ids is None or a.drone_id in ids)]
        for a in ended:
            self._end(a, "cleared by instructor")
        return CommandResult.ok(f"cleared {len(ended)} failure(s)")

    # ------------------------------------------------------------------ lifecycle
    def _end(self, a: ActiveFailure, why: str) -> None:
        if a in self.active:
            self.active.remove(a)
        eng = self.engine
        if a.drone_id not in eng.swarm:
            return
        d = eng.swarm.get(a.drone_id)
        if a.kind == "gps_loss" and eng.sensors is not None:
            eng.sensors.clear_gps_loss(d.id)
        elif a.kind == "comm_loss" and eng.comms is not None:
            eng.comms.clear_forced(d.id)
        elif a.kind == "wind_gust":
            d.wind_disturbance = np.zeros(3)
        elif a.kind == "motor" and a.params.get("severity") == "partial":
            d.clear_motor_failure()
        d.failures.discard(a.kind)
        eng.events.emit(EventCategory.DRONE, "failure_cleared", f"{a.kind} {why}", drone_id=d.id, failure=a.kind)

    def step(self, t: float) -> None:
        for a in [a for a in self.active if a.until is not None and t >= a.until]:
            self._end(a, "ended")
        alive = set(self.engine.swarm.ids)
        self.active = [a for a in self.active if a.drone_id in alive]

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.cfg.enabled, "active": [a.to_dict() for a in self.active], "injected": self.injected,
                "types": list(FAILURE_TYPES)}
