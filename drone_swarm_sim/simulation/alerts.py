"""Operator alerts derived from simulation events.

Every event on the bus is classified into an alert priority:

* **CRITICAL**: CRITICAL events (collision, hard-floor breach, emergency, vehicle failure, battery
  emergency, geofence breach, lost link...). They stay until an operator acknowledges them.
* **WARNING**: WARNING events (battery warning / RTL failsafe, separation violation, leader promoted,
  drone released from a mission...). They expire after ``warning_ttl_s`` or can be acknowledged.
* **INFO**: mission started / completed / aborted. They expire after ``info_ttl_s``.

Command results are not alerts (the operator sees them as toasts and in the event log).

A repeat of the same alert (same kind and drones) within ``dedup_window_s`` updates the existing
alert and increments its ``count`` instead of adding a new one, so a flapping condition cannot flood
the panel. Alerts live in the engine, so every connected GCS sees the same acknowledgement state, and
``ack_alert`` is an ordinary validated, logged command.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .events import EventBus, EventCategory, Severity, SimEvent

if TYPE_CHECKING:
    from .commands import Command, CommandProcessor
    from .config import AlertsConfig
    from .drone_interface import CommandResult


class AlertPriority(StrEnum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


TITLES = {
    "collision": "Collision",
    "min_separation_breach": "Separation floor breached",
    "separation_violation": "Separation violation",
    "emergency": "Emergency",
    "failure": "Vehicle failure",
    "battery_state": "Battery",
    "battery_failsafe": "Battery failsafe",
    "geofence_breach": "Geofence breach",
    "leader_promoted": "Leader promoted",
    "leader_lost": "Leader lost",
    "offboard_timeout": "Setpoint timeout",
    "landed": "Unplanned landing",
    "comm_lost": "Link lost",
    "comm_degraded": "Link degraded",
    "failsafe": "Failsafe",
    "failure_injected": "Failure injected",
    "mission": "Mission",
}
INFO_MISSION_WORDS = ("started", "completed", "aborted")


@dataclass
class Alert:
    id: int
    time: float
    priority: AlertPriority
    kind: str
    title: str
    message: str
    drone_ids: list[int]
    key: tuple
    count: int = 1
    last_time: float = 0.0
    acknowledged: bool = False
    ack_time: float | None = None
    ack_by: str | None = None

    @property
    def requires_ack(self) -> bool:
        return self.priority == AlertPriority.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "time": round(self.time, 2), "last_time": round(self.last_time, 2),
                "priority": str(self.priority), "kind": self.kind, "title": self.title, "message": self.message,
                "drone_ids": self.drone_ids, "count": self.count, "acknowledged": self.acknowledged,
                "requires_ack": self.requires_ack, "ack_by": self.ack_by,
                "ack_time": round(self.ack_time, 2) if self.ack_time is not None else None}


def classify(event: SimEvent) -> AlertPriority | None:
    """Alert priority of an event, or None if the event is not an alert."""
    if event.category == EventCategory.COMMAND:
        return None
    if event.severity == Severity.CRITICAL:
        return AlertPriority.CRITICAL
    if event.severity == Severity.WARNING:
        return AlertPriority.WARNING
    if event.category == EventCategory.MISSION and event.kind == "mission" \
            and any(w in event.message for w in INFO_MISSION_WORDS):
        return AlertPriority.INFO
    return None


class AlertManager:
    def __init__(self, config: "AlertsConfig", events: EventBus, commands: "CommandProcessor | None" = None) -> None:
        self.config = config
        self.events = events
        self._ids = itertools.count(1)
        self.active: list[Alert] = []
        self.history: deque[dict[str, Any]] = deque(maxlen=config.max_alerts)   # every alert raised (reports)
        self.version = 0
        self.total = {p: 0 for p in AlertPriority}
        self._time = 0.0
        events.subscribe(self._on_event)
        if commands is not None:
            commands.register("ack_alert", self._ack_command,
                              "Acknowledge alerts {ids? [..], all? true} (CRITICAL alerts require it)")

    # ------------------------------------------------------------------ intake
    def _on_event(self, event: SimEvent) -> None:
        priority = classify(event)
        if priority is None:
            return
        ids = sorted({i for i in (event.drone_id, event.data.get("drone_a"), event.data.get("drone_b")) if i is not None})
        key = (event.kind, tuple(ids))
        now = event.time
        for alert in self.active:
            if (alert.key == key and not alert.acknowledged and alert.priority == priority
                    and now - alert.last_time <= self.config.dedup_window_s):
                alert.count += 1
                alert.last_time = now
                alert.message = event.message
                self.version += 1
                return
        title = TITLES.get(event.kind, event.kind.replace("_", " ").capitalize())
        alert = Alert(next(self._ids), now, priority, event.kind, title, event.message, ids, key, last_time=now)
        self.active.append(alert)
        self.total[priority] += 1
        self.history.append(alert.to_dict())
        self.version += 1
        if len(self.active) > self.config.max_active:
            self._trim()

    def _trim(self) -> None:
        """Drop the oldest non-critical (or acknowledged) alerts first when the panel is full."""
        while len(self.active) > self.config.max_active:
            victim = next((a for a in self.active if a.acknowledged or not a.requires_ack), self.active[0])
            self.active.remove(victim)

    # ------------------------------------------------------------------ lifecycle
    def step(self, t: float) -> None:
        self._time = t
        c = self.config
        keep = []
        for a in self.active:
            if a.acknowledged:
                alive = a.ack_time is not None and t - a.ack_time < c.acknowledged_ttl_s
            elif a.priority == AlertPriority.WARNING:
                alive = t - a.last_time < c.warning_ttl_s
            elif a.priority == AlertPriority.INFO:
                alive = t - a.last_time < c.info_ttl_s
            else:
                alive = True                                   # CRITICAL waits for the operator
            if alive:
                keep.append(a)
        if len(keep) != len(self.active):
            self.active = keep
            self.version += 1

    def acknowledge(self, ids: list[int] | None, by: str = "operator") -> int:
        n = 0
        for a in self.active:
            if not a.acknowledged and (ids is None or a.id in ids):
                a.acknowledged = True
                a.ack_time = self._time
                a.ack_by = by
                n += 1
        if n:
            self.version += 1
        return n

    def _ack_command(self, cmd: "Command") -> "CommandResult":
        from .commands import CommandError
        from .drone_interface import CommandResult
        p = cmd.params
        if p.get("all") is True:
            ids = None
        else:
            ids = p.get("ids")
            if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
                raise CommandError("give 'ids' (list of alert ids) or 'all': true")
            known = {a.id for a in self.active}
            missing = [i for i in ids if i not in known]
            if missing:
                raise CommandError(f"unknown alert id(s) {missing}")
        n = self.acknowledge(ids, by=str(p.get("user", "operator"))[:40])
        return CommandResult.ok(f"acknowledged {n} alert(s)", acknowledged=n)

    # ------------------------------------------------------------------ telemetry
    @property
    def unacked_critical(self) -> int:
        return sum(1 for a in self.active if a.requires_ack and not a.acknowledged)

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "unacked_critical": self.unacked_critical,
            "active": [a.to_dict() for a in self.active],
            "totals": {str(k): v for k, v in self.total.items()},
        }
