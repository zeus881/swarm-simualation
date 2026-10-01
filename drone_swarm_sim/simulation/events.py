"""Simulation event bus.

Events are small immutable records (mode changes, failsafes, collisions,
operator commands...). The engine thread publishes them; subscribers such as
the recorder persist them and the UI receives the most recent ones inside each
telemetry snapshot.
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

log = logging.getLogger(__name__)


class EventCategory(StrEnum):
    SYSTEM = "SYSTEM"
    DRONE = "DRONE"
    MISSION = "MISSION"
    COMMAND = "COMMAND"
    COLLISION = "COLLISION"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(slots=True)
class SimEvent:
    time: float
    category: EventCategory
    kind: str
    message: str
    severity: Severity = Severity.INFO
    drone_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "time": round(self.time, 3),
            "category": str(self.category),
            "kind": self.kind,
            "severity": str(self.severity),
            "drone_id": self.drone_id,
            "message": self.message,
            "data": self.data,
        }


Subscriber = Callable[[SimEvent], None]

_LOG_LEVEL = {Severity.INFO: logging.INFO, Severity.WARNING: logging.WARNING, Severity.CRITICAL: logging.ERROR}


class EventBus:
    """Synchronous publish/subscribe with a bounded history."""

    def __init__(self, history: int = 500, clock: Callable[[], float] | None = None) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: deque[SimEvent] = deque(maxlen=history)
        self._seq = itertools.count(1)
        self._clock = clock or (lambda: 0.0)

    def set_clock(self, clock: Callable[[], float]) -> None:
        self._clock = clock

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def publish(self, event: SimEvent) -> SimEvent:
        event.seq = next(self._seq)
        self._history.append(event)
        log.log(_LOG_LEVEL[event.severity], "[t=%.2f] %s/%s %s%s", event.time, event.category, event.kind,
                f"D{event.drone_id:02d} " if event.drone_id is not None else "", event.message)
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:  # a faulty subscriber must never stop the simulation
                log.exception("Event subscriber %r failed", callback)
        return event

    def emit(
        self,
        category: EventCategory,
        kind: str,
        message: str,
        *,
        severity: Severity = Severity.INFO,
        drone_id: int | None = None,
        time: float | None = None,
        **data: Any,
    ) -> SimEvent:
        return self.publish(
            SimEvent(
                time=self._clock() if time is None else time,
                category=category,
                kind=kind,
                message=message,
                severity=severity,
                drone_id=drone_id,
                data=data,
            )
        )

    def recent(self, since_seq: int = 0, limit: int = 50) -> list[SimEvent]:
        items = [e for e in self._history if e.seq > since_seq]
        return items[-limit:]

    @property
    def last_seq(self) -> int:
        return self._history[-1].seq if self._history else 0
