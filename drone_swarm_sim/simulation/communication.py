"""GCS <-> drone communication model: range, packet loss, latency and jitter, heartbeats, link-loss
failsafe, command delivery with retries, and the delayed telemetry the GCS actually sees.

Per drone and per heartbeat (``heartbeat_hz``), one packet goes each way:

* **Success probability** ``p = (1 - p_loss) * e(r)`` with ``r`` the range to the GCS antenna and the
  edge factor ``e = 1`` below ``edge_fraction * range``, falling linearly to ``0`` at ``range`` (no link
  beyond). Link quality shown to the operator is ``100 * p``, smoothed over recent packets.
* **Delivery** after ``latency + jitter * N(0, 1)`` (clipped at 0) through a time-ordered queue.
* **Uplink** heartbeats keep the drone's link alive: no heartbeat for ``timeout_s`` -> ``COMM LOST``
  (CRITICAL event). After ``failsafe_timeout_s`` in LOST the drone runs ``failsafe_action``
  (RTL | LAND | HOLD) once per loss episode. A restored link is logged.
* **Downlink** packets carry the drone's telemetry captured *when sent*; the GCS view of a drone is the
  last delivered packet (so it is ``latency`` old, and frozen while the link is down), with its age.
* **Commands** from the operator travel the uplink: rejected immediately if the GCS has not heard the
  drone for ``timeout_s``; otherwise each attempt is lost with probability ``1 - p`` and retried every
  ``retry_interval_s`` up to ``command_retries`` times. A delivered command runs on the drone and its
  result comes back as a ``command_ack`` event (an unacknowledged command is a WARNING).

The model uses the engine RNG, so runs stay reproducible.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .drone_interface import CommandResult
from .events import EventBus, EventCategory, Severity
from .types import CommStatus, FlightMode

if TYPE_CHECKING:
    from .config import CommunicationConfig
    from .drone import Drone

QUALITY_WINDOW = 20        # packets used for the smoothed link quality


@dataclass
class LinkState:
    last_uplink: float = 0.0               # when the drone last heard the GCS
    last_downlink: float = 0.0             # when the GCS last heard the drone
    view: dict[str, Any] | None = None     # last telemetry delivered to the GCS
    view_time: float = 0.0                 # when that telemetry was captured on the drone
    outcomes: deque = field(default_factory=lambda: deque(maxlen=QUALITY_WINDOW))
    lost_since: float | None = None
    failsafe_done: bool = False
    forced_until: float = -math.inf        # injected comm loss
    status: CommStatus = CommStatus.ONLINE


class CommModel:
    def __init__(self, config: "CommunicationConfig", events: EventBus, rng: np.random.Generator,
                 gcs_position: np.ndarray) -> None:
        self.cfg = config
        self.events = events
        self.rng = rng
        self.gcs = np.asarray(gcs_position, dtype=np.float64) + np.array([0.0, 0.0, config.gcs_height])
        self.links: dict[int, LinkState] = {}
        self._queue: list[tuple[float, int, str, int, Any]] = []
        self._seq = itertools.count()
        self._next_beat = 0.0
        self.sent = 0
        self.delivered = 0
        self.commands_lost = 0

    # ------------------------------------------------------------------ geometry
    def success_probability(self, position: np.ndarray) -> float:
        c = self.cfg
        r = float(np.linalg.norm(np.asarray(position) - self.gcs))
        if r >= c.range:
            return 0.0
        edge = c.edge_fraction * c.range
        factor = 1.0 if r <= edge else max(0.0, (c.range - r) / (c.range - edge))
        return (1.0 - c.packet_loss) * factor

    def _latency(self) -> float:
        c = self.cfg
        return max(0.0, (c.latency_ms + c.jitter_ms * float(self.rng.standard_normal())) / 1000.0)

    def _push(self, when: float, kind: str, drone_id: int, payload: Any = None) -> None:
        heapq.heappush(self._queue, (when, next(self._seq), kind, drone_id, payload))

    def link(self, drone_id: int, t: float = 0.0) -> LinkState:
        state = self.links.get(drone_id)
        if state is None:
            state = self.links[drone_id] = LinkState(last_uplink=t, last_downlink=t)
        return state

    def force_down(self, drone_id: int, until: float) -> None:
        """Failure injection: no packet gets through until ``until``."""
        self.link(drone_id).forced_until = until

    def clear_forced(self, drone_id: int) -> None:
        self.link(drone_id).forced_until = -math.inf

    # ------------------------------------------------------------------ tick
    def step(self, t: float, drones: list["Drone"], capture: Callable[[list["Drone"]], list[Any]]) -> None:
        """Advance the link model. ``capture(drones)`` returns telemetry records (objects with ``to_dict``)
        for the drones whose downlink packet got through this heartbeat (batched for speed)."""
        c = self.cfg
        alive = {d.id for d in drones}
        for gone in [i for i in self.links if i not in alive]:
            del self.links[gone]
        by_id = {d.id: d for d in drones}
        if t + 1e-9 >= self._next_beat:
            self._next_beat = t + 1.0 / c.heartbeat_hz
            downlinked: list["Drone"] = []
            for d in drones:
                link = self.link(d.id, t)
                p = 0.0 if t < link.forced_until else self.success_probability(d.position)
                for kind in ("up", "down"):
                    self.sent += 1
                    ok = self.rng.random() < p
                    link.outcomes.append(ok)
                    if ok and kind == "up":
                        self._push(t + self._latency(), "up", d.id)
                    elif ok:
                        downlinked.append(d)
            # Telemetry captured when sent (so the GCS sees it one latency late), for all senders at once.
            for d, record in zip(downlinked, capture(downlinked) if downlinked else []):
                self._push(t + self._latency(), "down", d.id, (t, record))
        while self._queue and self._queue[0][0] <= t + 1e-9:
            _, _, kind, drone_id, payload = heapq.heappop(self._queue)
            link = self.links.get(drone_id)
            drone = by_id.get(drone_id)
            if link is None or drone is None:
                continue
            self.delivered += 1
            if kind == "up":
                link.last_uplink = t
            elif kind == "down":
                link.last_downlink = t
                link.view_time, link.view = payload
            elif kind == "cmd":
                self._deliver_command(drone, payload, t)
            elif kind == "cmd_retry":
                self._attempt(drone, payload, t)
        for d in drones:
            self._update_status(d, self.link(d.id, t), t)

    def _update_status(self, d: "Drone", link: LinkState, t: float) -> None:
        c = self.cfg
        quality = 100.0 * (sum(link.outcomes) / len(link.outcomes)) if link.outcomes else 100.0
        d.link_quality = quality
        d.comm_managed = True
        if t - link.last_uplink > c.timeout_s:
            status = CommStatus.LOST
        elif quality < c.degraded_quality:
            status = CommStatus.DEGRADED
        else:
            status = CommStatus.ONLINE
        if status != link.status:
            old, link.status = link.status, status
            if status == CommStatus.LOST:
                link.lost_since, link.failsafe_done = t, False
                self.events.emit(EventCategory.DRONE, "comm_lost",
                                 f"link lost (no GCS heartbeat for {c.timeout_s:.1f} s)", severity=Severity.CRITICAL,
                                 drone_id=d.id, time=t)
            elif old == CommStatus.LOST:
                self.events.emit(EventCategory.DRONE, "comm_restored",
                                 f"link restored after {t - (link.lost_since or t):.1f} s", severity=Severity.INFO,
                                 drone_id=d.id, time=t)
                link.lost_since = None
            elif status == CommStatus.DEGRADED:
                self.events.emit(EventCategory.DRONE, "comm_degraded", f"link degraded ({quality:.0f} %)",
                                 severity=Severity.WARNING, drone_id=d.id, time=t)
        d.comm_status = status
        if (status == CommStatus.LOST and not link.failsafe_done and link.lost_since is not None
                and t - link.lost_since >= c.failsafe_timeout_s):
            link.failsafe_done = True
            self._failsafe(d, t)

    def _failsafe(self, d: "Drone", t: float) -> None:
        action = self.cfg.failsafe_action
        if not d.in_flight or d.flight_mode in (FlightMode.RTL, FlightMode.LAND, FlightMode.EMERGENCY):
            return
        result = d.return_to_home() if action == "RTL" else d.land() if action == "LAND" else d.hover()
        self.events.emit(EventCategory.DRONE, "failsafe",
                         f"link-loss failsafe after {self.cfg.failsafe_timeout_s:.0f} s -> {action} ({result.message})",
                         severity=Severity.CRITICAL, drone_id=d.id, time=t, action=action)

    # ------------------------------------------------------------------ commands
    def gcs_has_link(self, drone_id: int, t: float) -> bool:
        link = self.links.get(drone_id)
        return link is None or t - link.last_downlink <= self.cfg.timeout_s

    def send(self, drone: "Drone", name: str, action: Callable[[], CommandResult], t: float) -> CommandResult:
        """Queue ``action`` for delivery over the uplink. Returns immediately (like a MAVLink send)."""
        if not self.gcs_has_link(drone.id, t):
            return CommandResult.fail(f"no link to {drone.name}")
        self._attempt(drone, {"name": name, "action": action, "tries": 0}, t)
        return CommandResult.ok(f"{name} sent")

    def _attempt(self, drone: "Drone", cmd: dict[str, Any], t: float) -> None:
        link = self.link(drone.id, t)
        cmd["tries"] += 1
        p = 0.0 if t < link.forced_until else self.success_probability(drone.position)
        if self.rng.random() < p:
            self._push(t + self._latency(), "cmd", drone.id, cmd)
        elif cmd["tries"] <= self.cfg.command_retries:
            self._push(t + self.cfg.retry_interval_s, "cmd_retry", drone.id, cmd)
        else:
            self.commands_lost += 1
            self.events.emit(EventCategory.COMMAND, "command_lost",
                             f"{cmd['name']} to {drone.name} not acknowledged after {cmd['tries']} attempt(s)",
                             severity=Severity.WARNING, drone_id=drone.id, time=t)

    def _deliver_command(self, drone: "Drone", cmd: dict[str, Any], t: float) -> None:
        try:
            result = cmd["action"]()
        except Exception as exc:          # a delivered command must never break the engine tick
            result = CommandResult.fail(f"failed on board ({exc.__class__.__name__}: {exc})")
        self.events.emit(EventCategory.COMMAND, "command_ack", f"{cmd['name']} {drone.name}: {result.message}",
                         severity=Severity.INFO if result.success else Severity.WARNING, drone_id=drone.id, time=t,
                         success=result.success, command=cmd["name"])

    # ------------------------------------------------------------------ GCS view
    def view(self, drone: "Drone", t: float, fallback: Callable[["Drone"], dict[str, Any]]) -> dict[str, Any]:
        """Telemetry of ``drone`` as the GCS currently knows it (delayed; frozen while the link is down)."""
        link = self.links.get(drone.id)
        if link is None or link.view is None:
            data = fallback(drone)
            data["telemetry_age_s"] = 0.0
        else:
            data = link.view.to_dict()
            data["telemetry_age_s"] = round(t - link.view_time, 2)
        data["communication"] = str(link.status) if link is not None else data.get("communication")
        data["link_quality"] = round(drone.link_quality, 0)
        return data

    def snapshot(self) -> dict[str, Any]:
        lost = [i for i, lk in self.links.items() if lk.status == CommStatus.LOST]
        return {"enabled": True, "range": self.cfg.range, "lost": lost, "sent": self.sent, "delivered": self.delivered,
                "commands_lost": self.commands_lost, "queue": len(self._queue)}
