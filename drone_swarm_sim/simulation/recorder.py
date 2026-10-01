"""Run recording: ``telemetry.csv``, ``mission_events.json``, ``collision_events.json``, ``world.json``.

Each engine run (start or reset) writes into its own folder::

    logs/<run_id>/telemetry.csv
    logs/<run_id>/mission_events.json      system, drone, mission and command events
    logs/<run_id>/collision_events.json    separation violations and collisions
    logs/<run_id>/config.json              effective configuration (for replay / reproducibility)
    logs/<run_id>/world.json               world description + geofence (replay and reports)

``logs/simulation.log`` (see logging_setup.py) is shared by all runs.

Memory stays flat over long runs: telemetry rows and events are streamed to disk, never accumulated.
The event files are JSON arrays written incrementally (``[`` + one event per line + ``]`` on close);
while a run is still recording the closing bracket is missing, which :func:`read_json_array` tolerates.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from .events import EventBus, EventCategory, SimEvent

log = logging.getLogger(__name__)

TELEMETRY_COLUMNS = [
    "timestamp", "drone_id", "x", "y", "z", "lat", "lon", "alt",
    "vx", "vy", "vz", "speed", "heading", "roll", "pitch",
    "battery", "mode", "task", "armed", "health", "communication", "collision_state",
    # Stage 5 (replay / reports)
    "name", "source", "altitude_agl", "battery_state", "nearest_distance", "airborne",
]


def read_json_array(path: Path) -> list[dict[str, Any]]:
    """Read an event file, also while it is still being written (missing ``]``, partial last line)."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    items: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:            # a line being written right now
            continue
    return items


class TelemetryRecorder:
    def __init__(self, path: Path, rate_hz: float) -> None:
        self.path = path
        self.period = 1.0 / rate_hz
        self._next_time = 0.0
        self._file = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(TELEMETRY_COLUMNS)
        self._last_flush = time.monotonic()
        self.rows_written = 0

    def due(self, sim_time: float) -> bool:
        return sim_time + 1e-9 >= self._next_time

    def record(self, sim_time: float, telemetry: Iterable[dict[str, Any]]) -> None:
        self._next_time = sim_time + self.period
        for t in telemetry:
            p, v, g = t["position"], t["velocity"], t["gps"]
            nearest = t.get("nearest_distance")
            self._writer.writerow([
                round(sim_time, 3), t["drone_id"], p["x"], p["y"], p["z"], g["lat"], g["lon"], g["alt"],
                v["x"], v["y"], v["z"], t["speed"], t["heading"], t["roll"], t["pitch"],
                t["battery"], t["mode"], t["task"], int(t["armed"]), t["health"], t["communication"],
                t["collision_state"],
                t["name"], t.get("source", "sim"), t["altitude_agl"], t["battery_state"],
                "" if nearest is None else nearest, int(t["airborne"]),
            ])
            self.rows_written += 1
        now = time.monotonic()
        if now - self._last_flush > 1.0:
            self._file.flush()
            self._last_flush = now

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class _JsonArrayStream:
    """A JSON array on disk that grows one element per line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._file.write("[")
        self.count = 0

    def write(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self._file.write(("," if self.count else "") + "\n" + json.dumps(item, default=str))
            self.count += 1
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.write("\n]\n")
            self._file.close()


class EventRecorder:
    """Streams events from the bus into two JSON arrays (batched every ``FLUSH_INTERVAL_S``)."""

    FLUSH_INTERVAL_S = 2.0

    def __init__(self, directory: Path, events: EventBus) -> None:
        self.mission_path = directory / "mission_events.json"
        self.collision_path = directory / "collision_events.json"
        self._mission = _JsonArrayStream(self.mission_path)
        self._collision = _JsonArrayStream(self.collision_path)
        self._pending_mission: list[dict[str, Any]] = []
        self._pending_collision: list[dict[str, Any]] = []
        self._last_flush = time.monotonic()
        self._unsubscribe = events.subscribe(self._on_event)

    def _on_event(self, event: SimEvent) -> None:
        target = self._pending_collision if event.category == EventCategory.COLLISION else self._pending_mission
        target.append(event.to_dict())

    @property
    def pending(self) -> int:
        return len(self._pending_mission) + len(self._pending_collision)

    def flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_flush < self.FLUSH_INTERVAL_S:
            return
        self._last_flush = now
        try:
            if self._pending_mission:
                self._mission.write(self._pending_mission)
            if self._pending_collision:
                self._collision.write(self._pending_collision)
        except OSError:
            log.exception("Event recording failed")
        self._pending_mission = []
        self._pending_collision = []

    def close(self) -> None:
        self._unsubscribe()
        self.flush(force=True)
        self._mission.close()
        self._collision.close()


class SimulationRecorder:
    def __init__(self, base_dir: Path, run_id: str, events: EventBus, telemetry_rate_hz: float,
                 config: dict[str, Any], record_telemetry: bool = True) -> None:
        self.directory = base_dir / run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.telemetry = (TelemetryRecorder(self.directory / "telemetry.csv", telemetry_rate_hz)
                          if record_telemetry else None)
        self.events = EventRecorder(self.directory, events)
        log.info("Recording run %s to %s", run_id, self.directory)

    def write_world(self, world: dict[str, Any]) -> None:
        """``world.json``: written at the start of a run and again when it closes (final geofence)."""
        path = self.directory / "world.json"
        try:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(world, default=str), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            log.exception("Could not write %s", path)

    def on_step(self, sim_time: float, telemetry_provider) -> None:
        if self.telemetry is not None and self.telemetry.due(sim_time):
            self.telemetry.record(sim_time, telemetry_provider())
        self.events.flush()

    def close(self) -> None:
        if self.telemetry is not None:
            self.telemetry.close()
        self.events.close()
