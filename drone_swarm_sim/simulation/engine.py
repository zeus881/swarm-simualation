"""Deterministic simulation engine.

The engine owns the complete world state and advances it with a fixed
timestep. It is single-threaded and performs no I/O except recording, so a
run is reproducible from (configuration, seed, command sequence). Real-time
pacing, threading and networking live in :mod:`simulation.runner` and
:mod:`backend`.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .alerts import AlertManager
from .commands import Command, CommandProcessor
from .communication import CommModel
from .failures import FailureInjector
from .sensors import SensorSuite
from .config import SimConfig
from .drone_interface import CommandResult
from .environment import Environment
from .events import EventBus, EventCategory, Severity
from .geo import GeoReference
from .recorder import SimulationRecorder
from .swarm import SwarmManager

log = logging.getLogger(__name__)


class SimulationEngine:
    def __init__(self, config: SimConfig, *, record: bool | None = None, log_dir: Path | None = None) -> None:
        self.config = config
        self._record = config.logging.record_telemetry if record is None else record
        self._record_events = record is not False
        self._log_dir = log_dir or config.logging.path
        self.recorder: SimulationRecorder | None = None
        self.run_id = ""
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """(Re)build the world from configuration. Starts a new recorded run."""
        self.close()
        cfg = self.config
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        self.sim_time = 0.0
        self.tick = 0
        self.rng = np.random.default_rng(cfg.simulation.seed)
        self.geo = GeoReference(cfg.origin.latitude, cfg.origin.longitude, cfg.origin.altitude)
        self.events = EventBus(history=cfg.telemetry.event_history, clock=lambda: self.sim_time)
        self.environment = Environment(cfg, self.rng)
        self.swarm = SwarmManager(cfg, self.environment, self.geo, self.events, self.rng)
        # Stage 4 realism models (each optional): GCS link, sensors + navigation filter, failure injection.
        self.comms = (CommModel(cfg.communication, self.events, self.rng, self.environment.home_position)
                      if cfg.communication.enabled else None)
        self.sensors = SensorSuite(cfg.sensors, self.rng) if cfg.sensors.enabled else None
        if self.sensors is not None:
            from estimation.navigation import Navigator
            self.navigator = Navigator(cfg.sensors)
        else:
            self.navigator = None
        self.commands = CommandProcessor(self)
        self.alerts = AlertManager(cfg.alerts, self.events, self.commands)
        self.failures = FailureInjector(self)
        # Swarm behaviours (formation, flocking, collision avoidance) plug into the swarm's
        # velocity pipeline and register their own commands. Imported here: algorithms/ builds on
        # simulation/, so the dependency points one way only at import time.
        from algorithms.coordinator import SwarmCoordinator
        self.coordinator = SwarmCoordinator(self)
        # Missions, groups and the geofence sit above the swarm behaviours (same one-way import rule).
        from missions.manager import MissionManager
        self.missions = MissionManager(self)
        self.step_time_ms = 0.0
        self.max_step_time_ms = 0.0
        if self._record_events:
            try:
                self.recorder = SimulationRecorder(
                    self._log_dir, self.run_id, self.events, cfg.logging.telemetry_record_rate_hz,
                    cfg.to_dict(), record_telemetry=self._record)
            except OSError:
                log.exception("Could not create run recorder - continuing without recording")
                self.recorder = None
        self.events.emit(EventCategory.SYSTEM, "reset", f"simulation initialised (run {self.run_id})")
        self.swarm.spawn(cfg.simulation.drone_count)
        if cfg.hardware.vehicles:
            # Real / SITL vehicles join the same swarm (links start in their own threads).
            from integration import create_remote_drone
            for entry in cfg.hardware.vehicles:
                try:
                    self.swarm.add_remote(lambda i, e=entry: create_remote_drone(e, i, cfg, self.environment, self.geo,
                                                                                 self.events))
                except Exception as exc:
                    log.exception("Hardware vehicle %s failed to start", entry.get("url"))
                    self.events.emit(EventCategory.SYSTEM, "hardware_error",
                                     f"{entry.get('type')} {entry.get('url')}: {exc}", severity=Severity.WARNING)
        if self.recorder is not None:
            self.recorder.write_world(self._world_record())

    def _world_record(self) -> dict[str, Any]:
        """What a replay / report needs besides telemetry and events (``logs/<run>/world.json``)."""
        missions = getattr(self, "missions", None)
        return {"run_id": self.run_id, "sim_time": round(self.sim_time, 3), "world": self.world_info(),
                "geofence": missions.snapshot().get("geofence") if missions is not None else None,
                "drones": [{"id": d.id, "name": d.name, "source": getattr(d, "source", "sim")} for d in self.swarm]}

    def close(self) -> None:
        swarm = getattr(self, "swarm", None)
        if self.recorder is not None:
            try:
                self.recorder.write_world(self._world_record())
            except Exception:  # pragma: no cover - never block shutdown on the summary file
                log.exception("Could not record the final world state")
        if swarm is not None:
            swarm.close()
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None

    @property
    def dt(self) -> float:
        return self.config.dt

    # ------------------------------------------------------------------ stepping
    def step(self, n: int = 1) -> None:
        substeps = self.config.simulation.physics_substeps
        for _ in range(n):
            start = time.perf_counter()
            dt = self.dt
            self.environment.step(dt)
            self.swarm.step(self.sim_time, dt, substeps)
            self.tick += 1
            # Recompute from the tick count instead of accumulating to avoid float drift.
            self.sim_time = self.tick * dt
            # Real vehicles have their own sensors, estimator and radio: the models apply to simulated drones only.
            drones = [d for d in self.swarm.drones if not getattr(d, "is_remote", False)]
            if self.sensors is not None:
                measurements = self.sensors.measure(self.sim_time, dt, drones)
                self.navigator.step(dt, drones, measurements)
            if self.comms is not None:
                self.comms.step(self.sim_time, drones, self._capture_telemetry)
            self.failures.step(self.sim_time)
            self.missions.step(self.sim_time)          # geofence + mission executors (after physics)
            self.alerts.step(self.sim_time)
            if self.recorder is not None:
                self.recorder.on_step(self.sim_time, self.swarm.telemetry)
            elapsed = (time.perf_counter() - start) * 1000.0
            # Exponential moving average for display; max for diagnostics.
            self.step_time_ms = elapsed if self.tick == 1 else 0.9 * self.step_time_ms + 0.1 * elapsed
            self.max_step_time_ms = max(self.max_step_time_ms, elapsed)

    def run_for(self, seconds: float) -> None:
        """Advance ``seconds`` of simulated time as fast as possible."""
        self.step(max(0, math.ceil(seconds / self.dt - 1e-9)))

    def execute(self, command: Command | Mapping[str, Any]) -> CommandResult:
        return self.commands.execute(command)

    @staticmethod
    def _drone_telemetry(drone) -> dict[str, Any]:
        return drone.get_telemetry().to_dict()

    def _capture_telemetry(self, drones: list) -> list:
        """Telemetry records for ``drones`` with one vectorised geodetic conversion (link-model downlink)."""
        lat, lon, alt = self.geo.enu_to_geodetic(np.array([d.position for d in drones]).reshape(-1, 3))
        return [d.get_telemetry(gps=(float(lat[i]), float(lon[i]), float(alt[i]))) for i, d in enumerate(drones)]

    def gcs_telemetry(self) -> list[dict[str, Any]]:
        """Drone telemetry as the GCS sees it: delayed / frozen through the link model when it is enabled,
        the ground truth otherwise. The recorder always stores the truth."""
        if self.comms is None:
            return self.swarm.telemetry()
        return [self._drone_telemetry(d) if getattr(d, "is_remote", False)
                else self.comms.view(d, self.sim_time, self._drone_telemetry) for d in self.swarm]

    # ------------------------------------------------------------------ output
    def world_info(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "origin": self.geo.to_dict(),
            **self.environment.describe(),
            "swarm": {
                "separation_distance": cfg.swarm.separation_distance,
                "warning_distance": cfg.swarm.warning_distance,
                "neighbor_radius": cfg.swarm.neighbor_radius,
                "formation": (str(self.coordinator.formation.shape)
                              if self.coordinator.formation.active else None),
                "mode": str(self.coordinator.mode),
            },
            "battery_thresholds": {
                "warning": cfg.battery.warning,
                "return_home": cfg.battery.return_home,
                "emergency": cfg.battery.emergency,
            },
            "simulation_rate": cfg.simulation.simulation_rate,
            "max_drones": cfg.simulation.max_drones,
            "alerts": {"critical_repeat_s": cfg.alerts.critical_repeat_s},
            "mission_defaults": {
                "altitude": cfg.mission.default_altitude, "speed": cfg.mission.default_speed,
                "loiter_radius": cfg.mission.loiter_radius, "camera_hfov_deg": cfg.mission.camera_hfov_deg,
                "survey_overlap": cfg.mission.survey_overlap, "survey_finish": cfg.mission.survey_finish,
                "max_speed": cfg.drone.max_horizontal_speed, "min_altitude": cfg.drone.min_altitude,
            },
        }

    def snapshot(self, since_event_seq: int = 0) -> dict[str, Any]:
        """Complete JSON-serialisable state of the world."""
        return {
            "run_id": self.run_id,
            "sim_time": round(self.sim_time, 3),
            "tick": self.tick,
            "world": self.world_info(),
            "wind": self.environment.wind.snapshot(),
            "summary": self.swarm.summary(),
            "swarm_control": self.coordinator.snapshot(),
            "missions": self.missions.snapshot(),
            "alerts": self.alerts.snapshot(),
            "failures": self.failures.snapshot(),
            "comms": self.comms.snapshot() if self.comms is not None else {"enabled": False},
            "sensors": {"enabled": self.sensors is not None, "control_source": self.config.sensors.control_source},
            "engine": {"step_ms": round(self.step_time_ms, 3), "max_step_ms": round(self.max_step_time_ms, 3)},
            "drones": self.gcs_telemetry(),
            # A short tail is enough: clients de-duplicate by seq and snapshots arrive at 20 Hz.
            "events": [e.to_dict() for e in self.events.recent(since_event_seq, limit=25)],
            "last_event_seq": self.events.last_seq,
        }
