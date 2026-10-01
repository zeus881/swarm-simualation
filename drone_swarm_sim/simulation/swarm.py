"""Swarm manager: drone registry, perception, velocity pipeline and stepping.

Per tick (see docs/ARCHITECTURE.md §3.1):

1. **Perception**: rebuild the KD-tree, assign neighbour lists and collision states.
2. **Guidance**: every drone turns its flight mode into a desired velocity.
3. **Velocity pipeline**: :class:`VelocityStage` objects refine the desired
   velocities in ascending priority. Each stage sees the output of the
   lower-priority stages, so the highest-priority stage (collision avoidance)
   has the final word::

       Mission (10) -> Formation (20) -> Obstacle avoidance (30) -> Collision avoidance (40)

4. **Control + physics**: each drone integrates its final velocity command
   over ``physics_substeps`` sub-steps.
5. **Post-step**: failsafes and housekeeping.

Phase 1 ships with an empty pipeline. Phase 2 registers formation/flocking
and collision-avoidance stages without changing this module.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from .config import SimConfig
from .drone import Drone
from .environment import Environment
from .events import EventBus, EventCategory, Severity
from .geo import GeoReference
from .spatial import CollisionMonitor, CollisionReport, SpatialIndex, nearest_neighbor_distances
from .types import CollisionState, Vector3, vec3

log = logging.getLogger(__name__)


@dataclass(slots=True)
class SwarmContext:
    """Read-only view of the swarm handed to velocity-pipeline stages."""

    time: float
    dt: float
    drones: list[Drone]
    ids: list[int]
    positions: np.ndarray          # (N, 3)
    velocities: np.ndarray         # (N, 3)
    airborne: np.ndarray           # (N,) bool
    controllable: np.ndarray       # (N,) bool: in a mode that accepts swarm velocity shaping
    neighbors: list[np.ndarray]    # per drone: indices into the arrays above
    index: SpatialIndex
    environment: Environment
    config: SimConfig


class VelocityStage(ABC):
    """One stage of the swarm velocity pipeline."""

    name: str = "stage"
    priority: int = 0              # higher runs later (and therefore wins)
    enabled: bool = True

    @abstractmethod
    def apply(self, ctx: SwarmContext, velocities: np.ndarray) -> np.ndarray:
        """Return the refined (N, 3) velocity array. Must not modify ``ctx``."""


class SwarmManager:
    def __init__(self, config: SimConfig, environment: Environment, geo: GeoReference,
                 events: EventBus, rng: np.random.Generator) -> None:
        self.config = config
        self.environment = environment
        self.geo = geo
        self.events = events
        self._rng = rng
        self._drones: dict[int, Drone] = {}
        self._next_id = 1
        sw = config.swarm
        self.index = SpatialIndex(use_kdtree=sw.use_kdtree)
        self.collision_monitor = CollisionMonitor(sw.separation_distance, sw.warning_distance,
                                                  sw.collision_distance, events, min_separation=sw.min_separation)
        self._stages: list[VelocityStage] = []
        self.last_report: CollisionReport | None = None
        self.min_separation: float | None = None
        self.obstacle_collisions = 0
        self._pad_columns = max(1, math.ceil(math.sqrt(max(config.simulation.drone_count, 1))))

    # ------------------------------------------------------------------ registry
    def __len__(self) -> int:
        return len(self._drones)

    def __iter__(self) -> Iterator[Drone]:
        return iter(self._drones.values())

    def __contains__(self, drone_id: int) -> bool:
        return drone_id in self._drones

    @property
    def drones(self) -> list[Drone]:
        return list(self._drones.values())

    @property
    def ids(self) -> list[int]:
        return list(self._drones)

    def get(self, drone_id: int) -> Drone:
        try:
            return self._drones[int(drone_id)]
        except KeyError:
            raise KeyError(f"unknown drone id {drone_id}") from None

    def select(self, drone_ids: Iterable[int] | None) -> list[Drone]:
        """Resolve ids to drones; ``None`` selects the whole swarm."""
        if drone_ids is None:
            return self.drones
        return [self.get(i) for i in drone_ids]

    def pad_position(self, slot: int) -> Vector3:
        """Landing pad ``slot`` around the home base, per the configured layout."""
        h = self.config.home
        home = self.environment.home_position
        s = h.pad_spacing
        if h.pad_layout == "line":
            offset = slot * s, 0.0
        elif h.pad_layout == "circle":
            if slot == 0:
                offset = 0.0, 0.0
            else:
                # Rings of increasing radius, each holding as many pads as fit at spacing s.
                ring, idx = 1, slot - 1
                while idx >= max(6 * ring, 1):
                    idx -= 6 * ring
                    ring += 1
                angle = 2 * math.pi * idx / (6 * ring)
                offset = ring * s * math.cos(angle), ring * s * math.sin(angle)
        else:
            cols = self._pad_columns
            row, col = divmod(slot, cols)
            offset = (col - (cols - 1) / 2.0) * s, -(row - (cols - 1) / 2.0) * s
        x, y = home[0] + offset[0], home[1] + offset[1]
        return vec3(x, y, self.environment.ground_height(x, y))

    def _free_pad(self) -> Vector3:
        spacing = self.config.home.pad_spacing
        homes = [d.home_position for d in self._drones.values()]
        grounded = [d.position for d in self._drones.values() if not d.airborne]
        occupied = np.array(homes + grounded) if homes or grounded else np.zeros((0, 3))
        slot = 0
        while True:
            pad = self.pad_position(slot)
            if len(occupied) == 0 or np.min(np.linalg.norm(occupied[:, :2] - pad[:2], axis=1)) > spacing * 0.5:
                return pad
            slot += 1

    def add_drone(self, position: Vector3 | None = None, initial_battery: float | None = None) -> Drone:
        max_drones = self.config.simulation.max_drones
        if len(self._drones) >= max_drones:
            raise ValueError(f"swarm already has the maximum of {max_drones} drones")
        if position is None:
            home = self._free_pad()
        else:
            p = np.asarray(position, dtype=np.float64)
            home = self.environment.clamp(vec3(p[0], p[1], 0.0))
        if initial_battery is None:
            b = self.config.battery
            initial_battery = float(self._rng.uniform(b.initial_percent_min, b.initial_percent_max))
        drone_id = self._next_id
        self._next_id += 1
        drone = Drone(drone_id, self.config, self.environment, self.geo, home=home,
                      initial_battery=initial_battery, events=self.events)
        self._drones[drone_id] = drone
        self.events.emit(EventCategory.SYSTEM, "drone_added",
                         f"{drone.name} added at ({home[0]:.1f}, {home[1]:.1f}) battery {initial_battery:.0f}%",
                         drone_id=drone_id)
        return drone

    def add_remote(self, factory) -> Drone:
        """Register a real / SITL vehicle: ``factory(drone_id)`` builds the :class:`RemoteDrone`."""
        drone_id = self._next_id
        self._next_id += 1
        drone = factory(drone_id)
        self._drones[drone_id] = drone
        self.events.emit(EventCategory.SYSTEM, "drone_added", f"{drone.name} added ({drone.source} {drone.link.url})",
                         drone_id=drone_id)
        return drone

    def close(self) -> None:
        """Close hardware links (simulated drones have nothing to release)."""
        for d in self._drones.values():
            if getattr(d, "is_remote", False):
                d.close()

    def remove_drone(self, drone_id: int) -> Drone:
        drone = self.get(drone_id)
        if getattr(drone, "is_remote", False):
            drone.close()
        del self._drones[drone.id]
        self.events.emit(EventCategory.SYSTEM, "drone_removed", f"{drone.name} removed",
                         severity=Severity.WARNING if drone.airborne else Severity.INFO, drone_id=drone.id)
        return drone

    def spawn(self, count: int) -> list[Drone]:
        return [self.add_drone() for _ in range(count)]

    # ------------------------------------------------------------------ pipeline
    def add_stage(self, stage: VelocityStage) -> None:
        self._stages.append(stage)
        self._stages.sort(key=lambda s: s.priority)

    def remove_stage(self, name: str) -> None:
        self._stages = [s for s in self._stages if s.name != name]

    @property
    def stages(self) -> list[VelocityStage]:
        return list(self._stages)

    # ------------------------------------------------------------------ stepping
    def positions(self) -> np.ndarray:
        return np.array([d.position for d in self._drones.values()]).reshape(-1, 3)

    def step(self, t: float, dt: float, substeps: int = 1) -> None:
        drones = self.drones
        if not drones:
            self.last_report = None
            self.min_separation = None
            return
        positions = np.array([d.position for d in drones])
        airborne = np.fromiter((d.airborne for d in drones), dtype=bool, count=len(drones))
        neighbors = self._update_perception(drones, positions, airborne, t)

        desired = np.array([d.compute_guidance(t) for d in drones])
        if self._stages:
            ctx = SwarmContext(
                time=t, dt=dt, drones=drones, ids=[d.id for d in drones], positions=positions,
                velocities=np.array([d.velocity for d in drones]), airborne=airborne,
                controllable=np.fromiter((d.motors_on and d.airborne for d in drones), dtype=bool,
                                         count=len(drones)),
                neighbors=neighbors, index=self.index, environment=self.environment, config=self.config,
            )
            for stage in self._stages:
                if stage.enabled:
                    desired = stage.apply(ctx, desired)

        # Terrain height and wind are evaluated once per tick for the whole swarm (a drone moves well under a
        # metre in one tick), instead of once per drone and physics sub-step.
        env = self.environment
        ground = env.ground_heights(positions[:, 0], positions[:, 1])
        winds = env.wind.velocities_at(positions, ground)
        for i, drone in enumerate(drones):
            if drone.wind_disturbance.any():
                winds[i] += drone.wind_disturbance
        ground_list = ground.tolist()
        sub_dt = dt / substeps
        for _ in range(substeps):
            for i, drone in enumerate(drones):
                drone.integrate(desired[i], winds[i], sub_dt, ground_z=ground_list[i])
        t_end = t + dt
        self._check_obstacle_collisions(drones, t_end)
        for drone in drones:
            drone.post_step(t_end)

    def _check_obstacle_collisions(self, drones: list[Drone], t: float) -> None:
        """A drone whose body touches an obstacle crashes (fails, motors off). Runs even with avoidance off."""
        obstacles = self.environment.obstacles
        if not len(obstacles):
            return
        live = [d for d in drones if d.airborne and not d.failed and not getattr(d, "is_remote", False)]
        if not live:
            return
        dist, _, which = obstacles.nearest(np.array([d.position for d in live]), 5.0)
        limit = self.config.drone.radius + self.config.obstacles.collision_margin
        for d, gap, k in zip(live, dist, which):
            if gap < limit:
                name = obstacles.obstacles[int(k)].name
                self.obstacle_collisions += 1
                self.events.emit(EventCategory.COLLISION, "obstacle_collision", f"{d.name} hit {name}",
                                 severity=Severity.CRITICAL, drone_id=d.id, time=t, obstacle=name)
                d.crash(f"collision with {name}")

    def _update_perception(self, drones: list[Drone], positions: np.ndarray, airborne: np.ndarray,
                           t: float) -> list[np.ndarray]:
        sw = self.config.swarm
        ids = [d.id for d in drones]
        self.index.rebuild(positions)
        neighbors = self.index.query_neighbors(sw.neighbor_radius, sw.max_neighbors)
        report = self.collision_monitor.update(ids, self.index, airborne, t)
        self.last_report = report
        # Link-quality estimate from the range to the GCS at the home base (replaced by the communication
        # model when it is enabled): 100 % close in, falling quadratically to 0 at the configured range.
        gcs = self.environment.home_position
        ranges = np.linalg.norm(positions - gcs, axis=1)
        quality = 100.0 * np.clip(1.0 - (ranges / self.config.communication.range) ** 2, 0.0, 1.0)
        # True distance to the nearest *airborne* drone (the collision report only covers the warning radius).
        nearest = np.full(len(drones), np.inf)
        air = np.flatnonzero(airborne)
        if len(air) >= 2:
            nearest[air] = nearest_neighbor_distances(positions[air])
        self.min_separation = float(nearest[air].min()) if len(air) >= 2 else None
        for i, drone in enumerate(drones):
            drone.neighbors = [ids[j] for j in neighbors[i]]
            drone.collision_state = report.states[i] if airborne[i] else CollisionState.CLEAR
            drone.nearest_distance = float(nearest[i]) if np.isfinite(nearest[i]) else None
            if not drone.comm_managed:
                drone.link_quality = float(quality[i])
        return neighbors

    # ------------------------------------------------------------------ telemetry
    def telemetry(self) -> list[dict]:
        drones = self.drones
        if not drones:
            return []
        lat, lon, alt = self.geo.enu_to_geodetic(self.positions())   # one vectorised conversion
        return [
            d.get_telemetry(gps=(float(lat[i]), float(lon[i]), float(alt[i]))).to_dict()
            for i, d in enumerate(drones)
        ]

    def summary(self) -> dict:
        drones = self.drones
        report = self.last_report
        flying = [d for d in drones if d.airborne]
        left = [d.battery.flight_time_left_s(True, d.cfg.mass_kg, d.cfg.payload_kg) for d in flying]
        return {
            "total": len(drones),
            "airborne": sum(d.airborne for d in drones),
            "armed": sum(d.armed for d in drones),
            "failed": sum(d.failed for d in drones),
            "collision_warnings": sum(d.collision_state != CollisionState.CLEAR for d in drones),
            # smallest current distance between two airborne drones (None with fewer than two airborne)
            "min_separation": round(self.min_separation, 2) if self.min_separation is not None else None,
            "separation_violations": self.collision_monitor.total_violations,
            "min_separation_breaches": self.collision_monitor.total_hard_violations,
            "lowest_separation": (round(self.collision_monitor.lowest_separation, 2)
                                  if math.isfinite(self.collision_monitor.lowest_separation) else None),
            "collisions": self.collision_monitor.total_collisions,
            "obstacle_collisions": self.obstacle_collisions,
            "average_battery": round(float(np.mean([d.battery.percent for d in drones])), 1) if drones else None,
            "min_battery": round(float(min(d.battery.percent for d in drones)), 1) if drones else None,
            # Predicted flight time of the airborne swarm: when the first drone reaches its emergency threshold.
            "flight_time_left_s": round(float(min(left)), 0) if left else None,
            "mean_flight_time_left_s": round(float(np.mean(left)), 0) if left else None,
        }
