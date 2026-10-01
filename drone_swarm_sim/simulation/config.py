"""Typed configuration system.

Configuration is described by nested dataclasses whose defaults are the
documented baseline. A YAML file overrides those defaults, and CLI style
``section.key=value`` overrides take precedence over the YAML file.

* Unknown keys inside a known section raise :class:`ConfigError` (typo protection).
* Unknown top-level sections are preserved in :attr:`SimConfig.extensions` with a
  warning, so configuration files written for later phases still load.
* Values are type-checked and coerced (``int`` -> ``float`` where a float is expected).
* Every section validates its own ranges; :meth:`SimConfig.validate` adds
  cross-section rules.
"""

from __future__ import annotations

import logging
import os
import types
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Union, get_args, get_origin, get_type_hints

import yaml

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation.yaml"
CONFIG_ENV_VAR = "SWARM_CONFIG"


class ConfigError(ValueError):
    """Raised for malformed or invalid configuration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


# --------------------------------------------------------------------------- sections


@dataclass
class SimulationConfig:
    drone_count: int = 10
    simulation_rate: float = 30.0          # engine ticks per simulated second
    physics_substeps: int = 2              # physics integrations per tick
    real_time_factor: float = 1.0          # 2.0 = run twice as fast as wall clock
    seed: int | None = 42                  # None -> non-deterministic
    auto_start: bool = True
    max_drones: int = 200

    def validate(self) -> None:
        _require(self.drone_count >= 0, "simulation.drone_count must be >= 0")
        _require(self.max_drones >= 1, "simulation.max_drones must be >= 1")
        _require(self.drone_count <= self.max_drones, "simulation.drone_count exceeds simulation.max_drones")
        _require(1.0 <= self.simulation_rate <= 1000.0, "simulation.simulation_rate must be in [1, 1000] Hz")
        _require(1 <= self.physics_substeps <= 50, "simulation.physics_substeps must be in [1, 50]")
        _require(0.01 <= self.real_time_factor <= 100.0, "simulation.real_time_factor must be in [0.01, 100]")


@dataclass
class OriginConfig:
    """Geodetic origin of the local ENU frame (default: PX4 SITL default home)."""

    latitude: float = 47.397742
    longitude: float = 8.545594
    altitude: float = 488.0

    def validate(self) -> None:
        _require(-90.0 <= self.latitude <= 90.0, "origin.latitude must be in [-90, 90]")
        _require(-180.0 <= self.longitude <= 180.0, "origin.longitude must be in [-180, 180]")


@dataclass
class EnvironmentConfig:
    size_x: float = 2000.0                 # world extent East-West [m], centred on origin
    size_y: float = 2000.0                 # world extent North-South [m]
    max_altitude: float = 300.0            # geofence ceiling above ground [m]
    ground_level: float = 0.0              # base terrain height in ENU z [m]
    scene_file: str | None = None          # YAML obstacle scene (buildings, towers, trees); null = none

    def validate(self) -> None:
        _require(self.size_x > 0 and self.size_y > 0, "environment.size_x/size_y must be > 0")
        _require(self.max_altitude > 5.0, "environment.max_altitude must be > 5 m")
        _require(self.scene_file is None or bool(self.scene_file.strip()), "environment.scene_file must be a path or null")

    @property
    def scene_path(self) -> Path | None:
        if not self.scene_file:
            return None
        p = Path(self.scene_file)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class TerrainConfig:
    """Terrain height field (Stage 4)."""

    enabled: bool = False
    source: str = "procedural"             # procedural | file
    file: str = ""                         # heightmap: .asc (ESRI grid) | .png (8/16-bit gray) | .npy | .csv
    max_height: float = 30.0               # procedural hill height / height of the brightest PNG pixel [m]
    vertical_scale: float = 1.0            # multiplies file heights
    offset: float = 0.0                    # added to every height [m]
    resolution: float = 10.0               # procedural grid cell [m]
    seed: int = 7
    hills: int = 8
    hill_radius: float = 180.0             # typical procedural hill radius [m]
    flatten_radius: float = 120.0          # level the ground around home (landing pads) [m]; 0 = off

    def validate(self) -> None:
        _require(self.source in ("procedural", "file"), "terrain.source must be procedural|file")
        _require(self.source != "file" or bool(self.file.strip()), "terrain.file is required when source is 'file'")
        _require(0.0 <= self.max_height <= 5000.0, "terrain.max_height must be in [0, 5000] m")
        _require(self.vertical_scale > 0, "terrain.vertical_scale must be > 0")
        _require(1.0 <= self.resolution <= 200.0, "terrain.resolution must be in [1, 200] m")
        _require(0 <= self.hills <= 200 and self.hill_radius > 0, "terrain.hills must be in [0, 200], hill_radius > 0")
        _require(self.flatten_radius >= 0, "terrain.flatten_radius must be >= 0")


@dataclass
class ObstaclesConfig:
    """Obstacle avoidance and path planning (Stage 4)."""

    avoidance: bool = True                 # obstacle-avoidance velocity stage (priority 30)
    clearance: float = 6.0                 # distance kept from obstacle surfaces [m]
    influence: float = 25.0                # obstacles closer than this are considered [m]
    terrain_clearance: float = 2.0         # minimum height above terrain kept in flight (except landing) [m]
    collision_margin: float = 0.3          # a drone closer than radius + margin to a surface has crashed [m]
    plan_paths: bool = True                # route gotos and mission legs around obstacles
    planner: str = "astar"                 # astar (2-D grid at flight altitude) | rrtstar (3-D sampling)
    grid_resolution: float = 4.0           # A* cell [m]
    max_grid_cells: int = 90000            # A* grid is coarsened to stay under this
    rrt_iterations: int = 2500
    rrt_step: float = 12.0                 # RRT* extension step [m]
    seed: int = 11                         # RRT* sampling seed (deterministic plans)

    def validate(self) -> None:
        _require(self.clearance > 0 and self.influence > self.clearance, "obstacles: need 0 < clearance < influence")
        _require(self.terrain_clearance >= 0, "obstacles.terrain_clearance must be >= 0")
        _require(self.collision_margin >= 0, "obstacles.collision_margin must be >= 0")
        _require(self.planner in ("astar", "rrtstar"), "obstacles.planner must be astar|rrtstar")
        _require(0.5 <= self.grid_resolution <= 50.0, "obstacles.grid_resolution must be in [0.5, 50] m")
        _require(1000 <= self.max_grid_cells <= 2_000_000, "obstacles.max_grid_cells must be in [1000, 2000000]")
        _require(100 <= self.rrt_iterations <= 100_000 and self.rrt_step > 0, "obstacles RRT* settings invalid")


@dataclass
class HomeConfig:
    """Home base and the layout of per-drone landing pads around it."""

    position_x: float = 0.0
    position_y: float = 0.0
    pad_layout: str = "grid"               # grid | line | circle
    pad_spacing: float = 12.0              # distance between pads [m]

    def validate(self) -> None:
        _require(self.pad_layout in ("grid", "line", "circle"), "home.pad_layout must be grid|line|circle")
        _require(self.pad_spacing > 0.5, "home.pad_spacing must be > 0.5 m")


@dataclass
class DroneConfig:
    # airframe
    mass_kg: float = 1.5
    radius: float = 0.35                   # physical half-size (collision body) [m]
    payload_kg: float = 0.0
    # envelope
    max_horizontal_speed: float = 15.0
    max_climb_rate: float = 5.0
    max_descent_rate: float = 3.0
    max_acceleration: float = 6.0
    max_tilt_deg: float = 35.0
    thrust_to_weight: float = 2.2
    max_yaw_rate_deg: float = 90.0
    # aerodynamics / actuation
    thrust_time_constant: float = 0.15     # first-order response of thrust vector [s]
    drag_coefficient: float = 0.3          # linear drag per unit mass [1/s]
    # controller gains
    position_gain: float = 1.0             # outer loop K_pos [1/s]
    velocity_kp: float = 3.0
    velocity_ki: float = 1.0
    velocity_integral_limit: float = 3.0
    yaw_gain: float = 2.0
    # behaviour
    cruise_speed: float = 10.0
    takeoff_altitude: float = 20.0
    min_altitude: float = 2.0              # lowest commandable altitude AGL (except landing)
    acceptance_radius: float = 1.0
    land_speed: float = 1.5
    rtl_altitude: float = 40.0
    offboard_timeout: float = 0.5          # velocity setpoint timeout -> HOVER [s]
    auto_disarm_delay: float = 5.0         # disarm when idle on the ground [s]
    hard_landing_speed: float = 3.5        # touchdown faster than this = crash [m/s]
    emergency_stop_behavior: str = "land"  # land | kill
    emergency_descent_rate: float = 2.5

    def validate(self) -> None:
        for name in (
            "mass_kg", "radius", "max_horizontal_speed", "max_climb_rate", "max_descent_rate",
            "max_acceleration", "max_yaw_rate_deg", "thrust_time_constant", "cruise_speed",
            "takeoff_altitude", "acceptance_radius", "land_speed", "rtl_altitude",
            "offboard_timeout", "position_gain", "velocity_kp", "emergency_descent_rate",
        ):
            _require(getattr(self, name) > 0, f"drone.{name} must be > 0")
        _require(self.payload_kg >= 0, "drone.payload_kg must be >= 0")
        _require(self.drag_coefficient >= 0, "drone.drag_coefficient must be >= 0")
        _require(5.0 <= self.max_tilt_deg <= 70.0, "drone.max_tilt_deg must be in [5, 70]")
        _require(self.thrust_to_weight > 1.05, "drone.thrust_to_weight must be > 1.05")
        _require(self.emergency_stop_behavior in ("land", "kill"), "drone.emergency_stop_behavior must be land|kill")
        _require(self.cruise_speed <= self.max_horizontal_speed, "drone.cruise_speed exceeds max_horizontal_speed")


@dataclass
class BatteryConfig:
    capacity_wh: float = 100.0
    cells: int = 4
    cell_voltage_full: float = 4.2
    cell_voltage_empty: float = 3.3
    internal_resistance: float = 0.05      # pack resistance [ohm]
    initial_percent_min: float = 85.0
    initial_percent_max: float = 100.0
    avionics_power_w: float = 5.0
    idle_power_w: float = 20.0
    hover_power_w: float = 180.0
    speed_power_coeff: float = 0.9         # W / (m/s)^2 of air-relative speed
    accel_power_coeff: float = 4.0         # W / (kg * m/s^2)
    climb_efficiency: float = 0.7
    drain_multiplier: float = 1.0          # >1 accelerates drain for testing
    warning: float = 30.0
    return_home: float = 20.0
    emergency: float = 10.0

    def validate(self) -> None:
        _require(self.capacity_wh > 0, "battery.capacity_wh must be > 0")
        _require(self.cells >= 1, "battery.cells must be >= 1")
        _require(self.cell_voltage_full > self.cell_voltage_empty > 0, "battery cell voltages invalid")
        _require(0 <= self.initial_percent_min <= self.initial_percent_max <= 100,
                 "battery.initial_percent_min/max must satisfy 0 <= min <= max <= 100")
        _require(0 < self.climb_efficiency <= 1, "battery.climb_efficiency must be in (0, 1]")
        _require(self.drain_multiplier >= 0, "battery.drain_multiplier must be >= 0")
        _require(0 <= self.emergency < self.return_home < self.warning <= 100,
                 "battery thresholds must satisfy 0 <= emergency < return_home < warning <= 100")


FORMATION_SHAPES = ("line", "column", "v", "diamond", "grid", "circle", "wedge", "custom")
AVOIDANCE_METHODS = ("orca", "potential_field")


def min_pairwise_distance(points: list[list[float]]) -> float:
    """Smallest distance between any two points (inf for fewer than two)."""
    best = float("inf")
    for a in range(len(points)):
        for b in range(a + 1, len(points)):
            d = sum((points[a][k] - points[b][k]) ** 2 for k in range(3)) ** 0.5
            best = min(best, d)
    return best


@dataclass
class SwarmConfig:
    separation_distance: float = 5.0       # safety radius -> AVOIDANCE (counted as a separation violation)
    warning_distance: float = 10.0         # warning radius -> WARNING
    collision_distance: float = 1.0        # contact distance -> COLLISION
    min_separation: float = 4.0            # HARD floor enforced by the safety filter [m]
    neighbor_radius: float = 60.0          # neighbourhood for swarm algorithms [m]
    max_neighbors: int = 8
    use_kdtree: bool = True
    # --- collision avoidance (applies to every airborne, controllable drone)
    collision_avoidance: bool = True
    avoidance_method: str = "orca"         # orca (velocity obstacles, primary) | potential_field
    avoidance_radius: float = 12.0         # potential-field influence radius d0 [m]
    avoidance_max_speed: float = 5.0       # largest correction the potential field may add [m/s]
    avoidance_horizon: float = 4.0         # look-ahead for closest-approach prediction [s]
    avoidance_margin: float = 2.0          # extra clearance beyond separation_distance [m]
    avoidance_slowdown: float = 0.7        # 0..1: how much a drone yields (slows) before a predicted conflict
    orca_time_horizon: float = 3.0         # ORCA: velocity obstacles are truncated at this time [s]
    orca_max_neighbors: int = 10           # ORCA: closest neighbours considered per drone
    safety_brake_fraction: float = 0.5     # share of max acceleration the hard safety filter assumes for braking
    # --- formation control
    formation: str = "v"                   # default shape
    formation_spacing: float = 15.0        # distance between neighbouring slots [m]
    formation_reference: str = "virtual"   # virtual (movable reference point) | leader
    formation_speed: float = 6.0           # cruise speed of the formation reference [m/s]
    formation_catchup_speed: float = 8.0   # max speed a member uses to reach its slot [m/s]
    formation_heading: float | None = None  # fixed formation heading [deg compass]; null = face direction of travel
    formation_altitude_layers: int = 1     # stack slots over this many altitude layers
    formation_layer_spacing: float = 6.0   # vertical distance between altitude layers [m]
    formation_transition: str = "synchronized"   # synchronized (all drones arrive together, no crossings) | direct
    v_angle_deg: float = 45.0              # half-angle of V/wedge arms
    custom_formation: list[list[float]] = field(default_factory=list)   # [[forward, left, up], ...] [m]
    # --- leader-follower
    leader_promotion: bool = True          # promote a follower when the leader fails / loses comms (else: virtual ref)

    def validate(self) -> None:
        _require(0 < self.collision_distance < self.separation_distance < self.warning_distance,
                 "swarm distances must satisfy 0 < collision < separation < warning")
        _require(self.collision_distance < self.min_separation <= self.separation_distance,
                 "swarm.min_separation must satisfy collision_distance < min_separation <= separation_distance")
        _require(self.neighbor_radius > 0, "swarm.neighbor_radius must be > 0")
        _require(self.max_neighbors >= 1, "swarm.max_neighbors must be >= 1")
        _require(self.avoidance_method in AVOIDANCE_METHODS,
                 f"swarm.avoidance_method must be one of {', '.join(AVOIDANCE_METHODS)}")
        _require(self.avoidance_radius > self.separation_distance, "swarm.avoidance_radius must exceed separation_distance")
        _require(self.avoidance_max_speed > 0 and self.avoidance_horizon > 0, "swarm avoidance speed/horizon must be > 0")
        _require(self.avoidance_margin >= 0, "swarm.avoidance_margin must be >= 0")
        _require(0.0 <= self.avoidance_slowdown <= 1.0, "swarm.avoidance_slowdown must be in [0, 1]")
        _require(0.5 <= self.orca_time_horizon <= 20.0, "swarm.orca_time_horizon must be in [0.5, 20] s")
        _require(1 <= self.orca_max_neighbors <= 50, "swarm.orca_max_neighbors must be in [1, 50]")
        _require(0.1 <= self.safety_brake_fraction <= 1.0, "swarm.safety_brake_fraction must be in [0.1, 1]")
        _require(self.formation.lower() in FORMATION_SHAPES, f"swarm.formation must be one of {', '.join(FORMATION_SHAPES)}")
        _require(self.formation_spacing > self.separation_distance,
                 "swarm.formation_spacing must exceed separation_distance")
        _require(self.formation_reference in ("virtual", "leader"), "swarm.formation_reference must be virtual|leader")
        _require(self.formation_speed > 0 and self.formation_catchup_speed > 0, "swarm formation speeds must be > 0")
        _require(self.formation_heading is None or -360.0 <= self.formation_heading <= 360.0,
                 "swarm.formation_heading must be null or in [-360, 360] deg")
        _require(1 <= self.formation_altitude_layers <= 10, "swarm.formation_altitude_layers must be in [1, 10]")
        _require(self.formation_layer_spacing >= self.min_separation,
                 "swarm.formation_layer_spacing must be >= min_separation")
        _require(self.formation_transition in ("synchronized", "direct"),
                 "swarm.formation_transition must be synchronized|direct")
        _require(10.0 <= self.v_angle_deg <= 80.0, "swarm.v_angle_deg must be in [10, 80]")
        _require(all(len(p) == 3 for p in self.custom_formation), "swarm.custom_formation entries must be [x, y, z]")
        _require(min_pairwise_distance(self.custom_formation) >= self.separation_distance,
                 "swarm.custom_formation slots must be at least separation_distance apart")


@dataclass
class FlockingConfig:
    """Reynolds flocking: V = W1*separation + W2*alignment + W3*cohesion + W4*goal + W5*obstacle."""

    separation_weight: float = 1.6
    alignment_weight: float = 0.6
    cohesion_weight: float = 0.8
    goal_weight: float = 1.0
    obstacle_weight: float = 2.0           # obstacle term arrives with Phase 3 obstacles
    separation_radius: float = 9.0         # neighbours closer than this push apart [m]
    perception_radius: float = 40.0        # neighbours considered for alignment/cohesion [m]
    cohesion_gain: float = 0.15            # [1/s] converts the centroid offset to a velocity
    max_speed: float = 8.0
    altitude_gain: float = 0.6             # [1/s] holds the flock altitude

    def validate(self) -> None:
        for name in ("separation_weight", "alignment_weight", "cohesion_weight", "goal_weight", "obstacle_weight"):
            _require(getattr(self, name) >= 0, f"flocking.{name} must be >= 0")
        _require(0 < self.separation_radius < self.perception_radius, "flocking radii must satisfy 0 < separation < perception")
        _require(self.max_speed > 0 and self.cohesion_gain >= 0 and self.altitude_gain >= 0, "flocking gains must be >= 0")


@dataclass
class MissionConfig:
    """Mission planning and execution (Stage 2)."""

    directory: str = "data/missions"       # where mission_*.json files are saved (relative to the project)
    default_altitude: float = 30.0         # new waypoints [m above home]
    default_speed: float = 8.0             # new waypoints [m/s]
    loiter_radius: float = 20.0            # LOITER orbit radius when a waypoint does not set one [m]
    arrival_radius: float = 2.0            # a single drone has reached a waypoint within this distance [m]
    formation_arrival_error: float = 3.0   # a formation has arrived when every slot error is below this [m]
    battery_reserve_percent: float = 20.0  # validation keeps this share of capacity untouched
    camera_hfov_deg: float = 70.0          # survey camera horizontal field of view
    survey_overlap: float = 70.0           # default side overlap between survey lines [%]
    survey_finish: str = "RTL"             # action appended to each survey track: RTL | LAND | HOLD
    default_formation: str = "grid"        # formation used when a multi-drone mission starts without one

    def validate(self) -> None:
        _require(bool(self.directory.strip()), "mission.directory must not be empty")
        _require(self.default_altitude > 0 and self.default_speed > 0, "mission default altitude/speed must be > 0")
        _require(1.0 <= self.loiter_radius <= 2000.0, "mission.loiter_radius must be in [1, 2000] m")
        _require(0.5 <= self.arrival_radius <= 50.0, "mission.arrival_radius must be in [0.5, 50] m")
        _require(0.5 <= self.formation_arrival_error <= 50.0, "mission.formation_arrival_error must be in [0.5, 50] m")
        _require(0.0 <= self.battery_reserve_percent <= 90.0, "mission.battery_reserve_percent must be in [0, 90]")
        _require(5.0 <= self.camera_hfov_deg <= 170.0, "mission.camera_hfov_deg must be in [5, 170]")
        _require(0.0 <= self.survey_overlap < 100.0, "mission.survey_overlap must be in [0, 100)")
        _require(self.survey_finish in ("RTL", "LAND", "HOLD"), "mission.survey_finish must be RTL|LAND|HOLD")
        _require(self.default_formation in FORMATION_SHAPES, "mission.default_formation must be a formation shape")

    @property
    def path(self) -> Path:
        p = Path(self.directory)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class GeofenceConfig:
    """Initial geofence (Stage 2). Polygons are [[x, y], ...] in local ENU metres."""

    enabled: bool = False
    action: str = "RTL"                    # breach action: RTL | LAND | HOLD
    lookahead_s: float = 2.0               # predictive breach: also check position + velocity * lookahead
    clear_time: float = 2.0                # a breach episode ends after this long back inside [s]
    max_altitude: float | None = None      # fence ceiling [m above home]; null = world ceiling only
    inclusion: list[list[float]] = field(default_factory=list)          # empty = no inclusion polygon
    exclusions: list[list[list[float]]] = field(default_factory=list)   # no-fly polygons

    def validate(self) -> None:
        _require(self.action in ("RTL", "LAND", "HOLD"), "geofence.action must be RTL|LAND|HOLD")
        _require(0.0 <= self.lookahead_s <= 30.0, "geofence.lookahead_s must be in [0, 30] s")
        _require(0.0 <= self.clear_time <= 60.0, "geofence.clear_time must be in [0, 60] s")
        _require(self.max_altitude is None or self.max_altitude > 0, "geofence.max_altitude must be > 0 or null")
        _require(not self.inclusion or len(self.inclusion) >= 3, "geofence.inclusion needs at least 3 points")
        _require(all(len(p) == 2 for p in self.inclusion), "geofence.inclusion points must be [x, y]")
        for k, zone in enumerate(self.exclusions):
            _require(len(zone) >= 3 and all(len(p) == 2 for p in zone),
                     f"geofence.exclusions[{k}] must be at least 3 [x, y] points")


@dataclass
class AlertsConfig:
    """Operator alerts (Stage 3)."""

    warning_ttl_s: float = 60.0            # an unacknowledged WARNING disappears after this long
    info_ttl_s: float = 20.0               # INFO alerts disappear after this long
    acknowledged_ttl_s: float = 30.0       # acknowledged alerts stay visible (greyed) this long
    dedup_window_s: float = 10.0           # a repeat within this window updates the existing alert
    max_active: int = 50                   # alerts kept in the panel
    max_alerts: int = 1000                 # alert history kept for reports
    critical_repeat_s: float = 8.0         # GCS: repeat the beep this often while a CRITICAL is unacknowledged

    def validate(self) -> None:
        _require(self.warning_ttl_s > 0 and self.info_ttl_s > 0 and self.acknowledged_ttl_s >= 0,
                 "alerts TTLs must be > 0")
        _require(0 <= self.dedup_window_s <= 600, "alerts.dedup_window_s must be in [0, 600] s")
        _require(5 <= self.max_active <= 500, "alerts.max_active must be in [5, 500]")
        _require(self.max_alerts >= self.max_active, "alerts.max_alerts must be >= max_active")
        _require(self.critical_repeat_s >= 1.0, "alerts.critical_repeat_s must be >= 1 s")


@dataclass
class CommunicationConfig:
    """GCS <-> drone link model (Stage 4)."""

    enabled: bool = False
    range: float = 500.0                   # hard link range from the GCS [m]
    latency_ms: float = 50.0               # mean one-way latency
    jitter_ms: float = 15.0                # latency standard deviation
    packet_loss: float = 0.02              # base loss probability per packet
    edge_fraction: float = 0.8             # loss rises from the base value at this share of range to 100 % at range
    heartbeat_hz: float = 10.0             # heartbeat / telemetry packet rate each way
    timeout_s: float = 1.5                 # no heartbeat for this long -> COMM LOST
    degraded_quality: float = 60.0         # link quality [%] below which the link is DEGRADED
    failsafe_action: str = "RTL"           # after failsafe_timeout_s in LOST: RTL | LAND | HOLD
    failsafe_timeout_s: float = 5.0
    command_retries: int = 3               # re-sends of an unacknowledged command
    retry_interval_s: float = 0.25
    gcs_height: float = 2.0                # antenna height above the home pad [m]

    def validate(self) -> None:
        _require(self.range > 0, "communication.range must be > 0")
        _require(self.latency_ms >= 0 and self.jitter_ms >= 0, "communication latency/jitter must be >= 0")
        _require(0.0 <= self.packet_loss <= 1.0, "communication.packet_loss must be in [0, 1]")
        _require(0.0 < self.edge_fraction <= 1.0, "communication.edge_fraction must be in (0, 1]")
        _require(0.5 <= self.heartbeat_hz <= 50.0, "communication.heartbeat_hz must be in [0.5, 50]")
        _require(self.timeout_s * self.heartbeat_hz >= 2, "communication.timeout_s must cover at least 2 heartbeats")
        _require(0 <= self.degraded_quality <= 100, "communication.degraded_quality must be in [0, 100]")
        _require(self.failsafe_action in ("RTL", "LAND", "HOLD"), "communication.failsafe_action must be RTL|LAND|HOLD")
        _require(self.failsafe_timeout_s >= 0, "communication.failsafe_timeout_s must be >= 0")
        _require(0 <= self.command_retries <= 20 and self.retry_interval_s > 0, "communication retry settings invalid")


@dataclass
class SensorsConfig:
    """Sensor noise and state estimation (Stage 4)."""

    enabled: bool = False
    control_source: str = "estimate"       # the autopilot flies on: estimate (Kalman filter) | truth
    gps_rate_hz: float = 5.0
    gps_noise_h: float = 1.0               # horizontal white noise, 1 sigma [m]
    gps_noise_v: float = 1.8               # vertical white noise, 1 sigma [m]
    gps_drift: float = 1.5                 # slowly wandering GPS bias, 1 sigma [m] (Gauss-Markov)
    gps_drift_time_s: float = 120.0        # correlation time of the GPS bias
    baro_rate_hz: float = 20.0
    baro_noise: float = 0.4                # [m]
    baro_drift: float = 0.5                # slowly wandering barometer bias, 1 sigma [m]
    compass_noise_deg: float = 2.0
    compass_bias_deg: float = 1.0          # per-drone constant compass bias, 1 sigma
    accel_noise: float = 0.15              # accelerometer noise [m/s^2] (Kalman prediction input)
    gyro_noise_deg: float = 0.5            # yaw-rate noise [deg/s]
    process_noise: float = 0.6             # Kalman filter acceleration noise density [m/s^2]

    def validate(self) -> None:
        _require(self.control_source in ("estimate", "truth"), "sensors.control_source must be estimate|truth")
        _require(0.2 <= self.gps_rate_hz <= 50 and 1 <= self.baro_rate_hz <= 200, "sensors rates out of range")
        for name in ("gps_noise_h", "gps_noise_v", "gps_drift", "baro_noise", "baro_drift", "compass_noise_deg",
                     "compass_bias_deg", "accel_noise", "gyro_noise_deg"):
            _require(getattr(self, name) >= 0, f"sensors.{name} must be >= 0")
        _require(self.gps_drift_time_s > 0 and self.process_noise > 0, "sensors drift time / process noise must be > 0")


@dataclass
class FailuresConfig:
    """Failure injection (instructor mode, Stage 4)."""

    enabled: bool = True                   # allow inject_failure commands
    motor_partial_thrust: float = 0.45     # share of thrust left after a partial motor failure
    default_gust_speed: float = 12.0       # [m/s]
    default_gust_duration_s: float = 6.0
    default_battery_drop: float = 30.0     # [% of capacity]

    def validate(self) -> None:
        _require(0.05 <= self.motor_partial_thrust < 1.0, "failures.motor_partial_thrust must be in [0.05, 1)")
        _require(self.default_gust_speed > 0 and self.default_gust_duration_s > 0, "failures gust defaults must be > 0")
        _require(0 < self.default_battery_drop <= 100, "failures.default_battery_drop must be in (0, 100]")


@dataclass
class WindConfig:
    enabled: bool = True
    speed: float = 5.0                     # mean wind at 10 m reference height [m/s]
    direction: float = 90.0                # meteorological: direction wind comes FROM [deg]
    gust_strength: float = 1.5             # gust standard deviation [m/s]
    gust_frequency: float = 0.2            # inverse gust correlation time [Hz]
    vertical_gust_ratio: float = 0.3
    shear: bool = True                     # power-law altitude profile

    def validate(self) -> None:
        _require(self.speed >= 0, "wind.speed must be >= 0")
        _require(self.gust_strength >= 0, "wind.gust_strength must be >= 0")
        _require(self.gust_frequency > 0, "wind.gust_frequency must be > 0")
        _require(self.vertical_gust_ratio >= 0, "wind.vertical_gust_ratio must be >= 0")


HARDWARE_LINKS = ("mavlink", "mavsdk")


@dataclass
class HardwareConfig:
    """Real / SITL vehicles driven by the same GCS as the simulated swarm (Stage 5).

    ``vehicles``: ``[{type: mavlink|mavsdk, url: <connection string>, name?: D-name, system_id?: int}]``.
    """

    vehicles: list[dict[str, Any]] = field(default_factory=list)
    heartbeat_timeout_s: float = 3.0       # no vehicle heartbeat for this long -> COMM LOST
    setpoint_rate_hz: float = 10.0         # velocity setpoints in FORMATION / OFFBOARD

    def validate(self) -> None:
        _require(0.5 <= self.heartbeat_timeout_s <= 60, "hardware.heartbeat_timeout_s must be in [0.5, 60]")
        _require(1.0 <= self.setpoint_rate_hz <= 50.0, "hardware.setpoint_rate_hz must be in [1, 50]")
        urls = []
        for k, v in enumerate(self.vehicles):
            _require(isinstance(v, dict), f"hardware.vehicles[{k}] must be a mapping")
            unknown = set(v) - {"type", "url", "name", "system_id"}
            _require(not unknown, f"hardware.vehicles[{k}]: unknown key(s) {', '.join(sorted(unknown))}")
            _require(v.get("type") in HARDWARE_LINKS, f"hardware.vehicles[{k}].type must be mavlink|mavsdk")
            _require(isinstance(v.get("url"), str) and v["url"].strip() != "", f"hardware.vehicles[{k}].url is required")
            _require(v.get("system_id") is None or (isinstance(v["system_id"], int) and 1 <= v["system_id"] <= 255),
                     f"hardware.vehicles[{k}].system_id must be in [1, 255]")
            urls.append(v["url"])
        _require(len(urls) == len(set(urls)), "hardware.vehicles: every vehicle needs its own url")


ROLES = ("operator", "observer")


@dataclass
class SecurityConfig:
    """Role-based access (Stage 5): Operator (full control) and Observer (read-only), plus the audit log.

    ``users``: ``[{username, role: operator|observer, password_hash: pbkdf2_sha256$<iter>$<salt>$<hash>}]``;
    make a hash with ``python -m backend.security hash-password``. With ``enabled: false`` every client is
    an operator called ``local`` (the audit log is still written).
    """

    enabled: bool = False                  # configs/simulation.yaml turns it on (needs users)
    token_ttl_s: float = 43200.0          # session length (tokens are HMAC-signed, in-memory secret)
    audit_log: str = "audit.log"           # JSON Lines, relative to logging.directory
    max_login_failures: int = 5            # per username, then a lockout
    lockout_s: float = 60.0
    users: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        _require(60 <= self.token_ttl_s <= 7 * 86400, "security.token_ttl_s must be in [60, 604800]")
        _require(self.audit_log.strip() != "", "security.audit_log must not be empty")
        _require(1 <= self.max_login_failures <= 100, "security.max_login_failures must be in [1, 100]")
        _require(0 <= self.lockout_s <= 3600, "security.lockout_s must be in [0, 3600]")
        names = []
        for k, u in enumerate(self.users):
            _require(isinstance(u, dict), f"security.users[{k}] must be a mapping")
            unknown = set(u) - {"username", "role", "password_hash"}
            _require(not unknown, f"security.users[{k}]: unknown key(s) {', '.join(sorted(unknown))}")
            name = u.get("username")
            _require(isinstance(name, str) and 1 <= len(name) <= 64 and name.strip() == name,
                     f"security.users[{k}].username must be a 1-64 character string")
            _require(u.get("role") in ROLES, f"security.users[{k}].role must be operator|observer")
            h = u.get("password_hash")
            _require(isinstance(h, str) and h.startswith("pbkdf2_sha256$") and h.count("$") == 3,
                     f"security.users[{k}].password_hash must be pbkdf2_sha256$<iterations>$<salt>$<hash>")
            names.append(name)
        _require(len(names) == len(set(names)), "security.users: usernames must be unique")
        _require(not self.enabled or any(u["role"] == "operator" for u in self.users),
                 "security.enabled needs at least one operator in security.users")


@dataclass
class ReplayConfig:
    """Mission replay of recorded runs (``logs/<run_id>``)."""

    chunk_s: float = 20.0                  # frames per request window
    cache_runs: int = 2                    # parsed runs kept in memory
    max_events: int = 2000                 # event markers sent to the scrubber

    def validate(self) -> None:
        _require(2.0 <= self.chunk_s <= 300.0, "replay.chunk_s must be in [2, 300]")
        _require(1 <= self.cache_runs <= 10, "replay.cache_runs must be in [1, 10]")
        _require(10 <= self.max_events <= 100000, "replay.max_events must be in [10, 100000]")


@dataclass
class ReportConfig:
    """One-click post-mission report (HTML / PDF)."""

    title: str = "Post-mission report"
    logo: str = "frontend/assets/gandiv-logo.png"   # relative to the project root; a wordmark if missing
    max_track_points: int = 600            # per drone in the flight-path plot
    max_timeline_events: int = 200

    def validate(self) -> None:
        _require(self.title.strip() != "", "report.title must not be empty")
        _require(50 <= self.max_track_points <= 10000, "report.max_track_points must be in [50, 10000]")
        _require(10 <= self.max_timeline_events <= 5000, "report.max_timeline_events must be in [10, 5000]")

    @property
    def logo_path(self) -> Path:
        p = Path(self.logo)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class TelemetryConfig:
    rate_hz: float = 20.0                  # WebSocket publish rate while running
    paused_rate_hz: float = 2.0
    event_history: int = 500

    def validate(self) -> None:
        _require(0.5 <= self.rate_hz <= 120, "telemetry.rate_hz must be in [0.5, 120]")
        _require(0.1 <= self.paused_rate_hz <= 30, "telemetry.paused_rate_hz must be in [0.1, 30]")
        _require(self.event_history >= 10, "telemetry.event_history must be >= 10")


@dataclass
class LoggingConfig:
    level: str = "INFO"
    directory: str = "logs"
    console: bool = True
    record_telemetry: bool = True
    telemetry_record_rate_hz: float = 5.0

    def validate(self) -> None:
        _require(self.level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
                 "logging.level must be a standard logging level")
        _require(0.1 <= self.telemetry_record_rate_hz <= 100, "logging.telemetry_record_rate_hz must be in [0.1, 100]")

    @property
    def path(self) -> Path:
        p = Path(self.directory)
        return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000

    def validate(self) -> None:
        _require(0 < self.port < 65536, "server.port must be a valid TCP port")


@dataclass
class SimConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    origin: OriginConfig = field(default_factory=OriginConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    obstacles: ObstaclesConfig = field(default_factory=ObstaclesConfig)
    home: HomeConfig = field(default_factory=HomeConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    flocking: FlockingConfig = field(default_factory=FlockingConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    geofence: GeofenceConfig = field(default_factory=GeofenceConfig)
    alerts: AlertsConfig = field(default_factory=AlertsConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    failures: FailuresConfig = field(default_factory=FailuresConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    extensions: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "SimConfig":
        for f in fields(self):
            section = getattr(self, f.name)
            if is_dataclass(section):
                section.validate()
        half_x, half_y = self.environment.size_x / 2, self.environment.size_y / 2
        _require(abs(self.home.position_x) < half_x and abs(self.home.position_y) < half_y,
                 "home position must lie inside the environment bounds")
        _require(self.drone.rtl_altitude < self.environment.max_altitude, "drone.rtl_altitude exceeds environment.max_altitude")
        _require(self.drone.takeoff_altitude < self.environment.max_altitude,
                 "drone.takeoff_altitude exceeds environment.max_altitude")
        _require(self.drone.min_altitude < self.drone.takeoff_altitude, "drone.min_altitude must be below takeoff_altitude")
        _require(self.obstacles.terrain_clearance <= self.drone.min_altitude,
                 "obstacles.terrain_clearance must not exceed drone.min_altitude (it would fight commanded altitudes)")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def dt(self) -> float:
        return 1.0 / self.simulation.simulation_rate


# --------------------------------------------------------------------------- loading


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Convert ``value`` to type ``tp`` or raise :class:`ConfigError`."""
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: value must not be null")
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: invalid value {value!r} ({'; '.join(errors)})")
    if tp is Any:
        return value
    if is_dataclass(tp):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return _build_dataclass(tp, value, path)
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        item_type = (get_args(tp) or (Any,))[0]
        return [_coerce(v, item_type, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin is dict or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping")
        return dict(value)
    if tp is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected true/false, got {value!r}")
    if tp is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected an integer, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{path}: expected an integer, got {value!r}")
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if isinstance(value, str):
            return value
        raise ConfigError(f"{path}: expected a string, got {value!r}")
    raise ConfigError(f"{path}: unsupported configuration type {tp!r}")


def _build_dataclass(cls: type, data: Mapping[str, Any], path: str) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown configuration key(s) in '{path}': {', '.join(unknown)}")
    kwargs = {name: _coerce(value, hints[name], f"{path}.{name}") for name, value in data.items()}
    return cls(**kwargs)


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if len(parts) < 2 or not all(parts):
        raise ConfigError(f"override '{dotted}' must look like section.key")
    node = tree
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override '{dotted}': '{part}' is not a section")
        node = child
    node[parts[-1]] = value


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Parse ``["simulation.drone_count=25", ...]``; values are YAML-typed."""
    result: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ConfigError(f"override '{item}' must have the form section.key=value")
        key, raw = item.split("=", 1)
        result[key.strip()] = yaml.safe_load(raw) if raw.strip() else None
    return result


def config_from_dict(data: Mapping[str, Any] | None) -> SimConfig:
    """Build and validate a :class:`SimConfig` from a (partial) nested mapping."""
    data = dict(data or {})
    known = {f.name for f in fields(SimConfig)} - {"extensions"}
    extensions = {k: data.pop(k) for k in list(data) if k not in known}
    if extensions:
        log.warning("Configuration sections not used by this version: %s", ", ".join(sorted(extensions)))
    cfg: SimConfig = _build_dataclass(SimConfig, data, "config")
    cfg.extensions = extensions
    return cfg.validate()


def load_config(
    path: str | os.PathLike[str] | None = None,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> SimConfig:
    """Load configuration.

    Resolution order: explicit ``path`` > ``$SWARM_CONFIG`` > ``configs/simulation.yaml``
    (skipped silently if absent). ``overrides`` may be a mapping of dotted keys to
    values or a list of ``"dotted.key=value"`` strings.
    """
    resolved = Path(path) if path else Path(os.environ[CONFIG_ENV_VAR]) if os.environ.get(CONFIG_ENV_VAR) else None
    raw: dict[str, Any] = {}
    if resolved is not None and not resolved.exists():
        raise ConfigError(f"configuration file not found: {resolved}")
    source = resolved or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
    if source is not None:
        try:
            loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {source}: {exc}") from exc
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{source}: top level must be a mapping")
        raw = loaded or {}
    if overrides:
        items = overrides.items() if isinstance(overrides, Mapping) else parse_overrides(overrides).items()
        for key, value in items:
            _set_dotted(raw, key, value)
    return config_from_dict(raw)
