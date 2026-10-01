"""Vehicle abstraction shared by simulated and real/SITL drones.

Everything above the vehicle layer (operator commands, missions, swarm
algorithms) talks to drones exclusively through :class:`DroneInterface`.
Phase 6 adds ``PX4Drone`` (MAVSDK) and ``MAVLinkDrone`` (pymavlink)
implementations; ``SimulatedDrone`` is :class:`simulation.drone.Drone`.

Conventions: positions/velocities are local ENU metres (x=East, y=North, z=Up);
headings are compass degrees (0 = North, clockwise).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from .types import BatteryState, CollisionState, CommStatus, FlightMode, HealthStatus


@dataclass(slots=True)
class CommandResult:
    success: bool
    message: str = ""
    details: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, message: str = "ok", **data: Any) -> "CommandResult":
        return cls(True, message, data=data)

    @classmethod
    def fail(cls, message: str) -> "CommandResult":
        return cls(False, message)

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "message": self.message, "details": self.details, "data": self.data}


def _xyz(v: Sequence[float], digits: int = 3) -> dict[str, float]:
    return {"x": round(float(v[0]), digits), "y": round(float(v[1]), digits), "z": round(float(v[2]), digits)}


@dataclass(slots=True)
class DroneTelemetry:
    """Complete telemetry record for one vehicle (the wire schema of a drone)."""

    drone_id: int
    name: str
    position: tuple[float, float, float]
    gps: tuple[float, float, float]            # lat [deg], lon [deg], alt AMSL [m]
    altitude_agl: float
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    heading: float                             # compass deg
    roll: float                                # deg
    pitch: float                               # deg
    yaw_rate: float                            # deg/s
    battery: float                             # percent
    battery_state: BatteryState
    battery_voltage: float
    power_w: float
    mode: FlightMode
    armed: bool
    airborne: bool
    health: HealthStatus
    communication: CommStatus
    task: str
    target: tuple[float, float, float] | None
    home: tuple[float, float, float]
    neighbors: list[int]
    collision_state: CollisionState
    nearest_distance: float | None
    flight_time: float
    distance_travelled: float
    link_quality: float = 100.0                # 0..100 % (GCS link)
    gps_fix: str = "3D"                        # NONE | 2D | 3D
    satellites: int = 14
    hdop: float = 0.7
    time_left_s: float | None = None           # predicted flight time to the emergency threshold
    est_position: tuple[float, float, float] | None = None   # navigation-filter estimate (sensor model on)
    est_heading: float | None = None           # compass deg
    pos_sigma: float | None = None             # 1-sigma horizontal estimate uncertainty [m]
    failures: list[str] = field(default_factory=list)       # injected failures in force
    source: str = "sim"                        # sim | mavlink | mavsdk (Stage 5 hardware adapters)

    @property
    def speed(self) -> float:
        vx, vy, vz = self.velocity
        return (vx * vx + vy * vy + vz * vz) ** 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "drone_id": self.drone_id,
            "name": self.name,
            "position": _xyz(self.position),
            "gps": {"lat": round(self.gps[0], 7), "lon": round(self.gps[1], 7), "alt": round(self.gps[2], 2)},
            "altitude_agl": round(self.altitude_agl, 2),
            "velocity": _xyz(self.velocity),
            "acceleration": _xyz(self.acceleration),
            "speed": round(self.speed, 2),
            "heading": round(self.heading, 1),
            "roll": round(self.roll, 1),
            "pitch": round(self.pitch, 1),
            "yaw_rate": round(self.yaw_rate, 1),
            "battery": round(self.battery, 1),
            "battery_state": str(self.battery_state),
            "battery_voltage": round(self.battery_voltage, 2),
            "power_w": round(self.power_w, 1),
            "mode": str(self.mode),
            "armed": self.armed,
            "airborne": self.airborne,
            "health": str(self.health),
            "communication": str(self.communication),
            "task": self.task,
            "target": _xyz(self.target, 2) if self.target is not None else None,
            "home": _xyz(self.home, 2),
            "neighbors": list(self.neighbors),
            "collision_state": str(self.collision_state),
            "nearest_distance": round(self.nearest_distance, 2) if self.nearest_distance is not None else None,
            "flight_time": round(self.flight_time, 1),
            "distance_travelled": round(self.distance_travelled, 1),
            "link_quality": round(self.link_quality, 0),
            "gps_fix": self.gps_fix,
            "satellites": self.satellites,
            "hdop": round(self.hdop, 2),
            "time_left_s": round(self.time_left_s, 0) if self.time_left_s is not None else None,
            "est_position": _xyz(self.est_position, 2) if self.est_position is not None else None,
            "est_heading": round(self.est_heading, 1) if self.est_heading is not None else None,
            "pos_error": (round(sum((a - b) ** 2 for a, b in zip(self.est_position, self.position)) ** 0.5, 2)
                          if self.est_position is not None else None),
            "pos_sigma": round(self.pos_sigma, 2) if self.pos_sigma is not None else None,
            "failures": list(self.failures),
            "source": self.source,
        }


class DroneInterface(ABC):
    """Command/telemetry contract for any vehicle (simulated, SITL or real)."""

    @property
    @abstractmethod
    def drone_id(self) -> int: ...

    @abstractmethod
    def arm(self) -> CommandResult: ...

    @abstractmethod
    def disarm(self) -> CommandResult: ...

    @abstractmethod
    def takeoff(self, altitude: float | None = None) -> CommandResult: ...

    @abstractmethod
    def land(self) -> CommandResult: ...

    @abstractmethod
    def hover(self) -> CommandResult: ...

    @abstractmethod
    def goto(self, position: Sequence[float], speed: float | None = None,
             heading: float | None = None) -> CommandResult: ...

    @abstractmethod
    def set_velocity(self, velocity: Sequence[float], heading: float | None = None) -> CommandResult: ...

    @abstractmethod
    def set_heading(self, heading: float) -> CommandResult: ...

    @abstractmethod
    def return_to_home(self) -> CommandResult: ...

    @abstractmethod
    def emergency_stop(self) -> CommandResult: ...

    @abstractmethod
    def get_telemetry(self) -> DroneTelemetry: ...

    def change_altitude(self, altitude: float) -> CommandResult:
        """Climb/descend to ``altitude`` [m AGL] keeping the horizontal target."""
        t = self.get_telemetry()
        x, y = (t.target[0], t.target[1]) if t.target is not None and t.mode == FlightMode.GOTO else t.position[:2]
        return self.goto((x, y, t.position[2] - t.altitude_agl + altitude))
