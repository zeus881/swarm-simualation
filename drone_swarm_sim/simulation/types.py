"""Enumerations and small value types shared by every layer of the platform.

All enums are ``StrEnum`` so they serialise to JSON as their name without any
custom encoder, and compare equal to plain strings coming from the UI.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

Vector3 = np.ndarray
"""A float64 array of shape (3,) in the local ENU frame (x=East, y=North, z=Up)."""


class FlightMode(StrEnum):
    """Autopilot flight mode (mirrors the PX4 / ArduPilot mode concept)."""

    DISARMED = "DISARMED"      # on the ground, motors off
    ARMED = "ARMED"            # on the ground, motors armed at idle
    TAKEOFF = "TAKEOFF"        # climbing vertically to takeoff altitude
    HOVER = "HOVER"            # position hold
    GOTO = "GOTO"              # flying to a position setpoint
    OFFBOARD = "OFFBOARD"      # external velocity setpoints (swarm algorithms)
    FORMATION = "FORMATION"    # formation keeping (Phase 2)
    LAND = "LAND"              # descending to land in place
    RTL = "RTL"                # return to launch/home and land
    EMERGENCY = "EMERGENCY"    # emergency descent or motor kill


AIRBORNE_MODES: frozenset[FlightMode] = frozenset(
    {
        FlightMode.TAKEOFF,
        FlightMode.HOVER,
        FlightMode.GOTO,
        FlightMode.OFFBOARD,
        FlightMode.FORMATION,
        FlightMode.LAND,
        FlightMode.RTL,
        FlightMode.EMERGENCY,
    }
)


class HealthStatus(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    FAILED = "FAILED"


class CommStatus(StrEnum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


class BatteryState(StrEnum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    RETURN_HOME = "RETURN_HOME"
    EMERGENCY = "EMERGENCY"
    DEPLETED = "DEPLETED"


class CollisionState(StrEnum):
    CLEAR = "CLEAR"
    WARNING = "WARNING"        # another drone inside the warning radius
    AVOIDANCE = "AVOIDANCE"    # another drone inside the safety radius
    COLLISION = "COLLISION"    # physical contact distance


COLLISION_SEVERITY: dict[CollisionState, int] = {
    CollisionState.CLEAR: 0,
    CollisionState.WARNING: 1,
    CollisionState.AVOIDANCE: 2,
    CollisionState.COLLISION: 3,
}


def vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vector3:
    """Create a float64 3-vector."""
    return np.array((x, y, z), dtype=np.float64)
