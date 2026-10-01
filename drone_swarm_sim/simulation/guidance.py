"""Outer-loop guidance laws: position setpoint -> velocity setpoint.

These are pure functions so swarm algorithms (formation keeping, target
tracking, ...) can reuse exactly the same approach behaviour as the autopilot.
"""

from __future__ import annotations

import math

import numpy as np

from .types import Vector3


def approach_velocity(
    position: Vector3,
    target: Vector3,
    *,
    cruise_speed: float,
    max_climb: float,
    max_descent: float,
    decel: float,
    gain: float,
) -> Vector3:
    """Velocity setpoint that flies a straight line to ``target`` and stops on it.

    ``speed = min(v_cruise, K * d, sqrt(2 * a_brake * d))``

    * ``K * d``: linear convergence near the target (critically damped with the
      inner loop for moderate K).
    * ``sqrt(2 a d)``: the minimum-time braking curve, so a drone decelerating
      at ``a_brake`` comes to rest exactly at the target.

    The vector is then scaled *uniformly* to respect the climb/descent limits, so
    the flight path stays a straight line instead of bending when one axis saturates.
    """
    delta = np.asarray(target, dtype=np.float64) - position
    dist = math.sqrt(delta @ delta)
    if dist < 1e-6:
        return np.zeros(3)
    speed = min(cruise_speed, gain * dist, math.sqrt(2.0 * decel * dist))
    v = delta * (speed / dist)
    if v[2] > max_climb:
        v *= max_climb / v[2]
    elif v[2] < -max_descent:
        v *= -max_descent / v[2]
    return v


def hold_xy_velocity(position: Vector3, hold: Vector3, gain: float, max_speed: float) -> Vector3:
    """Horizontal P-controller used while landing or climbing in place."""
    v = np.zeros(3)
    v[:2] = gain * (np.asarray(hold[:2]) - position[:2])
    h = math.hypot(v[0], v[1])
    if h > max_speed:
        v[:2] *= max_speed / h
    return v
