"""Obstacle and terrain avoidance as a velocity-pipeline stage (priority 30).

Pipeline order: formation / flocking (20) -> **obstacles (30)** -> drone-drone collision avoidance (40).

For every manoeuvrable airborne drone, the nearest obstacle within ``obstacles.influence`` (found with
the obstacle KD-tree) gives a signed distance ``d`` and an outward normal ``n``. Two effects:

1. **Braking barrier** (the same discrete control-barrier function as the drone-drone safety filter)::

       h = d - clearance - c_act * tau_lag           c = -v . n  (speed towards the surface)
       c <= sqrt(2 * a_brake * h)    (h > 0)          c <= h / T_push   (h <= 0: move away)

   The excess approach speed is removed along ``n``, so a drone never enters the clearance shell.
2. **Slide**: inside the influence radius the velocity component along the surface is kept and a small
   tangential component is added in the direction the drone was already heading, so drones flow around
   an obstacle instead of stopping in front of it. Real detours come from the path planner.

**Terrain**: the same barrier on height above ground, using the highest terrain under the drone and
under its position one second ahead: descent (or flying into rising ground) is limited so the drone
keeps ``terrain_clearance``. Landing, take-off and emergency descents are exempt.
"""

from __future__ import annotations

import numpy as np

from simulation.config import SimConfig
from simulation.swarm import SwarmContext, VelocityStage
from simulation.types import FlightMode

PUSH_TIME = 1.0
EXEMPT = frozenset({FlightMode.LAND, FlightMode.TAKEOFF, FlightMode.EMERGENCY})
SLIDE_GAIN = 0.35


class ObstacleAvoidance(VelocityStage):
    name = "obstacle_avoidance"
    priority = 30

    def __init__(self, config: SimConfig) -> None:
        oc, dc = config.obstacles, config.drone
        self.enabled = oc.avoidance
        self.clearance = oc.clearance + dc.radius
        self.influence = oc.influence
        self.terrain_clearance = oc.terrain_clearance
        max_h_accel = min(dc.max_acceleration, 9.80665 * np.tan(np.radians(dc.max_tilt_deg)))
        self.brake = config.swarm.safety_brake_fraction * max_h_accel
        self.brake_z = 0.5 * dc.max_acceleration
        self.lag = dc.thrust_time_constant + 1.0 / max(dc.velocity_kp, 1e-6)
        self.active_drones = 0
        self.min_distance: float | None = None

    def apply(self, ctx: SwarmContext, velocities: np.ndarray) -> np.ndarray:
        env = ctx.environment
        self.active_drones = 0
        self.min_distance = None
        n = len(ctx.drones)
        active = ctx.controllable & ctx.airborne & np.fromiter(
            (d.flight_mode not in EXEMPT and not (d.flight_mode == FlightMode.RTL and d.rtl_landing) for d in ctx.drones),
            bool, n)
        if not np.any(active):
            return velocities
        out = velocities.copy()
        idx = np.flatnonzero(active)
        p = ctx.positions[idx]
        v_act = ctx.velocities[idx]

        # --- obstacles
        if len(env.obstacles):
            dist, normal, which = env.obstacles.nearest(p, self.influence)
            near = which >= 0
            if np.any(near):
                self.min_distance = float(dist[near].min())
                k, d, nrm = idx[near], dist[near], normal[near]
                closing_now = np.maximum(-np.einsum("ij,ij->i", v_act[near], nrm), 0.0)
                h = d - self.clearance - closing_now * (self.lag + ctx.dt)
                allowed = np.where(h > 0, np.sqrt(2.0 * self.brake * np.maximum(h, 0.0)), h / PUSH_TIME)
                v = out[k]
                closing = -np.einsum("ij,ij->i", v, nrm)
                excess = np.maximum(closing - allowed, 0.0)
                v = v + excess[:, None] * nrm
                # slide: add a tangential push (in the direction of the current horizontal velocity) near the surface
                tang = v - np.einsum("ij,ij->i", v, nrm)[:, None] * nrm
                tn = np.linalg.norm(tang[:, :2], axis=1)
                w = np.clip((self.influence - d) / (self.influence - self.clearance), 0.0, 1.0) * (excess > 0)
                push = np.where(tn[:, None] > 0.2, tang / np.maximum(tn, 1e-9)[:, None], 0.0) * (SLIDE_GAIN * excess * w)[:, None]
                push[:, 2] = 0.0
                out[k] = v + push
                self.active_drones = int(np.count_nonzero(excess > 0))

        # --- terrain (and the flat ground when there is no terrain model)
        look = p + v_act * 1.0
        ground_now = env.ground_heights(p[:, 0], p[:, 1])
        ground_ahead = env.ground_heights(look[:, 0], look[:, 1])
        ground = np.maximum(ground_now, ground_ahead)
        h = p[:, 2] - ground - self.terrain_clearance
        allowed_down = np.where(h > 0, np.sqrt(2.0 * self.brake_z * np.maximum(h, 0.0)), h / PUSH_TIME)
        vz = out[idx, 2]
        # effective sink rate towards the ground includes flying into terrain rising ahead
        rise = np.maximum(ground_ahead - ground_now, 0.0)                 # m over the next second
        sink = -vz + rise
        fix = np.maximum(sink - allowed_down, 0.0)
        out[idx, 2] = vz + fix
        return out
