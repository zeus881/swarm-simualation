"""Reynolds flocking (boids) as a velocity-pipeline stage.

For each member i with neighbours N_i (other members within the perception radius)::

    S_i = sum_{j in N_i, d_ij < r_sep} (1 - d_ij / r_sep) * (p_i - p_j) / d_ij * v_max    separation
    A_i = mean_{j in N_i} v_j                                                           alignment
    C_i = k_coh * (mean_{j in N_i} p_j - p_i)                                           cohesion
    G_i = approach(p_i -> goal)       (zero when no goal is set)                        goal seeking
    O_i = 0                           (obstacle term: Phase 3 obstacle field)            obstacle avoidance

    V_i = W1*S_i + W2*A_i + W3*C_i + W4*G_i + W5*O_i,   then altitude hold and speed limit

Every term is expressed in m/s, so the weights are dimensionless and directly comparable.
Pairwise terms use a vectorised all-pairs computation over the members (O(m^2) memory; fine
for the <= 200 drones this platform targets).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from simulation.config import SimConfig
from simulation.dynamics import clip_velocity
from simulation.guidance import approach_velocity
from simulation.swarm import SwarmContext, VelocityStage
from simulation.types import FlightMode, Vector3


class FlockingController(VelocityStage):
    name = "flocking"
    priority = 20

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.cfg = dataclasses.replace(config.flocking)   # runtime weight changes never touch the loaded config
        self.active = False
        self.goal: Vector3 | None = None
        self.altitude: float | None = None
        self.members: list[int] = []

    def start(self, altitude: float | None, goal: Vector3 | None = None) -> None:
        self.active = True
        self.altitude = altitude
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64).copy()

    def stop(self) -> None:
        self.active = False
        self.goal = None
        self.members = []

    def set_goal(self, goal: Vector3 | None) -> None:
        self.goal = None if goal is None else np.asarray(goal, dtype=np.float64).copy()
        if self.goal is not None:
            self.altitude = float(self.goal[2])

    def set_weights(self, **weights: float) -> None:
        for key, value in weights.items():
            setattr(self.cfg, f"{key}_weight", float(value))

    def weights(self) -> dict[str, float]:
        """Current (runtime) rule weights."""
        c = self.cfg
        return {"separation": c.separation_weight, "alignment": c.alignment_weight,
                "cohesion": c.cohesion_weight, "goal": c.goal_weight, "obstacle": c.obstacle_weight}

    @staticmethod
    def _is_member(d) -> bool:
        return d.flight_mode == FlightMode.FORMATION and d.swarm_behavior == "flocking"

    def apply(self, ctx: SwarmContext, velocities: np.ndarray) -> np.ndarray:
        if not self.active:
            return velocities
        idx = np.array([k for k, d in enumerate(ctx.drones) if self._is_member(d)], dtype=int)
        self.members = [ctx.drones[k].id for k in idx]
        if len(idx) == 0:
            self.stop()
            return velocities
        c = self.cfg
        p = ctx.positions[idx]
        v = ctx.velocities[idx]
        m = len(idx)

        diff = p[:, None, :] - p[None, :, :]                       # p_i - p_j
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(dist, np.inf)
        neighbour = dist < c.perception_radius
        count = neighbour.sum(axis=1)

        # Separation: linear falloff inside r_sep, direction away from each close neighbour.
        close = dist < c.separation_radius
        weight = np.where(close, 1.0 - dist / c.separation_radius, 0.0)
        unit = diff / np.where(np.isfinite(dist), np.maximum(dist, 1e-6), 1.0)[:, :, None]
        sep = np.einsum("ij,ijk->ik", weight, unit) * c.max_speed

        has = count > 0
        safe_count = np.maximum(count, 1)[:, None]
        ali = np.where(has[:, None], (neighbour.astype(float) @ v) / safe_count, 0.0)
        centroid = (neighbour.astype(float) @ p) / safe_count
        coh = np.where(has[:, None], c.cohesion_gain * (centroid - p), 0.0)

        goal = np.zeros((m, 3))
        if self.goal is not None:
            dc = self.config.drone
            for r in range(m):
                goal[r] = approach_velocity(p[r], self.goal, cruise_speed=c.max_speed, max_climb=dc.max_climb_rate,
                                            max_descent=dc.max_descent_rate, decel=2.0, gain=dc.position_gain)

        vel = (c.separation_weight * sep + c.alignment_weight * ali + c.cohesion_weight * coh
               + c.goal_weight * goal)   # + obstacle_weight * obstacle (Phase 3)

        if self.altitude is not None:
            vel[:, 2] = c.altitude_gain * (self.altitude - p[:, 2]) + 0.5 * vel[:, 2]

        out = velocities.copy()
        dc = self.config.drone
        speed_cap = min(c.max_speed, dc.max_horizontal_speed)
        for r, k in enumerate(idx):
            out[k] = clip_velocity(vel[r], speed_cap, dc.max_climb_rate, dc.max_descent_rate)
            ctx.drones[k].swarm_target = self.goal
        return out

    def snapshot(self) -> dict:
        c = self.cfg
        return {
            "members": list(self.members),
            "goal": ({"x": round(float(self.goal[0]), 2), "y": round(float(self.goal[1]), 2),
                      "z": round(float(self.goal[2]), 2)} if self.goal is not None else None),
            "altitude": self.altitude,
            "weights": self.weights(),
        }
