"""Reciprocal collision avoidance: ORCA velocity obstacles, potential-field fallback, hard safety filter.

The stage runs last in the velocity pipeline (priority 40), so it overrides formation and mission
velocities (collision > obstacle > formation > mission). It has three layers:

1. **ORCA** (primary, ``swarm.avoidance_method: orca``, see :mod:`algorithms.orca`). For every
   manoeuvrable drone and its ``orca_max_neighbors`` closest airborne neighbours, a half-space of
   collision-free velocities over ``orca_time_horizon`` is built; the new velocity is the one closest to
   the desired velocity inside all half-spaces. Responsibility is shared 50/50, or taken fully when the
   neighbour cannot manoeuvre (motors killed, still on its takeoff roll).

2. **Potential field** (fallback when a drone's ORCA program is infeasible, or selected with
   ``avoidance_method: potential_field``)::

       |v_rep| = v_max * ((d0 - d) / (d0 - d_col))^2      along  n = (p_i - p_j) / d

   plus a closest-point-of-approach push on predicted conflicts
   (``t* = -(dp . dv) / |dv|^2``, ``d_cpa = |dp + dv t*|``) and yielding (slow down in proportion to
   ``1 - t*/horizon``).

3. **Hard safety filter** (always on while avoidance is enabled). A discrete control-barrier function
   on every pair keeps the closing speed below the speed from which both drones can still brake before
   reaching the hard floor ``d_min = swarm.min_separation``::

       h = d - d_min - c_act * tau_lag - margin      (tau_lag: velocity-loop lag)
       closing speed  c = -(v_i - v_j) . n_ij  <=  sqrt(2 * a_rel * h)        (h > 0)
                                               <=  -h / T_push                (h <= 0: push apart)

   ``a_rel`` is the braking both drones can contribute (``safety_brake_fraction`` of the horizontal
   acceleration limit each). Excess closing speed is removed along ``n_ij`` (split between the two
   drones, or taken fully by the one that can manoeuvre); a few Gauss-Seidel sweeps resolve drones that
   are in several close pairs at once. Because the constraint is written on the braking distance, the
   minimum separation is kept even when ORCA's idealised "instant velocity change" assumption fails.

With avoidance disabled (operator override, ``set_avoidance``) none of the layers run.
"""

from __future__ import annotations

import numpy as np

from simulation.config import SimConfig
from simulation.dynamics import clip_velocity
from simulation.swarm import SwarmContext, VelocityStage
from simulation.types import FlightMode

from .orca import linear_program3, orca_planes

T_MIN = 0.5            # s, floor for time-to-conflict (avoids huge corrections for imminent conflicts)
VERTICAL_SHARE = 0.3   # potential field: avoid mostly horizontally (climbing wastes energy, upsets formations)
FILTER_SWEEPS = 4      # Gauss-Seidel passes of the safety filter
PUSH_TIME = 1.0        # s, time over which the safety filter resolves a breach of the hard floor
FILTER_MARGIN = 0.3    # m, extra clearance kept by the safety filter above min_separation


class CollisionAvoidance(VelocityStage):
    name = "collision_avoidance"
    priority = 40

    def __init__(self, config: SimConfig) -> None:
        sw = config.swarm
        dc = config.drone
        self.enabled = sw.collision_avoidance
        self.method = sw.avoidance_method
        self.d0 = sw.avoidance_radius
        self.d_col = sw.collision_distance
        self.separation = sw.separation_distance
        self.min_separation = sw.min_separation
        self.r_safe = sw.separation_distance + sw.avoidance_margin
        self.v_max = sw.avoidance_max_speed
        self.horizon = sw.avoidance_horizon
        self.slowdown = sw.avoidance_slowdown
        self.tau = sw.orca_time_horizon
        self.orca_k = sw.orca_max_neighbors
        self.limits = (dc.max_horizontal_speed, dc.max_climb_rate, dc.max_descent_rate)
        self.max_speed = dc.max_horizontal_speed
        # Braking the safety filter may assume per drone, and the lag before a new velocity command bites.
        max_h_accel = min(dc.max_acceleration, 9.80665 * np.tan(np.radians(dc.max_tilt_deg)))
        self.brake = sw.safety_brake_fraction * max_h_accel
        self.lag = dc.thrust_time_constant + 1.0 / max(dc.velocity_kp, 1e-6)
        self.search_radius = min(max(self.d0, self.r_safe) + self.horizon * 2.0 * dc.max_horizontal_speed, 150.0)
        self.orca_radius = min(self.r_safe + self.tau * 2.0 * dc.max_horizontal_speed, self.search_radius)
        # telemetry
        self.active_pairs = 0
        self.orca_solves = 0
        self.fallbacks = 0
        self.filter_interventions = 0

    # ------------------------------------------------------------------ pipeline stage
    def apply(self, ctx: SwarmContext, velocities: np.ndarray) -> np.ndarray:
        n = len(ctx.drones)
        self.active_pairs = self.orca_solves = self.fallbacks = self.filter_interventions = 0
        if n < 2:
            return velocities
        # Drones that can manoeuvre: flying with motors on. A drone falling with motors killed is still
        # an obstacle for the others (it stays in the pair set) but receives no correction itself.
        active = ctx.controllable & np.fromiter(
            (d.flight_mode != FlightMode.TAKEOFF or d.altitude_agl > 2.0 for d in ctx.drones), bool, n)
        pairs, dists = ctx.index.query_pairs(self.search_radius)
        if len(pairs) == 0:
            return velocities
        keep = ctx.airborne[pairs[:, 0]] & ctx.airborne[pairs[:, 1]] & (active[pairs[:, 0]] | active[pairs[:, 1]])
        pairs, dists = pairs[keep], dists[keep]
        if len(pairs) == 0:
            return velocities

        if self.method == "orca":
            out = self._orca(ctx, velocities, pairs, dists, active)
        else:
            out = self._potential_field(ctx, velocities, pairs, dists, active)

        max_h, max_up, max_down = self.limits
        changed = active & np.any(np.abs(out - velocities) > 1e-9, axis=1)
        for k in np.flatnonzero(changed):
            out[k] = clip_velocity(out[k], max_h, max_up, max_down)

        out = self._safety_filter(ctx, out, pairs, dists, active)
        return out

    # ------------------------------------------------------------------ ORCA
    def _orca(self, ctx: SwarmContext, velocities: np.ndarray, pairs: np.ndarray, dists: np.ndarray,
              active: np.ndarray) -> np.ndarray:
        near = dists < self.orca_radius
        pairs, dists = pairs[near], dists[near]
        if len(pairs) == 0:
            return velocities
        i, j = pairs[:, 0], pairs[:, 1]
        # Ordered (agent, neighbour) rows for every manoeuvrable agent.
        agent = np.concatenate((i[active[i]], j[active[j]]))
        other = np.concatenate((j[active[i]], i[active[j]]))
        dist = np.concatenate((dists[active[i]], dists[active[j]]))
        if len(agent) == 0:
            return velocities
        # Keep the K closest neighbours per agent, closest first (the LP is order dependent: near first).
        order = np.lexsort((dist, agent))
        agent, other, dist = agent[order], other[order], dist[order]
        starts = np.r_[0, np.flatnonzero(np.diff(agent)) + 1]
        rank = np.arange(len(agent)) - np.repeat(starts, np.diff(np.r_[starts, len(agent)]))
        sel = rank < self.orca_k
        agent, other, dist = agent[sel], other[sel], dist[sel]

        p, v = ctx.positions, ctx.velocities
        share = np.where(active[other], 0.5, 1.0)
        points, normals = orca_planes(p[other] - p[agent], v[agent] - v[other], v[agent], self.r_safe, self.tau,
                                      T_MIN, share)
        # Only agents whose preferred velocity violates one of their half-spaces need a linear program.
        slack = np.einsum("ij,ij->i", velocities[agent] - points, normals)
        violated = slack < -1e-9
        self.active_pairs = int(np.count_nonzero(violated))
        if not np.any(violated):
            return velocities
        out = velocities.copy()
        fallback = None
        bounds = np.r_[0, np.flatnonzero(np.diff(agent)) + 1, len(agent)].tolist()
        needs = set(agent[violated].tolist())
        planes = np.hstack((points, normals)).tolist()     # plain floats: fast scalar maths in the LP
        prefs = velocities.tolist()
        for s, e in zip(bounds[:-1], bounds[1:]):
            k = int(agent[s])
            if k not in needs:
                continue
            self.orca_solves += 1
            result, fail = linear_program3(planes[s:e], self.max_speed, tuple(prefs[k]))
            if fail == e - s:
                out[k] = result
            else:
                # Dense conflict with no collision-free velocity: use the potential field for this drone.
                if fallback is None:
                    fallback = self._potential_field(ctx, velocities, pairs, dists, active)
                out[k] = fallback[k]
                self.fallbacks += 1
        return out

    # ------------------------------------------------------------------ potential field (+ CPA)
    def _potential_field(self, ctx: SwarmContext, velocities: np.ndarray, pairs: np.ndarray, dists: np.ndarray,
                         active: np.ndarray) -> np.ndarray:
        i, j = pairs[:, 0], pairs[:, 1]
        p = ctx.positions
        v = velocities
        dp = p[i] - p[j]
        d = np.maximum(dists, 1e-6)
        n_ij = dp / d[:, None]
        coincident = dists < 1e-6
        if np.any(coincident):
            n_ij[coincident] = np.array([1.0, 0.0, 0.0])
        corr = np.zeros_like(v)

        # 1. bounded repulsive potential
        close = d < self.d0
        if np.any(close):
            frac = np.clip((self.d0 - d[close]) / (self.d0 - self.d_col), 0.0, 1.0)
            push = (self.v_max * frac * frac)[:, None] * n_ij[close]
            np.add.at(corr, i[close], push)
            np.add.at(corr, j[close], -push)

        # 2. closest point of approach on the proposed (desired) velocities
        dv = v[i] - v[j]
        dv2 = np.einsum("ij,ij->i", dv, dv)
        moving = dv2 > 1e-6
        t_star = np.where(moving, -np.einsum("ij,ij->i", dp, dv) / np.where(moving, dv2, 1.0), -1.0)
        cpa = dp + dv * t_star[:, None]
        d_cpa = np.linalg.norm(cpa, axis=1)
        conflict = moving & (t_star > 0) & (t_star < self.horizon) & (d_cpa < self.r_safe)
        if np.any(conflict):
            miss = cpa[conflict]
            dm = d_cpa[conflict]
            head_on = dm < 1e-3
            miss_dir = np.where(dm[:, None] > 1e-3, miss / np.maximum(dm, 1e-9)[:, None], 0.0)
            if np.any(head_on):
                # Veer right: rotate the relative velocity by -90 deg in the horizontal plane.
                rel = dv[conflict][head_on]
                right = np.stack((rel[:, 1], -rel[:, 0], np.zeros(len(rel))), axis=1)
                norm = np.linalg.norm(right, axis=1)
                right[norm > 1e-9] /= norm[norm > 1e-9][:, None]
                right[norm <= 1e-9] = np.array([0.0, 1.0, 0.0])
                miss_dir[head_on] = right
            mag = 0.5 * (self.r_safe - dm) / np.maximum(t_star[conflict], T_MIN)
            push = np.minimum(mag, self.v_max)[:, None] * miss_dir
            np.add.at(corr, i[conflict], push)
            np.add.at(corr, j[conflict], -push)

        if self.method == "potential_field":
            self.active_pairs = int(np.count_nonzero(close | conflict))

        # Yield: slow the goal velocity in proportion to the urgency u = 1 - t*/horizon, so the
        # sideways correction has time to work. In dense convergences this matters more than pushing harder.
        scale = np.ones(len(v))
        if np.any(conflict) and self.slowdown > 0:
            urgency = 1.0 - t_star[conflict] / self.horizon
            factor = 1.0 - self.slowdown * urgency
            np.minimum.at(scale, i[conflict], factor)
            np.minimum.at(scale, j[conflict], factor)
        base = velocities * np.where(active, scale, 1.0)[:, None]
        corr[:, 2] *= VERTICAL_SHARE
        return base + np.where(active[:, None], corr, 0.0)

    # ------------------------------------------------------------------ hard safety filter
    def _safety_filter(self, ctx: SwarmContext, velocities: np.ndarray, pairs: np.ndarray, dists: np.ndarray,
                       active: np.ndarray) -> np.ndarray:
        i, j = pairs[:, 0], pairs[:, 1]
        a_i, a_j = active[i], active[j]
        both = a_i & a_j
        dp = ctx.positions[i] - ctx.positions[j]
        d = np.maximum(dists, 1e-6)
        n_ij = dp / d[:, None]                   # unit vector from j to i
        dv_act = ctx.velocities[i] - ctx.velocities[j]
        closing_now = np.maximum(-np.einsum("ij,ij->i", dv_act, n_ij), 0.0)
        # Distance still available for braking once the lag of the velocity loop has elapsed.
        h = d - self.min_separation - FILTER_MARGIN - closing_now * (self.lag + ctx.dt)
        a_rel = self.brake * (a_i.astype(float) + a_j.astype(float))
        allowed = np.where(h > 0, np.sqrt(2.0 * a_rel * np.maximum(h, 0.0)), h / PUSH_TIME)
        # Only pairs that could possibly bind: the largest closing speed is 2 * v_max.
        relevant = allowed < 2.0 * self.max_speed + 1.0
        if not np.any(relevant):
            return velocities
        i, j, n_ij, allowed, a_i, a_j, both = (x[relevant] for x in (i, j, n_ij, allowed, a_i, a_j, both))
        out = velocities.copy()
        out[~active] = ctx.velocities[~active]   # a drone that cannot manoeuvre keeps flying as it is
        share_i = np.where(both, 0.5, np.where(a_i, 1.0, 0.0))
        share_j = np.where(both, 0.5, np.where(a_j, 1.0, 0.0))
        for _ in range(FILTER_SWEEPS):
            closing = -np.einsum("ij,ij->i", out[i] - out[j], n_ij)
            excess = closing - allowed
            bad = excess > 1e-6
            if not np.any(bad):
                break
            fix = excess[bad][:, None] * n_ij[bad]
            np.add.at(out, i[bad], share_i[bad][:, None] * fix)
            np.add.at(out, j[bad], -share_j[bad][:, None] * fix)
            self.filter_interventions += int(np.count_nonzero(bad))
        # The filter only ever changes manoeuvrable drones; keep non-active rows exactly as they were.
        # Its output is deliberately not re-clipped: briefly exceeding the cruise envelope to brake is
        # preferable to breaking the separation floor (the velocity loop still limits acceleration).
        out[~active] = velocities[~active]
        return out
