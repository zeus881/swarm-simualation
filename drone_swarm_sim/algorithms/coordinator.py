"""Swarm coordinator: owns the swarm behaviours and the operator commands that drive them.

Swarm modes (spec §25, modes 1-3):

* ``BASIC``      each drone flies its own commands (goto, RTL, ...). Collision avoidance still runs.
* ``FORMATION``  members keep formation slots around a virtual reference or a leader drone.
* ``FLOCKING``   members follow Reynolds flocking rules towards an optional goal.

A drone joins a behaviour through :meth:`Drone.enter_swarm_control` (flight mode FORMATION)
and leaves it on any other command or failsafe, so the operator always keeps authority.

Commands registered here (see ``GET /api/commands``):
``set_formation``, ``swarm_goto``, ``start_flocking``, ``release_swarm``, ``set_avoidance``,
``set_flocking_weights``. Every one is validated here and logged by the :class:`CommandProcessor`.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from simulation.commands import Command, CommandError, CommandProcessor, param_number, param_vector
from simulation.config import AVOIDANCE_METHODS, FORMATION_SHAPES, min_pairwise_distance
from simulation.drone import heading_to_yaw
from simulation.drone_interface import CommandResult
from simulation.events import EventCategory, Severity
from simulation.types import FlightMode

from .collision_avoidance import CollisionAvoidance
from .flocking import FlockingController
from .formation import FormationController
from .obstacle_avoidance import ObstacleAvoidance
from .path_planning import PathPlanner, PlanResult

if TYPE_CHECKING:
    from simulation.engine import SimulationEngine


AGL_STEP = 25.0      # m, spacing of terrain-following points on AGL legs


class SwarmMode(StrEnum):
    BASIC = "BASIC"
    FORMATION = "FORMATION"
    FLOCKING = "FLOCKING"


class SwarmCoordinator:
    def __init__(self, engine: "SimulationEngine") -> None:
        self.engine = engine
        cfg = engine.config
        self.formation = FormationController(cfg, engine.events)
        self.flocking = FlockingController(cfg)
        self.obstacle_avoidance = ObstacleAvoidance(cfg)
        self.avoidance = CollisionAvoidance(cfg)
        for stage in (self.formation, self.flocking, self.obstacle_avoidance, self.avoidance):
            engine.swarm.add_stage(stage)
        # Path planner around obstacles / terrain, shared by gotos, the formation reference and missions.
        self.planner = PathPlanner(engine.environment, cfg.obstacles, cfg.drone.radius)
        engine.planner = self.planner
        self._register(engine.commands)

    # ------------------------------------------------------------------ routing
    def route(self, start: np.ndarray, goal: np.ndarray, *, clearance: float | None = None,
              agl: float | None = None, who: str = "", planned: "PlanResult | None" = None) -> list[np.ndarray]:
        """Waypoints from ``start`` to ``goal`` around obstacles (last = goal).

        With ``agl`` set (terrain following) the legs are densified every ``AGL_STEP`` metres and every
        point is placed ``agl`` metres above the terrain under it. Planning failures fall back to the
        direct leg and log a warning (the obstacle-avoidance stage still protects the drone).
        """
        env, oc = self.engine.environment, self.engine.config.obstacles
        path: list[np.ndarray] = [np.asarray(goal, dtype=np.float64)]
        if oc.plan_paths and len(env.obstacles):
            result = planned or self.planner.plan(start, goal, clearance=clearance, follow_terrain=agl is not None)
            if result.ok:
                path = result.path
                if result.method != "direct":
                    self.engine.events.emit(EventCategory.MISSION, "path_planned",
                                            f"{who + ': ' if who else ''}{result.method} path with {len(path) - 1} "
                                            f"detour point(s), {result.length:.0f} m", severity=Severity.INFO)
            else:
                self.engine.events.emit(EventCategory.MISSION, "path_planning_failed",
                                        f"{who + ': ' if who else ''}{result.message} - flying direct",
                                        severity=Severity.WARNING)
        if agl is not None and env.terrain is not None:
            dense: list[np.ndarray] = []
            prev = np.asarray(start, dtype=np.float64)
            for p in path:
                n = max(1, int(np.hypot(*(p[:2] - prev[:2])) / AGL_STEP))
                for k in range(1, n + 1):
                    q = prev + (p - prev) * (k / n)
                    q[2] = env.ground_height(q[0], q[1]) + agl
                    dense.append(q)
                prev = p
            path = dense
        return path

    # ------------------------------------------------------------------ state
    @property
    def mode(self) -> SwarmMode:
        if self.formation.active:
            return SwarmMode.FORMATION
        if self.flocking.active:
            return SwarmMode.FLOCKING
        return SwarmMode.BASIC

    def snapshot(self) -> dict[str, Any]:
        sw = self.engine.config.swarm
        av = self.avoidance
        return {
            "mode": str(self.mode),
            "avoidance": {"enabled": av.enabled, "method": av.method, "active_pairs": av.active_pairs,
                          "orca_solves": av.orca_solves, "fallbacks": av.fallbacks,
                          "filter_interventions": av.filter_interventions, "min_separation": av.min_separation},
            "obstacles": {"enabled": self.obstacle_avoidance.enabled, "active_drones": self.obstacle_avoidance.active_drones,
                          "min_distance": (round(self.obstacle_avoidance.min_distance, 1)
                                           if self.obstacle_avoidance.min_distance is not None else None),
                          "planner": self.engine.config.obstacles.planner},
            "formation": self.formation.snapshot() if self.formation.active else None,
            "flocking": self.flocking.snapshot() if self.flocking.active else None,
            "flocking_weights": self.flocking.weights(),
            "shapes": [s for s in FORMATION_SHAPES if s != "custom" or self.formation.custom],
            "custom_offsets": self.formation.custom,
            "default_spacing": sw.formation_spacing,
            "default_layers": sw.formation_altitude_layers,
            "default_layer_spacing": sw.formation_layer_spacing,
        }

    # ------------------------------------------------------------------ commands
    def _register(self, commands: CommandProcessor) -> None:
        commands.register("set_formation", self._set_formation,
                          "Fly a formation {shape, spacing?, reference? virtual|leader, leader_id?, heading?, "
                          "altitude?, layers?, layer_spacing?, offsets? [[fwd,left,up],...] (custom)}")
        commands.register("swarm_goto", self._swarm_goto,
                          "Move the active formation / flock {position [x,y,z], speed?, heading?}")
        commands.register("start_flocking", self._start_flocking, "Reynolds flocking {altitude?, goal?}")
        commands.register("release_swarm", self._release, "End formation/flocking; members hover")
        commands.register("set_avoidance", self._set_avoidance,
                          "Collision avoidance {enabled?, method? orca|potential_field}")
        commands.register("set_flocking_weights", self._set_weights,
                          "Flocking weights {separation?, alignment?, cohesion?, goal?}")

    def _custom_offsets(self, value: Any) -> list[list[float]]:
        """Validate a runtime custom formation: ``[[forward, left, up], ...]`` in metres."""
        sw = self.engine.config.swarm
        if not isinstance(value, (list, tuple)) or not value:
            raise CommandError("offsets must be a non-empty list of [forward, left, up]")
        if len(value) > self.engine.config.simulation.max_drones:
            raise CommandError(f"offsets may hold at most {self.engine.config.simulation.max_drones} slots")
        out = []
        for k, p in enumerate(value):
            if (not isinstance(p, (list, tuple)) or len(p) != 3
                    or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in p)):
                raise CommandError(f"offsets[{k}] must be three finite numbers [forward, left, up]")
            if max(abs(x) for x in p) > 1000:
                raise CommandError(f"offsets[{k}] is more than 1000 m from the formation centre")
            out.append([float(x) for x in p])
        closest = min_pairwise_distance(out)
        if closest < sw.separation_distance:
            raise CommandError(f"custom slots are only {closest:.1f} m apart (minimum {sw.separation_distance:g} m)")
        return out

    def _candidates(self, cmd: Command):
        drones = self.engine.swarm.select(cmd.drone_ids)
        flying = [d for d in drones if d.in_flight and d.flight_mode != FlightMode.EMERGENCY and not d.failed]
        if not flying:
            raise CommandError("no airborne drones in the selection - take off first")
        return flying

    def _set_formation(self, cmd: Command) -> CommandResult:
        p = cmd.params
        sw = self.engine.config.swarm
        shape = str(p.get("shape", self.formation.shape)).lower()
        if shape not in FORMATION_SHAPES:
            raise CommandError(f"shape must be one of {', '.join(FORMATION_SHAPES)}")
        custom = self._custom_offsets(p["offsets"]) if p.get("offsets") is not None else None
        if custom is not None and shape != "custom":
            raise CommandError("offsets can only be used with shape 'custom'")
        if shape == "custom" and custom is None and not self.formation.custom:
            raise CommandError("no custom formation defined (give 'offsets' or set swarm.custom_formation)")
        spacing = param_number(p, "spacing", default=self.formation.spacing, minimum=sw.separation_distance * 1.2,
                               maximum=200.0)
        reference = p.get("reference", sw.formation_reference if not self.formation.active else self.formation.reference_mode)
        if reference not in ("virtual", "leader"):
            raise CommandError("reference must be 'virtual' or 'leader'")
        heading = param_number(p, "heading", minimum=-360.0, maximum=360.0)
        altitude = param_number(p, "altitude", minimum=self.engine.config.drone.min_altitude,
                                maximum=self.engine.environment.max_altitude)
        layers = param_number(p, "layers", default=self.formation.layers, minimum=1, maximum=10)
        if layers != int(layers):
            raise CommandError("parameter 'layers' must be a whole number")
        layer_spacing = param_number(p, "layer_spacing", default=self.formation.layer_spacing,
                                     minimum=sw.min_separation, maximum=100.0)

        flying = self._candidates(cmd)
        leader = None
        if reference == "leader":
            leader_id = p.get("leader_id", flying[0].id)
            leader = next((d for d in flying if d.id == leader_id), None)
            if leader is None:
                raise CommandError(f"leader D{leader_id:02d} is not an airborne member of the selection"
                                   if isinstance(leader_id, int) else "leader_id must be an integer")
            if len(flying) < 2:
                raise CommandError("leader-follower needs at least two airborne drones")
            if leader.flight_mode == FlightMode.FORMATION:
                leader.hover()     # the leader flies under operator/mission control, not in a slot
        followers = [d for d in flying if d is not leader]

        if self.flocking.active:
            self.flocking.stop()
        self.formation.configure(shape=shape, spacing=spacing, reference=reference,
                                 leader_id=leader.id if leader else None,
                                 heading_yaw=heading_to_yaw(heading) if heading is not None else None,
                                 layers=int(layers), layer_spacing=layer_spacing, custom=custom)
        ground = self.engine.environment.ground_level
        if not self.formation.active:
            positions = np.array([d.position for d in followers])
            self.formation.start(positions, altitude=None if altitude is None else ground + altitude)
        elif altitude is not None:
            self.formation.ref_pos[2] = ground + altitude

        results = [(d, d.enter_swarm_control("formation")) for d in followers]
        joined = [d for d, r in results if r.success]
        if not joined:
            self.formation.stop()
            return CommandResult.fail("no drone could join the formation")
        details = {d.name: r.message for d, r in results if not r.success}
        msg = f"{shape.upper()} formation, {len(joined)} drones, spacing {spacing:.0f} m"
        if layers > 1:
            msg += f", {int(layers)} layers {layer_spacing:.0f} m apart"
        if leader:
            msg += f", following leader {leader.name}"
        return CommandResult(True, msg, details, {"members": [d.id for d in joined],
                                                  "leader": leader.id if leader else None})

    def _swarm_goto(self, cmd: Command) -> CommandResult:
        p = cmd.params
        if "position" not in p:
            raise CommandError("swarm_goto needs 'position' [x, y, z]")
        env = self.engine.environment
        raw = param_vector(p["position"], "position")
        agl = p.get("agl", False)
        if not isinstance(agl, bool):
            raise CommandError("parameter 'agl' must be true/false")
        if agl:                                        # z is height above the terrain under the target
            raw[2] += env.ground_height(raw[0], raw[1])
        target = env.clamp(raw, min_agl=self.engine.config.drone.min_altitude)
        speed = param_number(p, "speed", minimum=0.5, maximum=self.engine.config.drone.max_horizontal_speed)
        heading = param_number(p, "heading", minimum=-360.0, maximum=360.0)
        mode = self.mode
        where = f"({target[0]:.0f}, {target[1]:.0f}, {target[2]:.0f})"
        agl_alt = float(target[2] - env.ground_height(target[0], target[1])) if agl else None
        if mode == SwarmMode.FORMATION:
            if self.formation.reference_mode == "leader" and self.formation.leader_id is not None:
                leader = self.engine.swarm.get(self.formation.leader_id)
                route = self.route(leader.position, target, agl=agl_alt, who="formation leader")
                result = leader.goto_path(route, speed=speed, heading=heading)
                return CommandResult(result.success, f"leader {leader.name}: {result.message}")
            route = self._formation_route(target, agl_alt)
            self.formation.move_to(target, speed, heading_to_yaw(heading) if heading is not None else None,
                                   route=route[:-1])
            return CommandResult.ok(f"formation moving to {where}"
                                    + (f" via {len(route) - 1} waypoint(s)" if len(route) > 1 else ""))
        if mode == SwarmMode.FLOCKING:
            self.flocking.set_goal(target)
            return CommandResult.ok(f"flock heading to {where}")
        return CommandResult.fail("no formation or flock is active - use goto")

    def _formation_route(self, target: np.ndarray, agl: float | None) -> list[np.ndarray]:
        """Route for the formation reference: keep the whole shape clear if possible (clearance grown by
        the formation radius), otherwise plan for the reference and let obstacle avoidance handle members."""
        oc = self.engine.config.obstacles
        start = self.formation.ref_pos
        if oc.plan_paths and len(self.engine.environment.obstacles):
            wide = self.planner.plan(start, target, clearance=oc.clearance + self.formation.radius(),
                                     follow_terrain=agl is not None)
            if wide.ok:
                return self.route(start, target, agl=agl, who="formation", planned=wide)
        return self.route(start, target, agl=agl, who="formation")

    def _start_flocking(self, cmd: Command) -> CommandResult:
        p = cmd.params
        flying = self._candidates(cmd)
        if len(flying) < 2:
            raise CommandError("flocking needs at least two airborne drones")
        ground = self.engine.environment.ground_level
        altitude = param_number(p, "altitude", minimum=self.engine.config.drone.min_altitude,
                                maximum=self.engine.environment.max_altitude)
        goal = param_vector(p["goal"], "goal") if p.get("goal") is not None else None
        if self.formation.active:
            self.formation.stop()
        mean_z = float(np.mean([d.position[2] for d in flying]))
        self.flocking.start(altitude=ground + altitude if altitude is not None else mean_z, goal=goal)
        results = [(d, d.enter_swarm_control("flocking")) for d in flying]
        joined = [d for d, r in results if r.success]
        if not joined:
            self.flocking.stop()
            return CommandResult.fail("no drone could join the flock")
        return CommandResult(True, f"flocking with {len(joined)} drones",
                             {d.name: r.message for d, r in results if not r.success}, {"members": [d.id for d in joined]})

    def _release(self, cmd: Command) -> CommandResult:
        members = [d for d in self.engine.swarm.select(cmd.drone_ids) if d.flight_mode == FlightMode.FORMATION]
        for d in members:
            d.hover()
        if cmd.drone_ids is None or not any(d.flight_mode == FlightMode.FORMATION for d in self.engine.swarm):
            self.formation.stop()
            self.flocking.stop()
        return CommandResult.ok(f"released {len(members)} drones (holding position)")

    def _set_avoidance(self, cmd: Command) -> CommandResult:
        enabled = cmd.params.get("enabled")
        method = cmd.params.get("method")
        if enabled is None and method is None:
            raise CommandError("give 'enabled' (true/false) and/or 'method' (orca|potential_field)")
        if enabled is not None and not isinstance(enabled, bool):
            raise CommandError("parameter 'enabled' must be true/false")
        if method is not None and method not in AVOIDANCE_METHODS:
            raise CommandError(f"parameter 'method' must be one of {', '.join(AVOIDANCE_METHODS)}")
        if enabled is not None:
            self.avoidance.enabled = enabled
        if method is not None:
            self.avoidance.method = method
        label = "ORCA" if self.avoidance.method == "orca" else "potential field"
        return CommandResult.ok(f"collision avoidance {'ON' if self.avoidance.enabled else 'OFF'} ({label})")

    def _set_weights(self, cmd: Command) -> CommandResult:
        weights = {k: param_number(cmd.params, k, minimum=0.0, maximum=10.0)
                   for k in ("separation", "alignment", "cohesion", "goal") if k in cmd.params}
        if not weights:
            raise CommandError("give at least one of separation, alignment, cohesion, goal")
        self.flocking.set_weights(**weights)
        return CommandResult.ok("flocking weights " + ", ".join(f"{k}={v:g}" for k, v in weights.items()))
