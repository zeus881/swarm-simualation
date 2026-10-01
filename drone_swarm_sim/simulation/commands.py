"""Operator command processing.

A command is ``{"type": str, "drone_ids": [int] | None, "params": {...}}``;
``drone_ids = None`` addresses the whole swarm. Commands are executed in the
engine thread at a tick boundary, which keeps runs deterministic and
reproducible from a command log.

Later phases register additional commands (formation, mission, targets,
obstacles) through :meth:`CommandProcessor.register`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from .drone import Drone
from .drone_interface import CommandResult
from .events import EventCategory, Severity

if TYPE_CHECKING:
    from .engine import SimulationEngine

log = logging.getLogger(__name__)


class CommandError(ValueError):
    """Invalid command or parameters."""


@dataclass(slots=True)
class Command:
    type: str
    drone_ids: list[int] | None = None
    params: dict[str, Any] = field(default_factory=dict)
    issued_by: str | None = None        # set by the server from the authenticated user, never by the payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Command":
        if not isinstance(data, Mapping):
            raise CommandError("command must be an object")
        ctype = data.get("type")
        if not isinstance(ctype, str) or not ctype:
            raise CommandError("command.type must be a non-empty string")
        ids = data.get("drone_ids")
        if ids is not None:
            if not isinstance(ids, (list, tuple)) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
                raise CommandError("command.drone_ids must be a list of integers or null")
            ids = list(ids)
        params = data.get("params") or {}
        if not isinstance(params, Mapping):
            raise CommandError("command.params must be an object")
        return cls(ctype, ids, dict(params))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "drone_ids": self.drone_ids, "params": self.params}


# ----------------------------------------------------------------------------- param helpers

def _number(params: Mapping[str, Any], key: str, default: float | None = None, *, required: bool = False,
            minimum: float | None = None, maximum: float | None = None) -> float | None:
    value = params.get(key, default)
    if value is None:
        if required:
            raise CommandError(f"parameter '{key}' is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CommandError(f"parameter '{key}' must be a finite number")
    if minimum is not None and value < minimum:
        raise CommandError(f"parameter '{key}' must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise CommandError(f"parameter '{key}' must be <= {maximum}")
    return float(value)


def _vector(value: Any, key: str) -> np.ndarray:
    if isinstance(value, Mapping):
        try:
            value = [value["x"], value["y"], value["z"]]
        except KeyError as exc:
            raise CommandError(f"parameter '{key}' needs x, y and z") from exc
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise CommandError(f"parameter '{key}' must be [x, y, z]") from exc
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise CommandError(f"parameter '{key}' must be three finite numbers")
    return arr


param_number = _number
param_vector = _vector
"""Public aliases so modules that register extra commands validate parameters identically."""

Handler = Callable[[Command], CommandResult]


class CommandProcessor:
    def __init__(self, engine: "SimulationEngine") -> None:
        self._engine = engine
        self._handlers: dict[str, tuple[Handler, str]] = {}
        none = lambda p: ()   # noqa: E731 - commands without parameters
        for name, handler, doc in (
            ("arm", self._per_drone(none, lambda d, a: d.arm()), "Arm motors on the ground"),
            ("disarm", self._per_drone(none, lambda d, a: d.disarm()), "Disarm on the ground"),
            ("takeoff", self._per_drone(lambda p: (_number(p, "altitude", minimum=0.0),), lambda d, a: d.takeoff(*a)),
             "Take off {altitude?}"),
            ("land", self._per_drone(none, lambda d, a: d.land()), "Land in place"),
            ("hover", self._per_drone(none, lambda d, a: d.hover()), "Hold current position"),
            ("return_to_home", self._per_drone(none, lambda d, a: d.return_to_home()), "Return to home pad and land"),
            ("emergency_stop", self._per_drone(none, lambda d, a: d.emergency_stop()),
             "Emergency descent (or motor kill, per config)"),
            ("set_heading", self._per_drone(lambda p: (_number(p, "heading", required=True),),
                                            lambda d, a: d.set_heading(*a)), "Set heading {heading deg}"),
            ("set_altitude", self._per_drone(lambda p: (_number(p, "altitude", required=True, minimum=0.0),),
                                             lambda d, a: d.change_altitude(*a)), "Change altitude {altitude m AGL}"),
            ("set_velocity", self._per_drone(lambda p: (_vector(p.get("velocity"), "velocity"), _number(p, "heading")),
                                             lambda d, a: d.set_velocity(*a)),
             "Offboard velocity {velocity [x,y,z], heading?}"),
            ("goto", self._goto, "Fly to {position [x,y,z] | lat/lon/alt, speed?, heading?, keep_formation?, "
                                 "agl? (z above terrain), plan? (route around obstacles, default true)}"),
            ("add_drone", self._add_drone, "Add drones {count?, position?}"),
            ("remove_drone", self._remove_drone, "Remove drones (drone_ids required)"),
            ("set_wind", self._set_wind, "Set wind {speed?, direction?, gust_strength?, enabled?}"),
        ):
            self.register(name, handler, doc)

    # ------------------------------------------------------------------ registry
    def register(self, name: str, handler: Handler, description: str = "") -> None:
        self._handlers[name] = (handler, description)

    def available(self) -> dict[str, str]:
        return {name: doc for name, (_, doc) in sorted(self._handlers.items())}

    def execute(self, command: Command | Mapping[str, Any]) -> CommandResult:
        try:
            cmd = command if isinstance(command, Command) else Command.from_dict(command)
        except CommandError as exc:
            return CommandResult.fail(str(exc))
        entry = self._handlers.get(cmd.type)
        if entry is None:
            return CommandResult.fail(f"unknown command '{cmd.type}'")
        try:
            result = entry[0](cmd)
        except CommandError as exc:
            result = CommandResult.fail(f"{cmd.type}: {exc}")
        except KeyError as exc:
            result = CommandResult.fail(f"{cmd.type}: {exc.args[0] if exc.args else exc}")
        except ValueError as exc:
            result = CommandResult.fail(f"{cmd.type}: {exc}")
        except Exception as exc:  # never let a command crash the engine thread
            log.exception("Command %s failed", cmd.type)
            result = CommandResult.fail(f"{cmd.type}: internal error ({exc.__class__.__name__})")
        self._engine.events.emit(
            EventCategory.COMMAND, cmd.type, f"{cmd.type}: {result.message}",
            severity=Severity.INFO if result.success else Severity.WARNING,
            drone_ids=cmd.drone_ids, params=_jsonable(cmd.params), success=result.success,
            **({"user": cmd.issued_by} if cmd.issued_by else {}),
        )
        return result

    # ------------------------------------------------------------------ handlers
    def _per_drone(self, parse: Callable[[Mapping[str, Any]], tuple],
                   act: Callable[[Drone, tuple], CommandResult]) -> Handler:
        """Handler for a command addressed to individual drones. Parameters are validated first (errors go
        straight back to the operator); the action then runs directly, or over the radio link."""
        def handler(cmd: Command) -> CommandResult:
            drones = self._engine.swarm.select(cmd.drone_ids)
            if not drones:
                return CommandResult.fail("no drones selected")
            args = parse(cmd.params)
            return self._aggregate(cmd.type, [(d, self.dispatch(d, cmd.type, lambda d=d: act(d, args)))
                                              for d in drones])
        return handler

    def dispatch(self, drone: Drone, name: str, action: Callable[[], CommandResult]) -> CommandResult:
        """Run a drone action now, or send it over the communication link when that model is enabled."""
        comms = getattr(self._engine, "comms", None)
        if comms is None or getattr(drone, "is_remote", False):     # real vehicles use their own link
            return action()
        return comms.send(drone, name, action, self._engine.sim_time)

    @staticmethod
    def _aggregate(name: str, results: list[tuple[Drone, CommandResult]]) -> CommandResult:
        accepted = [d for d, r in results if r.success]
        details = {d.name: r.message for d, r in results if not r.success}
        if len(results) == 1:
            d, r = results[0]
            return CommandResult(r.success, f"{d.name}: {r.message}", details)
        message = f"{len(accepted)}/{len(results)} accepted"
        if details:
            first = next(iter(details.values()))
            message += f" ({len(details)} rejected: {first})"
        return CommandResult(bool(accepted), message, details, {"accepted": [d.id for d in accepted]})

    def _goto(self, cmd: Command) -> CommandResult:
        p = cmd.params
        if "position" in p:
            target = _vector(p["position"], "position")
        elif {"lat", "lon"} <= p.keys():
            lat = _number(p, "lat", required=True, minimum=-90, maximum=90)
            lon = _number(p, "lon", required=True, minimum=-180, maximum=180)
            geo = self._engine.geo
            alt_amsl = _number(p, "alt", default=geo.origin.altitude + self._engine.config.drone.takeoff_altitude)
            target = geo.geodetic_to_enu(lat, lon, alt_amsl)
        else:
            raise CommandError("goto needs 'position' [x, y, z] or 'lat'/'lon'[/'alt']")
        speed = _number(p, "speed", minimum=0.1)
        heading = _number(p, "heading")
        agl = p.get("agl", False)
        if not isinstance(agl, bool):
            raise CommandError("parameter 'agl' must be true/false")
        env = self._engine.environment
        agl_alt = float(target[2]) if agl else None
        drones = self._engine.swarm.select(cmd.drone_ids)
        if not drones:
            return CommandResult.fail("no drones selected")
        keep = bool(p.get("keep_formation", len(drones) > 1))
        flying = [d for d in drones if d.in_flight]
        centroid = np.mean([d.position for d in flying], axis=0) if flying else np.zeros(3)
        coordinator = getattr(self._engine, "coordinator", None)
        plan = bool(p.get("plan", True)) and coordinator is not None
        # One route for a group that keeps its shape (planned for the centroid), one per drone otherwise.
        group_route = None
        if plan and keep and len(flying) > 1:
            goal = target.copy()
            if agl_alt is not None:
                goal[2] = env.ground_height(goal[0], goal[1]) + agl_alt
            group_route = coordinator.route(centroid, goal, agl=agl_alt, who=f"goto ({len(flying)} drones)")
        results = []
        for d in drones:
            offset = np.zeros(3)
            if keep and flying:
                # Translate the group rigidly in the horizontal plane: every drone keeps its
                # offset from the group centroid, so a group goto never stacks drones on one point.
                offset[:2] = d.position[:2] - centroid[:2]
            goal = target + offset
            if agl_alt is not None:
                goal[2] = env.ground_height(goal[0], goal[1]) + agl_alt
            if group_route is not None and d.in_flight:
                route = [q + offset for q in group_route]
                if agl_alt is not None:
                    for q in route:
                        q[2] = env.ground_height(q[0], q[1]) + agl_alt
            elif plan and d.in_flight:
                route = coordinator.route(d.position, goal, agl=agl_alt, who=f"goto {d.name}")
            else:
                route = [goal]
            results.append((d, self.dispatch(d, "goto", lambda d=d, r=route: d.goto_path(r, speed=speed, heading=heading))))
        return self._aggregate("goto", results)

    def _add_drone(self, cmd: Command) -> CommandResult:
        count = int(_number(cmd.params, "count", default=1, minimum=1, maximum=100))
        position = cmd.params.get("position")
        pos = _vector(position, "position") if position is not None else None
        if pos is not None and count > 1:
            raise CommandError("'position' can only be used with count = 1")
        added = []
        for _ in range(count):
            try:
                added.append(self._engine.swarm.add_drone(pos))
            except ValueError as exc:
                if not added:
                    return CommandResult.fail(str(exc))
                break
        return CommandResult(True, f"added {', '.join(d.name for d in added)}", data={"ids": [d.id for d in added]})

    def _remove_drone(self, cmd: Command) -> CommandResult:
        if not cmd.drone_ids:
            raise CommandError("remove_drone requires explicit drone_ids")
        removed = [self._engine.swarm.remove_drone(i).name for i in cmd.drone_ids]
        return CommandResult.ok(f"removed {', '.join(removed)}", ids=list(cmd.drone_ids))

    def _set_wind(self, cmd: Command) -> CommandResult:
        p = cmd.params
        enabled = p.get("enabled")
        if enabled is not None and not isinstance(enabled, bool):
            raise CommandError("parameter 'enabled' must be true/false")
        self._engine.environment.wind.set_wind(
            speed=_number(p, "speed", minimum=0.0, maximum=60.0),
            direction=_number(p, "direction"),
            gust_strength=_number(p, "gust_strength", minimum=0.0, maximum=30.0),
            enabled=enabled,
        )
        w = self._engine.environment.wind
        return CommandResult.ok(f"wind {w.speed:.1f} m/s from {w.direction:.0f} deg"
                                + ("" if w.enabled else " (disabled)"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
