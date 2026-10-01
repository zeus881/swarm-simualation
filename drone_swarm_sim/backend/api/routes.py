"""REST API (docs/ARCHITECTURE.md §7.1)."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from simulation.commands import Command, CommandError
from simulation.runner import SimulationRunner

from ..websocket.hub import COMMAND_TIMEOUT_S, apply_simulation_action
from .auth import client_label

router = APIRouter(prefix="/api", tags=["simulation"])


class CommandRequest(BaseModel):
    type: str = Field(..., examples=["takeoff"])
    drone_ids: list[int] | None = Field(None, description="null = whole swarm")
    params: dict[str, Any] = Field(default_factory=dict, examples=[{"altitude": 25}])


class CommandResponse(BaseModel):
    success: bool
    message: str
    details: dict[str, str] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)


class SimulationResponse(BaseModel):
    success: bool
    message: str
    state: str


def _runner(request: Request) -> SimulationRunner:
    return request.app.state.runner


def _snapshot(request: Request) -> dict[str, Any]:
    _, snap = _runner(request).latest_snapshot()
    if not snap:
        raise HTTPException(503, "simulation not ready")
    return snap


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    return {"status": "ok", "state": str(_runner(request).state)}


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    snap = _snapshot(request)
    return {
        "state": snap.get("state"),
        "run_id": snap.get("run_id"),
        "sim_time": snap.get("sim_time"),
        "tick": snap.get("tick"),
        "stats": snap.get("stats"),
        "summary": snap.get("summary"),
        "wind": snap.get("wind"),
        "clients": request.app.state.hub.client_count,
    }


@router.get("/config")
async def config(request: Request) -> dict[str, Any]:
    return request.app.state.config.to_dict()


@router.get("/world")
async def world(request: Request) -> dict[str, Any]:
    return _snapshot(request)["world"]


@router.get("/world/scene")
async def world_scene(request: Request) -> dict[str, Any]:
    """Obstacles and a down-sampled terrain grid (clients fetch it when ``world.scene.version`` changes)."""
    future = _runner(request).call(lambda eng: eng.environment.scene())
    return await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)


@router.get("/drones")
async def drones(request: Request) -> list[dict[str, Any]]:
    return _snapshot(request)["drones"]


@router.get("/drones/{drone_id}")
async def drone(request: Request, drone_id: int) -> dict[str, Any]:
    for d in _snapshot(request)["drones"]:
        if d["drone_id"] == drone_id:
            return d
    raise HTTPException(404, f"drone {drone_id} not found")


@router.get("/events")
async def events(request: Request, limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    future = _runner(request).call(lambda eng: [e.to_dict() for e in eng.events.recent(0, limit)])
    return await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)


@router.get("/commands")
async def list_commands(request: Request) -> dict[str, str]:
    future = _runner(request).call(lambda eng: eng.commands.available())
    return await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)


def _audit(request: Request, action: str, **fields: Any) -> None:
    request.app.state.security.audit.record(request.state.principal, action, client=client_label(request),
                                            channel="rest", **fields)


@router.post("/commands", response_model=CommandResponse)
async def execute_command(request: Request, body: CommandRequest) -> CommandResponse:
    try:
        command = Command.from_dict(body.model_dump())
    except CommandError as exc:
        raise HTTPException(422, str(exc)) from None
    command.issued_by = request.state.principal.username
    future = _runner(request).submit_command(command)
    try:
        result = await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)
    except asyncio.TimeoutError:
        _audit(request, f"command {command.type}", target=command.drone_ids, params=command.params,
               success=False, result="timed out")
        raise HTTPException(504, "command timed out") from None
    _audit(request, f"command {command.type}", target=command.drone_ids, params=command.params,
           success=result.success, result=result.message)
    return CommandResponse(**result.to_dict())


@router.post("/simulation/{action}", response_model=SimulationResponse)
async def simulation_action(request: Request, action: Literal["start", "pause", "reset"]) -> SimulationResponse:
    runner = _runner(request)
    ok, message = await apply_simulation_action(runner, action)
    _audit(request, f"simulation {action}", success=ok, result=message)
    if not ok:
        raise HTTPException(500, message)
    return SimulationResponse(success=ok, message=message, state=str(runner.state))
