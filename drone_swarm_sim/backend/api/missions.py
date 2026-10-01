"""Mission REST API: library files, survey generation, QGC WPL 110 import/export, active mission paths.

State-changing mission operations (start, pause, abort, geofence) are *commands* and go through
``POST /api/commands`` / the WebSocket so they are validated and logged like every other command.
These endpoints are pure functions or file I/O.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from missions.model import Mission, MissionError, slugify
from missions.qgc import export_qgc_wpl, import_qgc_wpl
from missions.storage import MissionStore
from missions.survey import generate_survey
from missions.geometry import PolygonError

from ..websocket.hub import COMMAND_TIMEOUT_S

router = APIRouter(prefix="/api/missions", tags=["missions"])


class SurveyRequest(BaseModel):
    polygon: list[list[float]] = Field(..., description="[[x, y], ...] local ENU metres")
    altitude: float = Field(..., gt=0)
    speed: float | None = Field(None, gt=0)
    drones: int = Field(1, ge=1, le=200)
    line_spacing: float | None = Field(None, gt=0)
    overlap: float | None = Field(None, ge=0, lt=100)
    angle: float | None = None
    finish: str | None = None
    name: str = "survey"


class SaveRequest(BaseModel):
    mission: dict[str, Any]
    overwrite: bool = True


class ImportRequest(BaseModel):
    text: str = Field(..., max_length=2_000_000)
    name: str = "imported"


def _engine(request: Request):
    return request.app.state.engine


def _geo_to_enu(request: Request):
    geo = _engine(request).geo

    def convert(lat: float, lon: float) -> tuple[float, float]:
        e = geo.geodetic_to_enu(lat, lon, geo.origin.altitude)
        return float(e[0]), float(e[1])
    return convert


def _store(request: Request) -> MissionStore:
    return MissionStore(request.app.state.config.mission.path, _geo_to_enu(request))


def _parse(request: Request, data: dict[str, Any]) -> Mission:
    try:
        return Mission.from_dict(data, _geo_to_enu(request))
    except MissionError as exc:
        raise HTTPException(422, f"invalid mission: {exc}") from None


@router.get("/files")
async def list_files(request: Request) -> list[dict[str, Any]]:
    return _store(request).list()


@router.get("/files/{filename}")
async def load_file(request: Request, filename: str) -> dict[str, Any]:
    try:
        return _store(request).load(filename).to_dict()
    except FileNotFoundError:
        raise HTTPException(404, f"{filename} not found") from None
    except MissionError as exc:
        raise HTTPException(422, str(exc)) from None


@router.post("/files")
async def save_file(request: Request, body: SaveRequest) -> dict[str, Any]:
    try:
        filename, mission = _store(request).save(body.mission, overwrite=body.overwrite)
    except MissionError as exc:
        raise HTTPException(422, f"invalid mission: {exc}") from None
    return {"file": filename, "name": mission.name, "waypoints": mission.waypoint_count}


@router.delete("/files/{filename}")
async def delete_file(request: Request, filename: str) -> dict[str, Any]:
    try:
        _store(request).delete(filename)
    except FileNotFoundError:
        raise HTTPException(404, f"{filename} not found") from None
    except MissionError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"deleted": filename}


@router.post("/normalize")
async def normalize(request: Request, mission: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Schema-validate a mission (e.g. a JSON file the operator opened locally) and return it normalised."""
    return _parse(request, mission).to_dict()


@router.post("/survey")
async def survey(request: Request, body: SurveyRequest) -> dict[str, Any]:
    mc = request.app.state.config.mission
    try:
        result = generate_survey(body.polygon, altitude=body.altitude, speed=body.speed, drones=body.drones,
                                 line_spacing=body.line_spacing,
                                 overlap=body.overlap if body.overlap is not None or body.line_spacing else mc.survey_overlap,
                                 hfov_deg=mc.camera_hfov_deg, angle_deg=body.angle, finish=body.finish or mc.survey_finish)
    except (ValueError, PolygonError) as exc:
        raise HTTPException(422, str(exc)) from None
    mission = Mission(body.name, result.tracks,
                      survey={"polygon": body.polygon, "altitude": body.altitude, "speed": body.speed,
                              "drones": body.drones, **result.stats()})
    return {"mission": mission.to_dict(), "stats": result.stats()}


@router.post("/export/qgc", response_class=PlainTextResponse)
async def export_qgc(request: Request, mission: dict[str, Any] = Body(...), track: int = Query(0, ge=0)) -> PlainTextResponse:
    m = _parse(request, mission)
    engine = _engine(request)
    geo = engine.geo

    def enu_to_geo(x: float, y: float, z: float) -> tuple[float, float, float]:
        lat, lon, alt = geo.enu_to_geodetic([x, y, z])
        return float(lat), float(lon), float(alt)

    home = engine.environment.home_position
    try:
        text = export_qgc_wpl(m, enu_to_geo, (float(home[0]), float(home[1]), float(home[2])),
                              geo.origin.altitude + float(home[2]), track=track,
                              default_loiter_radius=request.app.state.config.mission.loiter_radius)
    except MissionError as exc:
        raise HTTPException(422, str(exc)) from None
    suffix = f"_track{track + 1}" if len(m.tracks) > 1 else ""
    return PlainTextResponse(text, headers={
        "Content-Disposition": f'attachment; filename="{slugify(m.name)}{suffix}.waypoints"'})


@router.post("/import/qgc")
async def import_qgc(request: Request, body: ImportRequest) -> dict[str, Any]:
    engine = _engine(request)
    home_amsl = engine.geo.origin.altitude + float(engine.environment.home_position[2])
    try:
        mission, warnings = import_qgc_wpl(body.text, _geo_to_enu(request), home_amsl, name=body.name)
    except MissionError as exc:
        raise HTTPException(422, str(exc)) from None
    return {"mission": mission.to_dict(), "warnings": warnings}


@router.get("/active")
async def active(request: Request) -> dict[str, Any]:
    future = request.app.state.runner.call(lambda eng: {"version": eng.missions.version,
                                                        "paths": eng.missions.active_paths()})
    return await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)
