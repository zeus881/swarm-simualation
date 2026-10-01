"""Replay and post-mission report endpoints (Stage 5).

``def`` (not ``async def``) handlers: FastAPI runs them in its thread pool, so parsing a long
recording never blocks the event loop or the simulation thread.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response

from simulation.replay import ReplayError, ReplayStore
from simulation.report import build_report_data, render_html, render_pdf

router = APIRouter(prefix="/api/replay", tags=["replay"])


def _store(request: Request) -> ReplayStore:
    return request.app.state.replay


def _run(request: Request, run_id: str):
    try:
        return _store(request).get(run_id)
    except ReplayError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/runs")
def runs(request: Request) -> list[dict[str, Any]]:
    """Recorded runs, newest first; ``current`` marks the run being recorded now."""
    current = getattr(request.app.state.engine, "run_id", None)
    return [dict(r, current=r["run_id"] == current) for r in _store(request).runs()]


@router.get("/runs/{run_id}")
def run_meta(request: Request, run_id: str) -> dict[str, Any]:
    """Duration, drones, world, geofence, event markers and the frame column layout."""
    cfg = request.app.state.config.replay
    meta = _run(request, run_id).meta(cfg.max_events)
    meta["chunk_s"] = cfg.chunk_s
    return meta


@router.get("/runs/{run_id}/frames")
def run_frames(request: Request, run_id: str, start: float = Query(0.0, ge=0.0),
               end: float | None = Query(None, ge=0.0)) -> dict[str, Any]:
    """Frames in ``[start, end]`` (at most ``replay.chunk_s`` seconds per request)."""
    chunk = request.app.state.config.replay.chunk_s
    end = start + chunk if end is None else min(end, start + chunk)
    try:
        return _run(request, run_id).frames(start, end)
    except ReplayError as exc:
        raise HTTPException(422, str(exc)) from None


@router.get("/runs/{run_id}/report")
def run_report(request: Request, run_id: str, format: str = Query("html", pattern="^(html|pdf)$")) -> Response:
    """One-click post-mission report as a self-contained HTML page or a PDF download."""
    cfg = request.app.state.config.report
    data = build_report_data(_run(request, run_id), title=cfg.title, max_track_points=cfg.max_track_points,
                             max_timeline_events=cfg.max_timeline_events)
    name = f"gandiv-report-{run_id}"
    if format == "pdf":
        return Response(render_pdf(data, cfg.logo_path), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'})
    return HTMLResponse(render_html(data, cfg.logo_path),
                        headers={"Content-Disposition": f'inline; filename="{name}.html"'})
