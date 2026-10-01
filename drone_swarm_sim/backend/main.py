"""FastAPI application: REST API, telemetry WebSocket and the static GCS frontend.

Run with ``python -m simulation`` (preferred) or
``uvicorn backend.main:app --host 127.0.0.1 --port 8000``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import mimetypes
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from simulation.config import SimConfig, load_config
from simulation.engine import SimulationEngine
from simulation.replay import ReplayStore
from simulation.runner import SimulationRunner

from .api.auth import client_label
from .api.auth import router as auth_router
from .api.missions import router as missions_router
from .api.replay import router as replay_router
from .api.routes import router as api_router
from .security import READ_METHODS, AuditLog, AuthError, SecurityManager, bearer_token
from .websocket.hub import TelemetryHub
from .websocket.routes import router as ws_router

log = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Windows can map .js to text/plain through the registry, which makes browsers
# refuse ES modules. Pin the correct types explicitly.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


# No login needed: the login itself, whether a login is required, and the liveness probe.
PUBLIC_PATHS = frozenset({"/api/health", "/api/auth/login", "/api/auth/config"})
# Any logged-in role may call these even though they are POSTs.
ANY_ROLE_PATHS = frozenset({"/api/auth/logout"})
# These write their own, more detailed audit entries.
SELF_AUDITED_PREFIXES = ("/api/commands", "/api/simulation/", "/api/auth/")


def create_app(config: SimConfig | None = None, *, record: bool | None = None) -> FastAPI:
    config = config or load_config()
    # record=False (tests, benchmarks) keeps the audit trail in memory instead of logs/audit.log.
    audit_path = None if record is False else config.logging.path / config.security.audit_log
    audit = AuditLog(audit_path)
    security = SecurityManager(config.security, audit)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = SimulationEngine(config, record=record)
        runner = SimulationRunner(engine)
        hub = TelemetryHub(runner, config.telemetry.rate_hz, security)
        app.state.config = config
        app.state.engine = engine
        app.state.runner = runner
        app.state.hub = hub
        app.state.replay = ReplayStore(config.logging.path, config.replay.cache_runs)
        runner.start_thread()
        if config.simulation.auto_start:
            runner.resume()
        broadcaster = asyncio.create_task(hub.run(), name="telemetry-broadcast")
        log.info("Swarm simulation backend ready (%d drones)", config.simulation.drone_count)
        try:
            yield
        finally:
            broadcaster.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await broadcaster
            runner.shutdown()
            engine.close()
            audit.close()
            log.info("Swarm simulation backend stopped")

    app = FastAPI(
        title="Drone Swarm Simulation",
        version="0.1.0",
        description="Multi-UAV swarm simulation platform - REST control and WebSocket telemetry.",
        lifespan=lifespan,
    )
    app.state.security = security

    @app.middleware("http")
    async def access_control(request: Request, call_next):
        """Role check for every /api request (reads: any role; changes: operator) + audit of changes."""
        path = request.url.path
        if not path.startswith("/api/") or path in PUBLIC_PATHS:
            return await call_next(request)
        mutating = request.method not in READ_METHODS
        try:
            principal = security.verify(bearer_token(request.headers, request.query_params))
        except AuthError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=401)
        if mutating and path not in ANY_ROLE_PATHS and not principal.can_control:
            audit.record(principal, f"{request.method} {path}", success=False, result="denied: observer (read-only)",
                         client=client_label(request), channel="rest")
            return JSONResponse({"detail": f"{principal.username} is an observer (read-only)"}, status_code=403)
        request.state.principal = principal
        response = await call_next(request)
        if mutating and not path.startswith(SELF_AUDITED_PREFIXES):
            audit.record(principal, f"{request.method} {path}", success=response.status_code < 400,
                         result=f"HTTP {response.status_code}", client=client_label(request), channel="rest")
        return response

    app.include_router(auth_router)
    app.include_router(api_router)
    app.include_router(missions_router)
    app.include_router(replay_router)
    app.include_router(ws_router)
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    else:
        log.warning("Frontend directory %s not found - serving API only", FRONTEND_DIR)
    return app


def serve(config: SimConfig, *, open_browser: bool = False) -> None:
    import uvicorn

    host, port = config.server.host, config.server.port
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
    print(f"\n  Swarm Control Center  ->  {url}\n  REST API docs         ->  {url}/docs\n  Press Ctrl+C to stop.\n")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(config), host=host, port=port, log_config=None, log_level="warning", ws="auto")


def _app_from_env() -> FastAPI:
    """App object for ``uvicorn backend.main:app`` (config from $SWARM_CONFIG or defaults)."""
    return create_app()


def __getattr__(name: str):
    # Lazily build ``app`` so importing this module (e.g. in tests) does not load config.
    if name == "app":
        return _app_from_env()
    raise AttributeError(name)
