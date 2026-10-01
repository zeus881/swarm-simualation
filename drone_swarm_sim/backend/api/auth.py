"""Login / logout, the current user and the audit log (Stage 5)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..security import AuthError, Principal, SecurityManager

router = APIRouter(prefix="/api", tags=["security"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., max_length=256)


def _security(request: Request) -> SecurityManager:
    return request.app.state.security


def client_label(request: Request) -> str:
    return request.client.host if request.client else ""


@router.get("/auth/config")
async def auth_config(request: Request) -> dict[str, Any]:
    """Public: whether the GCS asks for a login."""
    return {"enabled": _security(request).enabled}


@router.post("/auth/login")
def login(request: Request, body: LoginRequest) -> dict[str, Any]:
    """Credentials -> bearer token. Sync handler: the PBKDF2 check runs in the thread pool."""
    try:
        token, principal = _security(request).login(body.username, body.password, client_label(request))
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from None
    return {"token": token, "user": principal.to_dict()}


@router.get("/auth/me")
async def me(request: Request) -> dict[str, Any]:
    p: Principal = request.state.principal
    return {"enabled": _security(request).enabled, "user": p.to_dict()}


@router.post("/auth/logout")
async def logout(request: Request) -> dict[str, Any]:
    _security(request).logout(request.state.principal, client_label(request))
    return {"success": True}


@router.get("/audit")
async def audit(request: Request, limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
    """Most recent audit entries (who / what / when), oldest first."""
    return _security(request).audit.recent(limit)
