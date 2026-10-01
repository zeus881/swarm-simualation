"""Wire protocol for the ``/ws/telemetry`` WebSocket (docs/ARCHITECTURE.md §7.2).

Server -> client: ``telemetry``, ``command_result``, ``simulation_result``, ``pong``, ``error``.
Client -> server: ``command``, ``simulation``, ``ping``.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1


def _default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats (not valid JSON) with null."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def encode(message: dict[str, Any]) -> str:
    try:
        return json.dumps(message, separators=(",", ":"), default=_default, allow_nan=False)
    except ValueError:
        return json.dumps(_sanitize(message), separators=(",", ":"), default=_default)


def telemetry_message(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {"type": "telemetry", "protocol": PROTOCOL_VERSION, "server_time": time.time(), **snapshot}


def result_message(kind: str, request_id: Any, success: bool, message: str, **extra: Any) -> dict[str, Any]:
    return {"type": kind, "id": request_id, "success": success, "message": message, **extra}


def error_message(message: str, request_id: Any = None) -> dict[str, Any]:
    return {"type": "error", "id": request_id, "message": message}


def pong_message(request_id: Any, client_time: Any) -> dict[str, Any]:
    return {"type": "pong", "id": request_id, "client_time": client_time, "server_time": time.time()}
