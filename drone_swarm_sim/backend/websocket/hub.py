"""Telemetry hub: broadcasts snapshots to WebSocket clients and relays commands.

Each snapshot is JSON-encoded exactly once and offered to every client
through a size-1 "latest wins" slot. A slow client therefore drops frames
instead of accumulating latency or back-pressuring the simulation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from simulation.commands import Command, CommandError
from simulation.runner import SimulationRunner

from ..security import LOCAL_OPERATOR, AuthError, Principal, SecurityManager
from ..telemetry.messages import encode, error_message, pong_message, result_message, telemetry_message

log = logging.getLogger(__name__)

COMMAND_TIMEOUT_S = 5.0


AUTH_CLOSE_CODE = 4401          # WebSocket close code: login required / session expired


class _Client:
    def __init__(self, websocket: WebSocket, principal: Principal, token: str | None) -> None:
        self.ws = websocket
        self.principal = principal
        self.token = token
        self.label = websocket.client.host if websocket.client else ""
        self._latest: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self._send_lock = asyncio.Lock()
        self.dropped = 0

    def offer(self, text: str) -> None:
        if self._latest.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._latest.get_nowait()
                self.dropped += 1
        self._latest.put_nowait(text)

    async def send(self, text: str) -> None:
        async with self._send_lock:
            await self.ws.send_text(text)

    async def pump(self) -> None:
        while True:
            await self.send(await self._latest.get())


class TelemetryHub:
    def __init__(self, runner: SimulationRunner, rate_hz: float, security: SecurityManager | None = None) -> None:
        self.runner = runner
        self.security = security
        self.period = 1.0 / rate_hz
        self._clients: set[_Client] = set()
        self._last_seq = -1
        self._last_text: str | None = None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def _current_text(self) -> str | None:
        seq, snapshot = self.runner.latest_snapshot()
        if not snapshot:
            return None
        if seq != self._last_seq:
            self._last_seq = seq
            self._last_text = encode(telemetry_message(snapshot))
        return self._last_text

    async def run(self) -> None:
        """Broadcast loop (one task per application)."""
        while True:
            try:
                seq, _ = self.runner.latest_snapshot()
                if self._clients and seq != self._last_seq:
                    text = self._current_text()
                    if text is not None:
                        for client in list(self._clients):
                            client.offer(text)
            except Exception:
                log.exception("Telemetry broadcast failed")
            await asyncio.sleep(self.period / 2)   # sample at 2x rate so frames are never missed

    def _authenticate(self, token: str | None) -> Principal:
        return self.security.verify(token) if self.security is not None else LOCAL_OPERATOR

    async def handle(self, websocket: WebSocket) -> None:
        await websocket.accept()
        token = websocket.query_params.get("token")
        try:
            principal = self._authenticate(token)
        except AuthError as exc:
            # Accept first, then close with a code the browser can read (a rejected handshake is just "1006").
            await websocket.send_text(encode({"type": "auth_error", "message": str(exc)}))
            await websocket.close(code=AUTH_CLOSE_CODE)
            return
        client = _Client(websocket, principal, token)
        self._clients.add(client)
        log.info("Telemetry client connected (%d total)", len(self._clients))
        text = self._current_text()
        if text is not None:
            client.offer(text)
        pump = asyncio.create_task(client.pump())
        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except (ValueError, TypeError):
                    await client.send(encode(error_message("invalid JSON")))
                    continue
                await self._dispatch(client, message)
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:   # connection closed while sending
            log.debug("WebSocket closed: %s", exc)
        finally:
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump
            self._clients.discard(client)
            log.info("Telemetry client disconnected (%d total)", len(self._clients))

    async def _dispatch(self, client: _Client, message: Any) -> None:
        if not isinstance(message, dict):
            await client.send(encode(error_message("message must be an object")))
            return
        kind, request_id = message.get("type"), message.get("id")
        if kind == "ping":
            reply = pong_message(request_id, message.get("client_time"))
        elif kind in ("command", "simulation"):
            reply_type = f"{kind}_result"
            try:
                client.principal = self._authenticate(client.token)        # the session may have expired
            except AuthError as exc:
                await client.send(encode({"type": "auth_error", "message": str(exc)}))
                await client.ws.close(code=AUTH_CLOSE_CODE)
                return
            if not client.principal.can_control:
                cmd = message.get("command")
                what = (cmd.get("type") if isinstance(cmd, dict) else None) if kind == "command" else message.get("action")
                self._audit(client, f"{kind} {what}", success=False, result="denied: observer (read-only)")
                reply = result_message(reply_type, request_id, False,
                                       f"{client.principal.username} is an observer (read-only)")
            elif kind == "command":
                reply = await self._command(client, request_id, message.get("command"))
            else:
                reply = await self._simulation(client, request_id, message.get("action"))
        else:
            reply = error_message(f"unknown message type {kind!r}", request_id)
        await client.send(encode(reply))

    def _audit(self, client: _Client, action: str, **fields: Any) -> None:
        if self.security is not None:
            self.security.audit.record(client.principal, action, client=client.label, channel="ws", **fields)

    async def _command(self, client: _Client, request_id: Any, payload: Any) -> dict[str, Any]:
        try:
            command = Command.from_dict(payload if isinstance(payload, dict) else {})
        except CommandError as exc:
            self._audit(client, "command", params=payload, success=False, result=str(exc))
            return result_message("command_result", request_id, False, str(exc))
        command.issued_by = client.principal.username
        audit = {"target": command.drone_ids, "params": command.params}
        try:
            future = self.runner.submit_command(command)
            result = await asyncio.wait_for(asyncio.wrap_future(future), COMMAND_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._audit(client, f"command {command.type}", success=False, result="timed out", **audit)
            return result_message("command_result", request_id, False, "command timed out")
        except RuntimeError as exc:
            self._audit(client, f"command {command.type}", success=False, result=str(exc), **audit)
            return result_message("command_result", request_id, False, str(exc))
        self._audit(client, f"command {command.type}", success=result.success, result=result.message, **audit)
        return result_message("command_result", request_id, result.success, result.message,
                              details=result.details, data=result.data, command=command.type)

    async def _simulation(self, client: _Client, request_id: Any, action: Any) -> dict[str, Any]:
        ok, message = await apply_simulation_action(self.runner, action)
        self._audit(client, f"simulation {action}", success=ok, result=message)
        return result_message("simulation_result", request_id, ok, message, action=action)


async def apply_simulation_action(runner: SimulationRunner, action: Any) -> tuple[bool, str]:
    """Shared by REST and WebSocket: start | pause | reset."""
    if action == "start":
        runner.resume()
        return True, "simulation running"
    if action == "pause":
        runner.pause()
        return True, "simulation paused"
    if action == "reset":
        try:
            await asyncio.wait_for(asyncio.wrap_future(runner.reset()), COMMAND_TIMEOUT_S)
        except Exception as exc:
            log.exception("Reset failed")
            return False, f"reset failed: {exc}"
        return True, "simulation reset"
    return False, f"unknown simulation action {action!r} (start | pause | reset)"
