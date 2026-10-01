"""Role-based access and the command audit log (Stage 5).

* Two roles: ``operator`` (full control) and ``observer`` (read-only: telemetry, missions, replay, reports).
* Passwords are stored as ``pbkdf2_sha256$<iterations>$<salt>$<hash>`` (standard library only, offline).
* A successful login returns a bearer token ``<payload>.<signature>``: the payload (user, role, expiry,
  token id) is HMAC-SHA256 signed with a per-process secret, so tokens die with the server unless
  ``$SWARM_SECRET`` pins the secret. Logout revokes the token id.
* REST: ``Authorization: Bearer <token>`` (or ``?token=`` for download links); WebSocket: ``?token=``.
  Reads need any role, anything that changes state needs ``operator``.
* Audit log: ``logs/audit.log``, one JSON object per line - who (user, role, client), what (action,
  target, parameters, result) and when (UTC ISO time) for every command, simulation action, mission
  change, login and denied request.

Command line: ``python -m backend.security hash-password [password]``.
"""

from __future__ import annotations

import base64
import getpass
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simulation.config import SecurityConfig

log = logging.getLogger(__name__)

PBKDF2_ITERATIONS = 200_000
SECRET_ENV_VAR = "SWARM_SECRET"
READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
# Hashes of the shipped demo accounts (operator/operator, observer/observer): the server warns while they are in use.
DEFAULT_PASSWORDS = {"operator": "operator", "observer": "observer"}


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS, salt: bytes | None = None) -> str:
    """``pbkdf2_sha256$<iterations>$<salt>$<hash>`` for ``password``."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, encoded: str) -> bool:
    """Constant-time check of ``password`` against an encoded hash (False for malformed hashes)."""
    try:
        scheme, iterations, salt, digest = encoded.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _unb64(salt), int(iterations))
        return hmac.compare_digest(actual, _unb64(digest))
    except (ValueError, TypeError):
        return False


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller."""

    username: str
    role: str
    token_id: str = ""
    expires: float = 0.0

    @property
    def can_control(self) -> bool:
        return self.role == "operator"

    def to_dict(self) -> dict[str, Any]:
        return {"username": self.username, "role": self.role, "can_control": self.can_control,
                "expires": self.expires}


LOCAL_OPERATOR = Principal("local", "operator")


class AuthError(Exception):
    """Authentication failed (401) - or, with ``forbidden``, the role is not allowed (403)."""

    def __init__(self, message: str, *, forbidden: bool = False) -> None:
        super().__init__(message)
        self.forbidden = forbidden


class AuditLog:
    """Append-only JSON Lines audit trail; the tail is kept in memory for ``GET /api/audit``.

    ``path=None`` keeps the trail in memory only (unrecorded test servers).
    """

    def __init__(self, path: Path | None, tail: int = 500) -> None:
        self.path = path
        self._tail: deque[dict[str, Any]] = deque(maxlen=tail)
        self._lock = threading.Lock()
        self._file = None
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8", buffering=1)      # line buffered: one write per entry
        except OSError:
            log.exception("Cannot open the audit log %s - audit entries are kept in memory only", path)

    def record(self, principal: Principal | None, action: str, *, target: Any = None, params: Any = None,
               success: bool | None = None, result: str = "", client: str = "", channel: str = "") -> dict[str, Any]:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "user": principal.username if principal else None,
            "role": principal.role if principal else None,
            "action": action,
            "target": target,
            "params": params,
            "success": success,
            "result": result[:500],
            "client": client,
            "channel": channel,
        }
        with self._lock:
            self._tail.append(entry)
            if self._file is not None:
                try:
                    self._file.write(json.dumps(entry, default=str) + "\n")
                except OSError:
                    log.exception("Audit log write failed")
        return entry

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._tail)[-limit:]

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class SecurityManager:
    """Users, login, token signing / verification and role checks."""

    def __init__(self, config: SecurityConfig, audit: AuditLog, *, secret: bytes | None = None) -> None:
        self.config = config
        self.audit = audit
        env = os.environ.get(SECRET_ENV_VAR)
        self._secret = secret or (env.encode("utf-8") if env else secrets.token_bytes(32))
        self._users = {u["username"]: u for u in config.users}
        self._revoked: dict[str, float] = {}          # token id -> expiry (pruned on use)
        self._failures: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        if config.enabled:
            weak = [u["username"] for u in config.users
                    if u["username"] in DEFAULT_PASSWORDS and verify_password(DEFAULT_PASSWORDS[u["username"]],
                                                                             u["password_hash"])]
            if weak:
                log.warning("Security: account(s) %s still use the shipped default password - change them "
                            "(python -m backend.security hash-password)", ", ".join(weak))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # ------------------------------------------------------------------ login / tokens
    def login(self, username: str, password: str, client: str = "") -> tuple[str, Principal]:
        """Check credentials; returns ``(token, principal)`` or raises :class:`AuthError`."""
        if not self.enabled:
            return "", LOCAL_OPERATOR
        now = time.monotonic()
        with self._lock:
            recent = self._failures.setdefault(username, deque(maxlen=self.config.max_login_failures))
            while recent and now - recent[0] > self.config.lockout_s:
                recent.popleft()
            locked = len(recent) >= self.config.max_login_failures
        if locked:
            self.audit.record(Principal(username, "?"), "login", success=False, result="locked out", client=client)
            raise AuthError(f"too many failed logins - try again in {self.config.lockout_s:.0f} s")
        user = self._users.get(username)
        # Always run the hash (also for unknown users) so the response time does not reveal valid names.
        ok = verify_password(password, user["password_hash"] if user else hash_password("x", iterations=1000))
        if not user or not ok:
            with self._lock:
                self._failures[username].append(now)
            self.audit.record(Principal(username, "?"), "login", success=False, result="bad credentials", client=client)
            raise AuthError("invalid username or password")
        with self._lock:
            self._failures.pop(username, None)
        principal = Principal(username, user["role"], secrets.token_hex(8), time.time() + self.config.token_ttl_s)
        self.audit.record(principal, "login", success=True, result=f"role {principal.role}", client=client)
        return self._sign(principal), principal

    def logout(self, principal: Principal, client: str = "") -> None:
        if principal.token_id:
            with self._lock:
                self._revoked[principal.token_id] = principal.expires
        self.audit.record(principal, "logout", success=True, client=client)

    def _sign(self, p: Principal) -> str:
        payload = _b64(json.dumps({"u": p.username, "r": p.role, "id": p.token_id, "exp": p.expires},
                                  separators=(",", ":")).encode("utf-8"))
        sig = _b64(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
        return f"{payload}.{sig}"

    def verify(self, token: str | None) -> Principal:
        """Principal for a bearer token; raises :class:`AuthError`. Security off -> the local operator."""
        if not self.enabled:
            return LOCAL_OPERATOR
        if not token:
            raise AuthError("login required")
        try:
            payload, sig = token.split(".")
            expected = _b64(hmac.new(self._secret, payload.encode("ascii"), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, expected):
                raise ValueError("bad signature")
            data = json.loads(_unb64(payload))
            p = Principal(str(data["u"]), str(data["r"]), str(data["id"]), float(data["exp"]))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            raise AuthError("invalid token") from None
        now = time.time()
        if p.expires < now:
            raise AuthError("session expired")
        with self._lock:
            for tid in [t for t, exp in self._revoked.items() if exp < now]:
                del self._revoked[tid]
            if p.token_id in self._revoked:
                raise AuthError("logged out")
        user = self._users.get(p.username)
        if user is None or user["role"] != p.role:          # account removed or role changed since login
            raise AuthError("account changed - log in again")
        return p

    @staticmethod
    def require_operator(p: Principal) -> None:
        if not p.can_control:
            raise AuthError(f"{p.username} is an observer (read-only)", forbidden=True)


def bearer_token(headers: Any, query: Any) -> str | None:
    """Token from ``Authorization: Bearer`` or the ``token`` query parameter."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return query.get("token") or None


def _main(argv: list[str]) -> int:
    if len(argv) < 1 or argv[0] != "hash-password":
        print("usage: python -m backend.security hash-password [password]")
        return 2
    password = argv[1] if len(argv) > 1 else getpass.getpass("Password: ")
    if len(password) < 8:
        print("warning: passwords shorter than 8 characters are weak", file=sys.stderr)
    print(hash_password(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
