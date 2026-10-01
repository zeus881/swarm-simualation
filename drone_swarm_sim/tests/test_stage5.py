"""Stage 5: replay, post-mission report, hardware adapters, role-based access + audit, recorder memory."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.security import AuditLog, AuthError, SecurityManager, hash_password, verify_password
from integration import create_remote_drone
from integration.remote import RemoteLink
from simulation.config import ConfigError, SecurityConfig
from simulation.engine import SimulationEngine
from simulation.recorder import read_json_array
from simulation.replay import ReplayError, ReplayStore, RunLog, list_runs
from simulation.report import build_report_data, render_html, render_pdf
from simulation.types import CommStatus, FlightMode

from .conftest import make_config

OP_HASH = hash_password("op-secret", iterations=1000)
OBS_HASH = hash_password("obs-secret", iterations=1000)
USERS = [{"username": "alice", "role": "operator", "password_hash": OP_HASH},
         {"username": "bob", "role": "observer", "password_hash": OBS_HASH}]


# ============================================================================ recorded run fixture

@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """A short recorded run: takeoff, formation, move, land (6 drones)."""
    log_dir = tmp_path_factory.mktemp("logs")
    cfg = make_config(simulation__drone_count=6, logging__record_telemetry=True, logging__telemetry_record_rate_hz=5.0)
    eng = SimulationEngine(cfg, log_dir=log_dir)
    eng.execute({"type": "takeoff", "params": {"altitude": 15}})
    eng.run_for(8)
    eng.execute({"type": "set_formation", "params": {"shape": "line", "spacing": 10}})
    eng.run_for(10)
    eng.execute({"type": "swarm_goto", "params": {"position": [40, 30, 20]}})
    eng.run_for(12)
    eng.execute({"type": "land"})
    eng.run_for(6)
    run_dir = eng.recorder.directory
    eng.close()
    return run_dir


# ============================================================================ recorder

def test_event_files_stream_and_stay_readable(tmp_path):
    cfg = make_config(logging__record_telemetry=True)
    eng = SimulationEngine(cfg, log_dir=tmp_path)
    run_dir = eng.recorder.directory
    eng.execute({"type": "takeoff"})
    eng.run_for(3)
    eng.recorder.events.flush(force=True)                  # the batching timer is wall-clock; run_for is faster
    partial = (run_dir / "mission_events.json").read_text()
    with pytest.raises(json.JSONDecodeError):
        json.loads(partial)                                # no closing bracket while recording...
    assert any(e["kind"] == "takeoff" for e in read_json_array(run_dir / "mission_events.json"))   # ...but readable
    assert eng.recorder.events.pending == 0                # nothing accumulates in memory between flushes
    eng.close()
    events = json.loads((run_dir / "mission_events.json").read_text())      # valid JSON after close
    assert events and all("seq" in e for e in events)
    world = json.loads((run_dir / "world.json").read_text())
    assert world["run_id"] == run_dir.name and len(world["drones"]) == 3 and world["world"]["bounds"]["max"]["x"] > 0


def test_recorder_memory_is_bounded_over_a_long_run(tmp_path):
    """Hundreds of events and many telemetry rows: nothing grows in memory (the 30-min soak in fast forward)."""
    cfg = make_config(logging__record_telemetry=True, simulation__drone_count=4)
    eng = SimulationEngine(cfg, log_dir=tmp_path)
    eng.execute({"type": "takeoff", "params": {"altitude": 10}})
    for k in range(40):
        eng.execute({"type": "set_heading", "params": {"heading": 10 * k}})
        eng.run_for(3)
    rec = eng.recorder
    assert rec.events.pending < 50
    assert len(eng.events._history) <= cfg.telemetry.event_history
    assert rec.telemetry.rows_written >= 4 * 5 * 110
    eng.close()


def test_command_events_carry_the_issuing_user(engine):
    from simulation.commands import Command
    cmd = Command.from_dict({"type": "hover"})
    cmd.issued_by = "alice"
    engine.execute(cmd)
    ev = [e for e in engine.events.recent(0, 10) if e.kind == "hover"][-1]
    assert ev.data["user"] == "alice"
    # a client cannot set it through the payload
    assert Command.from_dict({"type": "hover", "issued_by": "mallory"}).issued_by is None


# ============================================================================ replay

def test_replay_runlog(recorded):
    run = RunLog(recorded)
    assert 35 < run.duration < 37
    assert [d["name"] for d in run.drones()] == [f"D{i:02d}" for i in range(1, 7)]
    meta = run.meta(500)
    assert meta["frames"] == len(run.frame_times) and abs(meta["record_rate_hz"] - 5.0) < 0.3
    assert any(e["category"] == "COMMAND" and e["kind"] == "set_formation" for e in meta["events"])
    fr = run.frames(10.0, 12.0)["frames"]
    assert len(fr) in (10, 11) and all(10 - 1e-6 <= f["t"] <= 12 + 1e-6 for f in fr)
    n, c = len(meta["columns"]["numeric"]), len(meta["columns"]["categorical"])
    row = fr[0]["rows"][0]
    assert len(row) == 1 + n + c + len(meta["columns"]["flags"])
    mode = meta["vocab"]["mode"][row[1 + n]]
    assert mode in ("HOVER", "FORMATION", "TAKEOFF")
    assert row[1 + n + c + 1] == 1                         # airborne flag
    sep = run.separation()
    assert np.nanmin(sep[:, 1]) > 1.0
    stats = run.per_drone_stats()
    assert all(s["distance_m"] > 20 for s in stats) and all(s["max_agl_m"] > 13 for s in stats)
    with pytest.raises(ReplayError):
        run.frames(5, 1)


def test_replay_store_listing_cache_and_path_safety(recorded):
    store = ReplayStore(recorded.parent, cache_runs=1)
    runs = store.runs()
    assert runs[0]["run_id"] == recorded.name and runs[0]["has_telemetry"] and runs[0]["drones"] == 6
    assert store.get(recorded.name) is store.get(recorded.name)          # cached
    for bad in ("../etc", "..", "a/b", "", "x" * 200, "nope"):
        with pytest.raises(ReplayError):
            store.get(bad)


def test_replay_reads_pre_stage5_recordings(tmp_path):
    run = tmp_path / "20250101-120000-000"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"simulation": {"drone_count": 1, "seed": 3}}))
    header = "timestamp,drone_id,x,y,z,lat,lon,alt,vx,vy,vz,speed,heading,roll,pitch,battery,mode,task,armed,health,communication,collision_state"
    rows = [f"{t:.1f},1,0,0,{t},47.0,8.0,500,0,0,1,1,0,0,0,{100 - t},TAKEOFF,TAKEOFF,1,OK,ONLINE,CLEAR" for t in range(5)]
    (run / "telemetry.csv").write_text("\n".join([header, *rows, "5.0,1,0"]) + "\n")      # + a half-written row
    (run / "mission_events.json").write_text('[\n{"seq": 1, "time": 0.0, "category": "COMMAND", "kind": "takeoff", "severity": "INFO", "message": "x"}')
    log = RunLog(run)
    assert log.duration == 4.0 and log.drones() == [{"id": 1, "name": "D01", "source": "sim"}]
    assert len(log.events()) == 1                          # unterminated event file
    assert list_runs(tmp_path)[0]["duration_s"] == 5.0     # tail line (partial row ignored by the parser)
    assert build_report_data(log)["drones"][0]["max_agl_m"] == 4.0


# ============================================================================ report

def test_report_html_and_pdf(recorded, tmp_path):
    data = build_report_data(RunLog(recorded), title="Test report")
    assert data["commands"] and data["min_separation"] > 1
    html = render_html(data, logo=None)
    for part in ("Test report", "Flight paths", "Battery", "Separation violations", "Alerts", "Event timeline",
                 "<svg", "GANDIV", recorded.name):
        assert part in html
    pdf = render_pdf(data, logo=None)
    assert pdf.startswith(b"%PDF-1.4") and pdf.rstrip().endswith(b"%%EOF")
    # the xref table points at the objects
    xref = int(pdf[pdf.rindex(b"startxref") + 10:].split()[0])
    assert pdf[xref:xref + 4] == b"xref"
    count = int(pdf[xref:].split(b"\n")[1].split()[1])
    offsets = [int(line.split()[0]) for line in pdf[xref:].split(b"\n")[3:2 + count]]
    for k, off in enumerate(offsets, start=1):
        assert pdf[off:off + len(f"{k} 0 obj")] == f"{k} 0 obj".encode()
    assert pdf.count(b"/Type /Page ") >= 2


def test_report_embeds_a_png_logo(recorded, tmp_path):
    import struct
    import zlib

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    raw = b"".join(b"\x00" + bytes([40, 80, 40] * 4) for _ in range(4))
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    logo = tmp_path / "logo.png"
    logo.write_bytes(png)
    data = build_report_data(RunLog(recorded))
    assert "data:image/png;base64," in render_html(data, logo)
    assert b"/Subtype /Image" in render_pdf(data, logo)


# ============================================================================ security

def test_password_hashing():
    h = hash_password("hunter22")
    assert h.startswith("pbkdf2_sha256$200000$") and verify_password("hunter22", h)
    assert not verify_password("hunter23", h) and not verify_password("x", "garbage")
    assert hash_password("a") != hash_password("a")        # salted


def _security(**kw) -> SecurityManager:
    cfg = SecurityConfig(enabled=True, users=USERS, **kw)
    cfg.validate()
    return SecurityManager(cfg, AuditLog(None), secret=b"k" * 32)


def test_tokens_roles_expiry_revocation_and_lockout():
    sec = _security(max_login_failures=3, lockout_s=60)
    token, p = sec.login("alice", "op-secret")
    assert p.can_control and sec.verify(token).username == "alice"
    obs_token, obs = sec.login("bob", "obs-secret")
    assert not obs.can_control
    with pytest.raises(AuthError) as e:
        sec.require_operator(sec.verify(obs_token))
    assert e.value.forbidden
    payload, sig = token.split(".")
    for bad in (None, "", "abc", f"{payload}.{sig[:-2]}xx", f"{obs_token.split('.')[0]}.{sig}"):
        with pytest.raises(AuthError):
            sec.verify(bad)
    sec.logout(sec.verify(token))
    with pytest.raises(AuthError):
        sec.verify(token)
    for _ in range(3):
        with pytest.raises(AuthError):
            sec.login("alice", "wrong")
    with pytest.raises(AuthError, match="too many"):
        sec.login("alice", "op-secret")                    # locked out even with the right password
    assert [a["success"] for a in sec.audit.recent()][-4:] == [False, False, False, False]
    # an expired token
    sec2 = _security(token_ttl_s=60)
    t2, _ = sec2.login("alice", "op-secret")
    sec2.config.token_ttl_s = 60
    import backend.security as bs
    real = bs.time.time
    bs.time.time = lambda: real() + 120
    try:
        with pytest.raises(AuthError, match="expired"):
            sec2.verify(t2)
    finally:
        bs.time.time = real


def test_security_config_validation():
    with pytest.raises(ConfigError):
        make_config(**{"security.enabled": True, "security.users": []})
    with pytest.raises(ConfigError):
        make_config(**{"security.users": [{"username": "x", "role": "admin", "password_hash": OP_HASH}]})
    with pytest.raises(ConfigError):
        make_config(**{"security.users": [{"username": "x", "role": "operator", "password_hash": "plain"}]})
    with pytest.raises(ConfigError):
        make_config(**{"security.users": USERS + [USERS[0]]})
    with pytest.raises(ConfigError):
        make_config(**{"replay.chunk_s": 0.5})
    cfg = make_config(**{"security.enabled": True, "security.users": USERS})
    assert cfg.security.enabled and cfg.report.logo_path.name == "gandiv-logo.png"


@pytest.fixture
def secure_client(recorded):
    cfg = make_config(**{"simulation.real_time_factor": 4.0, "security.enabled": True, "security.users": USERS,
                         "logging.directory": str(recorded.parent)})
    app = create_app(cfg, record=False)
    with TestClient(app) as c:
        yield c


def _login(client, user, password) -> dict[str, str]:
    r = client.post("/api/auth/login", json={"username": user, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['token']}"}


def test_rest_access_control_and_audit(secure_client):
    c = secure_client
    assert c.get("/api/auth/config").json() == {"enabled": True}
    assert c.get("/api/health").status_code == 200                     # public liveness probe
    assert c.get("/api/drones").status_code == 401
    assert c.post("/api/commands", json={"type": "takeoff"}).status_code == 401
    assert c.post("/api/auth/login", json={"username": "alice", "password": "nope"}).status_code == 401
    op, obs = _login(c, "alice", "op-secret"), _login(c, "bob", "obs-secret")
    assert c.get("/api/auth/me", headers=obs).json()["user"]["role"] == "observer"
    for _ in range(50):
        if c.get("/api/drones", headers=obs).status_code == 200:
            break
        time.sleep(0.05)
    assert len(c.get("/api/drones", headers=obs).json()) == 3          # observers read
    r = c.post("/api/commands", json={"type": "takeoff"}, headers=obs)
    assert r.status_code == 403 and "observer" in r.json()["detail"]
    assert c.post("/api/simulation/pause", headers=obs).status_code == 403
    assert c.post("/api/missions/geofence", json={}, headers=obs).status_code == 403
    r = c.post("/api/commands", json={"type": "takeoff", "params": {"altitude": 12}}, headers=op)
    assert r.status_code == 200 and r.json()["success"]
    token = op["Authorization"].split()[1]
    assert c.get(f"/api/drones?token={token}").status_code == 200       # download links
    audit = c.get("/api/audit?limit=50", headers=op).json()
    actions = [(a["user"], a["action"], a["success"]) for a in audit]
    assert ("alice", "command takeoff", True) in actions
    assert ("bob", "POST /api/commands", False) in actions
    assert ("alice", "login", False) in actions or any(a["action"] == "login" and not a["success"] for a in audit)
    ev = [e for e in c.get("/api/events?limit=200", headers=op).json() if e["kind"] == "takeoff" and e["category"] == "COMMAND"]
    assert ev and ev[-1]["data"]["user"] == "alice"
    assert c.post("/api/auth/logout", headers=obs).status_code == 200     # observers may log out
    assert c.get("/api/drones", headers=obs).status_code == 401


def test_websocket_auth_and_observer_read_only(secure_client):
    from starlette.websockets import WebSocketDisconnect
    c = secure_client
    with c.websocket_connect("/ws/telemetry") as ws:
        assert ws.receive_json()["type"] == "auth_error"
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_json()
        assert e.value.code == 4401
    obs = _login(c, "bob", "obs-secret")["Authorization"].split()[1]
    with c.websocket_connect(f"/ws/telemetry?token={obs}") as ws:
        assert ws.receive_json()["type"] == "telemetry"
        ws.send_json({"type": "command", "id": 1, "command": {"type": "emergency_stop"}})
        ws.send_json({"type": "simulation", "id": 2, "action": "reset"})
        ws.send_json({"type": "ping", "id": 3, "client_time": 1})
        replies = {}
        for _ in range(300):
            m = ws.receive_json()
            if m["type"] != "telemetry":
                replies[m["id"]] = m
            if len(replies) == 3:
                break
        assert not replies[1]["success"] and "read-only" in replies[1]["message"]
        assert not replies[2]["success"] and replies[3]["type"] == "pong"
    op = _login(c, "alice", "op-secret")["Authorization"].split()[1]
    with c.websocket_connect(f"/ws/telemetry?token={op}") as ws:
        ws.send_json({"type": "command", "id": 5, "command": {"type": "hover"}})
        for _ in range(300):
            m = ws.receive_json()
            if m.get("id") == 5:
                break
        assert m["type"] == "command_result"
    audit = c.get("/api/audit", headers={"Authorization": f"Bearer {op}"}).json()
    assert any(a["user"] == "bob" and a["channel"] == "ws" and a["success"] is False for a in audit)
    assert any(a["user"] == "alice" and a["action"] == "command hover" and a["channel"] == "ws" for a in audit)


def test_replay_and_report_endpoints(secure_client, recorded):
    c = secure_client
    obs = _login(c, "bob", "obs-secret")
    runs = c.get("/api/replay/runs", headers=obs).json()
    assert any(r["run_id"] == recorded.name for r in runs)
    meta = c.get(f"/api/replay/runs/{recorded.name}", headers=obs).json()
    assert meta["chunk_s"] == 20 and len(meta["drones"]) == 6
    fr = c.get(f"/api/replay/runs/{recorded.name}/frames?start=0&end=999", headers=obs).json()
    assert fr["end"] == 20 and fr["frames"][-1]["t"] <= 20                # capped to one chunk
    assert c.get("/api/replay/runs/..%2F..%2Fetc/frames", headers=obs).status_code == 404
    html = c.get(f"/api/replay/runs/{recorded.name}/report", headers=obs)
    assert html.status_code == 200 and "Post-mission report" in html.text
    token = obs["Authorization"].split()[1]
    pdf = c.get(f"/api/replay/runs/{recorded.name}/report?format=pdf&token={token}")
    assert pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
    assert "attachment" in pdf.headers["content-disposition"]


def test_security_disabled_means_local_operator():
    app = create_app(make_config(simulation__real_time_factor=4.0), record=False)
    with TestClient(app) as c:
        assert c.get("/api/auth/me").json()["user"]["username"] == "local"
        assert c.post("/api/commands", json={"type": "hover"}).status_code == 200
        assert c.get("/api/audit").json()[-1]["user"] == "local"


# ============================================================================ hardware adapter layer (fake link)

class FakeAutopilot(RemoteLink):
    """In-process vehicle: commands change its state instantly (enough to exercise RemoteDrone)."""

    kind = "mavlink"

    def __init__(self, geo, home_enu) -> None:
        super().__init__("fake://1")
        self.geo = geo
        self.sent: list[tuple] = []
        lat, lon, alt = geo.enu_to_geodetic(np.asarray(home_enu, dtype=float))
        self.ground = float(alt)
        self._update(connected=True, last_heartbeat=time.monotonic(), mode="STABILIZE", has_position=True,
                     lat=float(lat), lon=float(lon), alt_amsl=float(alt), gps_fix=3, satellites=12, hdop=0.8,
                     battery_pct=87.0, ready=True, home=(float(lat), float(lon), float(alt)))

    def beat(self) -> None:
        self._update(last_heartbeat=time.monotonic())

    def _send(self, command: str, args: tuple) -> None:
        self.sent.append((command, *args))
        s = self._state
        if command == "takeoff":
            self._update(armed=True, in_air=True, mode="GUIDED", alt_amsl=self.ground + args[0], rel_alt=args[0])
        elif command == "goto":
            self._update(lat=args[0], lon=args[1], alt_amsl=args[2], mode="GUIDED")
        elif command == "mode":
            self._update(mode=args[0])
        elif command == "disarm":
            self._update(armed=False, in_air=False)

    def land_now(self) -> None:
        self._update(armed=False, in_air=False, alt_amsl=self.ground, mode="LAND")


def _remote(engine):
    fake = FakeAutopilot(engine.geo, engine.environment.home_position + np.array([30.0, 0.0, 0.0]))
    drone = engine.swarm.add_remote(lambda i: create_remote_drone({"type": "mavlink", "url": fake.url, "name": "HW1"}, i,
                                                                  engine.config, engine.environment, engine.geo,
                                                                  engine.events, link=fake))
    return fake, drone


def test_remote_drone_follows_the_autopilot(engine):
    fake, d = _remote(engine)
    engine.step()
    assert d.is_remote and d.source == "mavlink" and d.name == "HW1" and d.comm_status == CommStatus.ONLINE
    assert abs(d.position[0] - 30.0) < 0.5 and d.battery.percent == pytest.approx(87.0, abs=0.5)
    res = engine.execute({"type": "takeoff", "drone_ids": [d.id], "params": {"altitude": 12}})
    assert res.success and fake.sent[-1] == ("takeoff", 12.0)
    fake.beat()
    engine.step()
    assert d.flight_mode == FlightMode.HOVER and abs(d.altitude_agl - 12.0) < 0.5
    res = engine.execute({"type": "goto", "drone_ids": [d.id], "params": {"position": [50, 20, 15], "plan": False}})
    assert res.success and fake.sent[-1][0] == "goto"
    fake.beat()
    engine.step()
    assert d.flight_mode == FlightMode.HOVER and np.linalg.norm(d.position - [50, 20, 15]) < 0.5      # arrived
    tel = d.get_telemetry().to_dict()
    assert tel["source"] == "mavlink" and tel["gps_fix"] == "3D"
    engine.execute({"type": "land", "drone_ids": [d.id]})
    assert fake.sent[-1] == ("mode", "LAND")
    fake.land_now()
    fake.beat()
    engine.step()
    assert d.flight_mode == FlightMode.DISARMED
    # heartbeat timeout -> COMM LOST (critical event), commands refused
    fake._update(last_heartbeat=time.monotonic() - 10)
    engine.step()
    assert d.comm_status == CommStatus.LOST
    assert any(e.kind == "comm_lost" and e.drone_id == d.id for e in engine.events.recent(0, 50))
    assert not engine.execute({"type": "takeoff", "drone_ids": [d.id]}).success


def test_remote_drone_in_a_formation_streams_velocity(engine):
    fake, d = _remote(engine)
    engine.step()                                          # first sync: link ONLINE
    engine.execute({"type": "takeoff", "params": {"altitude": 12}})
    for _ in range(int(8 / engine.dt)):
        fake.beat()
        engine.step()
    assert d.airborne
    engine.execute({"type": "set_formation", "params": {"shape": "line", "spacing": 10}})
    for _ in range(int(2 / engine.dt)):
        fake.beat()
        engine.step()
    assert d.flight_mode == FlightMode.FORMATION
    vel = [s for s in fake.sent if s[0] == "velocity"]
    rate = engine.config.hardware.setpoint_rate_hz
    assert 0.5 * 2 * rate <= len(vel) <= 2 * rate + 2                   # rate limited to hardware.setpoint_rate_hz


def test_hardware_config_validation():
    for bad in ([{"type": "serial", "url": "x"}], [{"type": "mavlink"}], [{"type": "mavlink", "url": "a", "port": 1}],
                [{"type": "mavlink", "url": "a"}, {"type": "mavsdk", "url": "a"}],
                [{"type": "mavlink", "url": "a", "system_id": 300}]):
        with pytest.raises(ConfigError):
            make_config(**{"hardware.vehicles": bad})


# ============================================================================ pymavlink loopback

def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_mavlink_link_loopback():
    """MAVLinkLink against a scripted MAVLink vehicle over TCP, like SITL's serial port (no SITL needed)."""
    mavutil = pytest.importorskip("pymavlink.mavutil")
    from integration.mavlink_drone import MAVLinkLink
    port = _free_tcp_port()
    vehicle = mavutil.mavlink_connection(f"tcpin:127.0.0.1:{port}", source_system=1, source_component=1)
    link = MAVLinkLink(f"tcp:127.0.0.1:{port}")
    link.start()
    m = mavutil.mavlink
    got_arm = False
    try:
        deadline = time.monotonic() + 10
        sent_arm = False
        while time.monotonic() < deadline:
            vehicle.mav.heartbeat_send(m.MAV_TYPE_QUADROTOR, m.MAV_AUTOPILOT_ARDUPILOTMEGA,
                                       m.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 4, m.MAV_STATE_STANDBY)   # 4 = GUIDED
            vehicle.mav.global_position_int_send(0, int(47.3977 * 1e7), int(8.5456 * 1e7), 488000, 0, 150, -250, 0, 9000)
            vehicle.mav.attitude_send(0, 0.01, -0.02, 1.5708, 0, 0, 0)
            s = link.snapshot()
            if s.connected and s.has_position and not sent_arm:
                link.send("arm")
                sent_arm = True
            msg = vehicle.recv_match(type="COMMAND_LONG", blocking=True, timeout=0.2)
            if msg is not None and msg.command == m.MAV_CMD_COMPONENT_ARM_DISARM and msg.param1 == 1:
                got_arm = True
                break
        s = link.snapshot()
        assert s.connected and s.mode == "GUIDED" and s.autopilot == "ardupilot"
        assert s.lat == pytest.approx(47.3977) and s.vel_ned == pytest.approx((1.5, -2.5, 0.0))
        assert s.yaw_ned == pytest.approx(1.5708, abs=1e-4)
        assert got_arm, "the vehicle never received the arm command"
    finally:
        link.close()
        vehicle.close()


# ============================================================================ ArduPilot SITL (opt-in)

@pytest.mark.sitl
@pytest.mark.skipif(not os.environ.get("SWARM_SITL_DIR"), reason="set SWARM_SITL_DIR to an ArduCopter SITL build")
def test_ardupilot_sitl_swarm():
    """3 ArduPilot SITL instances in one swarm through the MAVLink adapter (see scripts/sitl_swarm.py)."""
    import subprocess
    import sys
    script = Path(__file__).resolve().parent.parent / "scripts" / "sitl_swarm.py"
    out = subprocess.run([sys.executable, str(script), "--instances", "3", "--link", os.environ.get("SWARM_SITL_LINK", "mavlink")],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    assert "RESULT: PASS" in out.stdout
