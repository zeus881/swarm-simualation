import time

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app

from .conftest import make_config


@pytest.fixture
def client():
    app = create_app(make_config(simulation__real_time_factor=4.0), record=False)
    with TestClient(app) as c:
        yield c


def test_health_status_and_config(client):
    assert client.get("/api/health").json()["status"] == "ok"
    for _ in range(50):
        r = client.get("/api/status")
        if r.status_code == 200:
            break
        time.sleep(0.05)
    body = r.json()
    assert body["state"] == "RUNNING" and body["summary"]["total"] == 3
    assert client.get("/api/config").json()["simulation"]["drone_count"] == 3


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "SWARM CONTROL CENTER" in r.text
    js = client.get("/src/main.js")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert client.get("/vendor/three/three.module.js").status_code == 200


def test_rest_command_and_drone_telemetry(client):
    r = client.post("/api/commands", json={"type": "takeoff", "params": {"altitude": 15}})
    assert r.status_code == 200 and r.json()["success"]
    time.sleep(1.0)
    drones = client.get("/api/drones").json()
    assert len(drones) == 3 and all(d["armed"] for d in drones)
    assert client.get("/api/drones/1").json()["drone_id"] == 1
    assert client.get("/api/drones/99").status_code == 404
    bad = client.post("/api/commands", json={"type": "nope"})
    assert bad.status_code == 200 and not bad.json()["success"]
    assert "takeoff" in client.get("/api/commands").json()
    assert any(e["kind"] == "takeoff" for e in client.get("/api/events?limit=50").json())


def test_simulation_control(client):
    assert client.post("/api/simulation/pause").json()["state"] == "PAUSED"
    t1 = client.get("/api/status").json()["sim_time"]
    time.sleep(0.6)
    assert client.get("/api/status").json()["sim_time"] == t1
    assert client.post("/api/simulation/start").json()["state"] == "RUNNING"
    assert client.post("/api/simulation/reset").json()["success"]
    assert client.post("/api/simulation/explode").status_code == 422


def test_websocket_telemetry_and_commands(client):
    with client.websocket_connect("/ws/telemetry") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "telemetry" and len(msg["drones"]) == 3
        d = msg["drones"][0]
        for key in ("drone_id", "position", "velocity", "battery", "heading", "mode", "task", "communication"):
            assert key in d
        ws.send_json({"type": "command", "id": 7, "command": {"type": "takeoff", "drone_ids": [2], "params": {}}})
        ws.send_json({"type": "ping", "id": 8, "client_time": 123})
        ws.send_json({"type": "bogus", "id": 9})
        replies = {}
        for _ in range(200):
            m = ws.receive_json()
            if m["type"] != "telemetry":
                replies[m["id"]] = m
            if len(replies) == 3:
                break
        assert replies[7]["type"] == "command_result" and replies[7]["success"]
        assert replies[8]["type"] == "pong" and replies[8]["client_time"] == 123
        assert replies[9]["type"] == "error"
