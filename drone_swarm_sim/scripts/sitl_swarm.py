"""End-to-end hardware-adapter check: N ArduPilot SITL copters + simulated drones in one swarm.

Starts ``N`` ArduCopter SITL instances (TCP 5760 + 10*i, homes 12 m apart), adds them to a GCS engine
through the MAVLink (pymavlink) or MAVSDK adapter together with two simulated drones, then flies:
takeoff -> group goto (SITL only) -> LINE formation (SITL + simulated) -> land. Prints
``RESULT: PASS`` / ``RESULT: FAIL``.

Usage::

    python scripts/sitl_swarm.py --sitl-dir <dir with ArduCopter(.elf|.exe) and copter.parm> \
        [--instances 3] [--link mavlink|mavsdk]

``--sitl-dir`` defaults to ``$SWARM_SITL_DIR``. Get a SITL build from https://firmware.ardupilot.org
(``Tools/autotest/sim_vehicle.py`` builds one from source; Windows builds need the Cygwin DLLs next to it)
and the default parameters from ``Tools/autotest/default_params/copter.parm``.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.config import load_config  # noqa: E402
from simulation.engine import SimulationEngine  # noqa: E402
from simulation.types import FlightMode  # noqa: E402

HOME = (47.397742, 8.545594, 488.0)


def find_binary(d: Path) -> Path:
    for name in ("ArduCopter.elf", "ArduCopter.exe", "arducopter", "ArduCopter"):
        if (d / name).is_file():
            return d / name
    raise SystemExit(f"no ArduCopter SITL binary in {d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sitl-dir", default=os.environ.get("SWARM_SITL_DIR"), help="SITL binary + copter.parm")
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--link", choices=("mavlink", "mavsdk"), default="mavlink")
    args = ap.parse_args()
    if not args.sitl_dir:
        raise SystemExit("--sitl-dir (or $SWARM_SITL_DIR) is required")
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sitl = Path(args.sitl_dir)
    binary = find_binary(sitl)
    params = sitl / "copter.parm"
    procs = []
    for i in range(args.instances):
        work = sitl / f"run{i}"
        work.mkdir(exist_ok=True)
        lon = HOME[1] + i * 12 / (111320 * np.cos(np.radians(HOME[0])))
        cmd = [str(binary), "--model", "+", "--speedup", "1", f"-I{i}", "--home", f"{HOME[0]},{lon},{HOME[2]},0"]
        if params.exists():
            cmd += ["--defaults", str(params)]
        procs.append(subprocess.Popen(cmd, cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    time.sleep(3)
    url = "tcp:127.0.0.1:{}" if args.link == "mavlink" else "tcpout://127.0.0.1:{}"
    vehicles = [{"type": args.link, "url": url.format(5760 + 10 * i), "name": f"SITL{i + 1}"} for i in range(args.instances)]
    cfg = load_config(overrides={
        "simulation.drone_count": 2, "logging.record_telemetry": False, "terrain.enabled": False,
        "environment.scene_file": None, "communication.enabled": False, "sensors.enabled": False,
        "security.enabled": False, "home.position_x": -60.0, "hardware.vehicles": vehicles,
        "origin.latitude": HOME[0], "origin.longitude": HOME[1], "origin.altitude": HOME[2]})
    eng = SimulationEngine(cfg, record=False)
    remote = [d for d in eng.swarm if getattr(d, "is_remote", False)]
    checks: list[tuple[str, bool]] = []

    def run(seconds, until=None) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            eng.step()
            time.sleep(eng.dt)
            if until is not None and until():
                return True
        return False

    try:
        print(f"swarm: {len(eng.swarm)} drones ({len(remote)} {args.link} SITL + {len(eng.swarm) - len(remote)} simulated)")
        ok = run(120, lambda: all(str(d.comm_status) == "ONLINE" and d.remote_ready and d.gps_fix == "3D" for d in remote))
        checks.append(("connected + pre-arm ready", ok))
        print("takeoff:", eng.execute({"type": "takeoff", "params": {"altitude": 15}}).message)
        ok = run(90, lambda: all(d.flight_mode == FlightMode.HOVER and d.altitude_agl > 13 for d in eng.swarm))
        checks.append(("all airborne at 15 m", ok))
        ids = [d.id for d in remote]
        print("group goto:", eng.execute({"type": "goto", "drone_ids": ids, "params": {"position": [40, 60, 20], "speed": 6}}).message)
        ok = run(90, lambda: all(d.flight_mode == FlightMode.HOVER for d in remote))
        cent = np.mean([d.position for d in remote], axis=0)
        checks.append(("group goto arrived", ok and np.linalg.norm(cent[:2] - [40, 60]) < 5))
        print("formation:", eng.execute({"type": "set_formation", "params": {"shape": "line", "spacing": 15}}).message)
        run(40)
        f = eng.coordinator.formation
        min_sep = eng.swarm.summary()["min_separation"]
        print(f"formation error {f.max_error:.2f} m, min separation {min_sep} m")
        checks.append(("mixed formation holds (error < 3 m, separation > 10 m)", f.max_error < 3.0 and (min_sep or 0) > 10))
        print("land:", eng.execute({"type": "land"}).message)
        ok = run(90, lambda: all(not d.armed for d in eng.swarm))
        checks.append(("landed + disarmed", ok))
    finally:
        eng.close()
        for p in procs:
            p.terminate()
    for name, ok in checks:
        print(f"  [{'x' if ok else ' '}] {name}")
    passed = len(checks) == 5 and all(ok for _, ok in checks)
    print("RESULT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
