"""Command-line entry point: ``python -m simulation``.

Examples::

    python -m simulation                                  # GCS at http://127.0.0.1:8000
    python -m simulation --drones 25 --open               # 25 drones, open the browser
    python -m simulation --set wind.speed=10 --set battery.drain_multiplier=20
    python -m simulation --headless --duration 60 --takeoff
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from .config import ConfigError, load_config, parse_overrides
from .logging_setup import configure_logging

log = logging.getLogger("simulation")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m simulation", description="Multi-UAV swarm simulation platform")
    p.add_argument("--config", help="YAML configuration file (default: configs/simulation.yaml or $SWARM_CONFIG)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a configuration value (repeatable)")
    p.add_argument("--drones", type=int, help="shortcut for --set simulation.drone_count=N")
    p.add_argument("--host", help="server bind address (default from config)")
    p.add_argument("--port", type=int, help="server port (default from config)")
    p.add_argument("--open", action="store_true", help="open the dashboard in a web browser")
    p.add_argument("--headless", action="store_true", help="run without the web server")
    p.add_argument("--duration", type=float, default=30.0, help="headless: simulated seconds to run")
    p.add_argument("--takeoff", action="store_true", help="headless: take off all drones at start")
    return p


def run_headless(config, duration: float, takeoff: bool) -> None:
    from .engine import SimulationEngine

    engine = SimulationEngine(config)
    run_dir = engine.recorder.directory if engine.recorder is not None else None
    if takeoff:
        print(engine.execute({"type": "takeoff"}).message)
    wall_start = time.perf_counter()
    report_every = max(1, int(config.simulation.simulation_rate))
    try:
        while engine.sim_time < duration:
            engine.step()
            if engine.tick % report_every == 0:
                s = engine.swarm.summary()
                print(f"t={engine.sim_time:7.2f}s  airborne={s['airborne']:3d}/{s['total']:<3d} "
                      f"battery={s['average_battery']}%  min_sep={s['min_separation']}  "
                      f"step={engine.step_time_ms:.2f} ms")
    finally:
        engine.close()
    wall = time.perf_counter() - wall_start
    print(f"simulated {engine.sim_time:.1f} s in {wall:.2f} s wall ({engine.sim_time / max(wall, 1e-9):.1f}x real time)")
    if run_dir is not None:
        print(f"recordings: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overrides = parse_overrides(args.overrides)
        if args.drones is not None:
            overrides["simulation.drone_count"] = args.drones
        if args.host:
            overrides["server.host"] = args.host
        if args.port:
            overrides["server.port"] = args.port
        config = load_config(args.config, overrides)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    configure_logging(config.logging)

    if args.headless:
        run_headless(config, args.duration, args.takeoff)
        return 0

    from backend.main import serve

    serve(config, open_browser=args.open)
    return 0


if __name__ == "__main__":
    sys.exit(main())
