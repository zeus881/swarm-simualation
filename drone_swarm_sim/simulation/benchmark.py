"""Stress test / benchmark: ``python -m simulation.benchmark``.

For each swarm size the benchmark takes off every drone, sends it on a
cross-swarm goto (so all drones move at once), then measures:

* engine step time (mean / p95 / max) and the tick rate it can sustain,
* snapshot build + JSON encoding time and payload size (the telemetry path),
* process CPU usage and resident memory (requires psutil).

Results are printed as a table and written to ``logs/benchmark.json``.
Network latency is measured end-to-end by the dashboard (Latency metric).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .engine import SimulationEngine

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def run_case(drones: int, seconds: float, rate: float) -> dict:
    cfg = load_config(overrides={
        "simulation.drone_count": drones,
        "simulation.simulation_rate": rate,
        "logging.record_telemetry": False,
        "home.pad_spacing": 12.0,
    })
    engine = SimulationEngine(cfg, record=False)
    engine.execute({"type": "takeoff", "params": {"altitude": 30}})
    engine.run_for(8.0)
    # Mirror the formation through the home point so every drone flies at the same time.
    for d in engine.swarm:
        p = d.position
        d.goto((-p[0] * 3, -p[1] * 3, 30 + (d.id % 5) * 3))

    proc = psutil.Process() if psutil else None
    if proc:
        proc.cpu_percent(None)
    ticks = int(seconds * rate)
    step_ms = np.empty(ticks)
    for i in range(ticks):
        t0 = time.perf_counter()
        engine.step()
        step_ms[i] = (time.perf_counter() - t0) * 1000
    cpu = proc.cpu_percent(None) if proc else None

    snap_ms, sizes = [], []
    for _ in range(20):
        t0 = time.perf_counter()
        text = json.dumps(engine.snapshot(), separators=(",", ":"))
        snap_ms.append((time.perf_counter() - t0) * 1000)
        sizes.append(len(text))
    engine.close()

    mean = float(step_ms.mean())
    budget = 1000.0 / rate
    return {
        "drones": drones,
        "rate_hz": rate,
        "step_mean_ms": round(mean, 3),
        "step_p95_ms": round(float(np.percentile(step_ms, 95)), 3),
        "step_max_ms": round(float(step_ms.max()), 3),
        "max_sustainable_hz": round(1000.0 / mean, 1),
        "real_time_ok": bool(np.percentile(step_ms, 95) < budget),
        "snapshot_ms": round(statistics.mean(snap_ms), 3),
        "snapshot_kb": round(statistics.mean(sizes) / 1024, 1),
        "cpu_percent": round(cpu, 1) if cpu is not None else None,
        "rss_mb": round(proc.memory_info().rss / 2**20, 1) if proc else None,
        "separation_violations": engine.swarm.collision_monitor.total_violations,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m simulation.benchmark")
    p.add_argument("--drones", type=int, nargs="+", default=[10, 25, 50, 100])
    p.add_argument("--seconds", type=float, default=10.0, help="simulated seconds measured per case")
    p.add_argument("--rate", type=float, default=30.0, help="simulation rate [Hz]")
    p.add_argument("--output", default=None, help="JSON output path (default logs/benchmark.json)")
    args = p.parse_args(argv)
    # The crossing manoeuvre deliberately produces separation violations (no avoidance until
    # Phase 2); keep the per-event log lines out of the benchmark table.
    logging.getLogger("simulation.events").setLevel(logging.ERROR)

    header = f"{'drones':>6} {'step ms':>8} {'p95 ms':>7} {'max ms':>7} {'max Hz':>8} {'RT@' + str(int(args.rate)):>6} " \
             f"{'snap ms':>8} {'snap kB':>8} {'CPU %':>6} {'RSS MB':>7}"
    print(header)
    print("-" * len(header))
    results = []
    for n in args.drones:
        r = run_case(n, args.seconds, args.rate)
        results.append(r)
        print(f"{r['drones']:>6} {r['step_mean_ms']:>8.2f} {r['step_p95_ms']:>7.2f} {r['step_max_ms']:>7.2f} "
              f"{r['max_sustainable_hz']:>8.0f} {'yes' if r['real_time_ok'] else 'NO':>6} {r['snapshot_ms']:>8.2f} "
              f"{r['snapshot_kb']:>8.1f} {r['cpu_percent'] if r['cpu_percent'] is not None else '-':>6} "
              f"{r['rss_mb'] if r['rss_mb'] is not None else '-':>7}")
    out = Path(args.output) if args.output else load_config().logging.path / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresults written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
