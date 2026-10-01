"""Real-time runner: drives a :class:`SimulationEngine` from a dedicated thread.

* Fixed-rate loop paced against ``time.perf_counter`` and the real-time factor.
  If the loop falls behind by more than ``MAX_LAG_S`` it drops time instead of
  spiralling (the simulation slows down; it never skips physics).
* Commands and control actions are queued and executed in the engine thread
  at tick boundaries; callers receive a ``concurrent.futures.Future``.
* Snapshots are published at the telemetry rate as immutable dicts. Readers
  on any thread call :meth:`latest_snapshot`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from enum import StrEnum
from typing import Any, Callable, Mapping

from .commands import Command
from .drone_interface import CommandResult
from .engine import SimulationEngine

log = logging.getLogger(__name__)


class RunState(StrEnum):
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class SimulationRunner:
    MAX_LAG_S = 0.25

    def __init__(self, engine: SimulationEngine) -> None:
        self.engine = engine
        cfg = engine.config
        self.real_time_factor = cfg.simulation.real_time_factor
        self._telemetry_period = 1.0 / cfg.telemetry.rate_hz
        self._paused_period = 1.0 / cfg.telemetry.paused_rate_hz
        self._state = RunState.PAUSED
        self._queue: queue.SimpleQueue[tuple[Callable[[SimulationEngine], Any], Future]] = queue.SimpleQueue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshot_lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._snapshot_seq = 0
        self._dirty = True
        self._tick_times: deque[float] = deque(maxlen=120)
        self.overruns = 0

    # ------------------------------------------------------------------ control (any thread)
    @property
    def state(self) -> RunState:
        return self._state

    def start_thread(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="simulation-engine", daemon=True)
        self._thread.start()

    def resume(self) -> None:
        if self._state != RunState.STOPPED:
            self._state = RunState.RUNNING
            self._dirty = True
            self._wake.set()

    start = resume

    def pause(self) -> None:
        if self._state != RunState.STOPPED:
            self._state = RunState.PAUSED
            self._dirty = True
            self._wake.set()

    def reset(self) -> Future:
        return self.call(lambda eng: eng.reset())

    def shutdown(self, timeout: float = 5.0) -> None:
        self._state = RunState.STOPPED
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        # Fail any requests that will never be served.
        while True:
            try:
                _, fut = self._queue.get_nowait()
            except queue.Empty:
                break
            fut.set_exception(RuntimeError("simulation runner stopped"))

    def call(self, fn: Callable[[SimulationEngine], Any]) -> Future:
        """Run ``fn(engine)`` in the engine thread at the next tick boundary."""
        fut: Future = Future()
        if self._state == RunState.STOPPED:
            fut.set_exception(RuntimeError("simulation runner stopped"))
            return fut
        self._queue.put((fn, fut))
        self._wake.set()
        return fut

    def submit_command(self, command: Command | Mapping[str, Any]) -> "Future[CommandResult]":
        return self.call(lambda eng: eng.execute(command))

    def latest_snapshot(self) -> tuple[int, dict[str, Any]]:
        with self._snapshot_lock:
            return self._snapshot_seq, self._snapshot

    # ------------------------------------------------------------------ engine thread
    def _drain(self) -> bool:
        executed = False
        while True:
            try:
                fn, fut = self._queue.get_nowait()
            except queue.Empty:
                return executed
            executed = True
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn(self.engine))
            except Exception as exc:
                log.exception("Engine call failed")
                fut.set_exception(exc)

    def stats(self) -> dict[str, Any]:
        times = self._tick_times
        rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 2 and times[-1] > times[0] else 0.0
        nominal = self.engine.config.simulation.simulation_rate
        return {
            "state": str(self._state),
            "sim_rate_hz": round(rate, 1),
            "target_rate_hz": round(nominal * self.real_time_factor, 1),
            "real_time_factor": round(rate / nominal, 2) if nominal else 0.0,
            "configured_real_time_factor": self.real_time_factor,
            "step_ms": round(self.engine.step_time_ms, 3),
            "max_step_ms": round(self.engine.max_step_time_ms, 3),
            "overruns": self.overruns,
        }

    def _publish(self) -> None:
        snap = self.engine.snapshot()
        snap["state"] = str(self._state)
        snap["stats"] = self.stats()
        with self._snapshot_lock:
            self._snapshot_seq += 1
            self._snapshot = snap
        self._dirty = False

    def _loop(self) -> None:
        log.info("Simulation thread started (%.0f Hz, RTF %.2f)",
                 self.engine.config.simulation.simulation_rate, self.real_time_factor)
        next_tick = time.perf_counter()
        last_publish = 0.0
        next_publish = next_tick
        run_id = self.engine.run_id
        while not self._stop.is_set():
            try:
                if self._drain():
                    self._dirty = True
                    if self.engine.run_id != run_id:      # reset happened
                        run_id = self.engine.run_id
                        self._tick_times.clear()
                        next_tick = time.perf_counter()
                now = time.perf_counter()
                if self._state == RunState.RUNNING:
                    if now >= next_tick:
                        self.engine.step()
                        self._tick_times.append(now)
                        next_tick += self.engine.dt / self.real_time_factor
                        if now - next_tick > self.MAX_LAG_S:
                            self.overruns += 1
                            next_tick = now
                        # Deadline schedule (not "time since last"): with 33 ms ticks and a 50 ms
                        # period this publishes on ticks 0, 2, 3, 5, 6... = 20 Hz on average.
                        if now >= next_publish or self._dirty:
                            self._publish()
                            last_publish = now
                            next_publish = max(next_publish + self._telemetry_period, now - self._telemetry_period)
                    timeout = max(0.0, next_tick - time.perf_counter())
                else:
                    next_tick = next_publish = now
                    if self._dirty or now - last_publish >= self._paused_period:
                        self._publish()
                        last_publish = now
                    timeout = self._paused_period
            except Exception:
                # Keep the thread alive: pause and surface the problem instead of dying silently.
                log.exception("Simulation step failed - pausing")
                self._state = RunState.PAUSED
                self._dirty = True
                timeout = 0.1
            if timeout > 0:
                self._wake.wait(timeout)
            self._wake.clear()
        log.info("Simulation thread stopped")
