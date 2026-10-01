"""Mission replay: reads the recorded runs in ``logs/<run_id>`` (Stage 5).

A run's ``telemetry.csv`` is parsed once into compact NumPy columns (float32 numbers + small
categorical codes) and cached (LRU, ``replay.cache_runs``). The REPLAY tab then fetches windows of
frames (``replay.chunk_s`` seconds each) while it plays, so even a long 50-drone run never travels
to the browser in one piece. Everything here is plain file I/O + NumPy: the web layer runs it in
its thread pool, never in the simulation tick.

Runs recorded before Stage 5 (without the name / altitude_agl / ... columns) are still readable:
missing columns get sensible defaults.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .recorder import read_json_array
from .types import AIRBORNE_MODES

log = logging.getLogger(__name__)

RUN_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,79}$")

# Columns sent to the browser for each drone and frame, in this order.
NUMERIC = ("x", "y", "z", "vx", "vy", "vz", "speed", "heading", "roll", "pitch", "battery", "altitude_agl",
           "nearest_distance", "lat", "lon", "alt")
CATEGORICAL = ("mode", "health", "communication", "collision_state", "battery_state", "task", "name", "source")
FLAGS = ("armed", "airborne")


class ReplayError(ValueError):
    """Unknown run or unreadable recording."""


def _started_at(run_id: str) -> str | None:
    try:
        return datetime.strptime(run_id[:15], "%Y%m%d-%H%M%S").isoformat()
    except ValueError:
        return None


def _tail_line(path: Path) -> str:
    """Last complete line of a text file (cheap: reads at most 8 KiB)."""
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 8192))
        lines = f.read().decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        if line.strip() and not line.startswith("timestamp"):
            return line
    return ""


def list_runs(log_dir: Path) -> list[dict[str, Any]]:
    """Recorded runs, newest first (cheap: no telemetry parsing)."""
    runs = []
    if not log_dir.is_dir():
        return runs
    for d in log_dir.iterdir():
        if not d.is_dir() or not RUN_ID_RE.match(d.name) or not (d / "config.json").exists():
            continue
        tel = d / "telemetry.csv"
        duration, size = 0.0, 0
        if tel.exists():
            size = tel.stat().st_size
            try:
                last = _tail_line(tel)
                duration = float(last.split(",", 1)[0]) if last else 0.0
            except (OSError, ValueError):
                duration = 0.0
        drones = None
        world = d / "world.json"
        if world.exists():
            try:
                drones = len(json.loads(world.read_text(encoding="utf-8")).get("drones") or [])
            except (OSError, ValueError):
                drones = None
        if drones is None:
            try:
                drones = json.loads((d / "config.json").read_text(encoding="utf-8"))["simulation"]["drone_count"]
            except (OSError, ValueError, KeyError, TypeError):
                drones = 0
        runs.append({"run_id": d.name, "started_at": _started_at(d.name), "duration_s": round(duration, 1),
                     "drones": drones, "telemetry_bytes": size, "has_telemetry": size > 0})
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


@dataclass(slots=True)
class _Columns:
    t: np.ndarray                    # float64 [rows]
    ids: np.ndarray                  # int32 [rows]
    num: np.ndarray                  # float32 [rows, len(NUMERIC)] (NaN = missing)
    latlon: np.ndarray               # float64 [rows, 2] (full precision geodetic position)
    cat: np.ndarray                  # int16 [rows, len(CATEGORICAL)]
    flags: np.ndarray                # int8 [rows, len(FLAGS)]
    vocab: dict[str, list[str]]


class RunLog:
    """One recorded run, parsed."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.run_id = directory.name
        self.config = self._json("config.json") or {}
        self.world = self._json("world.json") or {}
        self._cols = self._parse(directory / "telemetry.csv")
        c = self._cols
        self.frame_times, self._starts = np.unique(c.t, return_index=True)
        self._starts = np.append(self._starts, len(c.t))
        self._events: list[dict[str, Any]] | None = None
        self._separation: np.ndarray | None = None

    # ------------------------------------------------------------------ loading
    def _json(self, name: str) -> Any:
        path = self.directory / name
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, ValueError):
            log.warning("Replay: cannot read %s", path)
            return None

    @staticmethod
    def _parse(path: Path) -> _Columns:
        if not path.exists():
            raise ReplayError(f"{path.parent.name}: no telemetry recorded (logging.record_telemetry was off)")
        vocab: dict[str, dict[str, int]] = {k: {} for k in CATEGORICAL}
        t, ids, num, cat, flags = [], [], [], [], []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ReplayError(f"{path.parent.name}: empty telemetry file") from None
            col = {name: i for i, name in enumerate(header)}
            if not {"timestamp", "drone_id", "x", "y", "z"} <= col.keys():
                raise ReplayError(f"{path.parent.name}: not a telemetry recording")
            n_idx = [col.get(k) for k in NUMERIC]
            c_idx = [col.get(k) for k in CATEGORICAL]
            f_idx = [col.get(k) for k in FLAGS]
            i_t, i_id, i_z, i_mode = col["timestamp"], col["drone_id"], col["z"], col.get("mode")
            width = len(header)
            for row in reader:
                if len(row) < width:                     # the row being written while recording
                    continue
                try:
                    t.append(float(row[i_t]))
                    did = int(row[i_id])
                except ValueError:
                    continue
                ids.append(did)
                num.append([float(row[i]) if i is not None and row[i] != "" else np.nan for i in n_idx])
                codes = []
                for k, i in zip(CATEGORICAL, c_idx):
                    if i is not None:
                        v = row[i]
                    elif k == "name":
                        v = f"D{did:02d}"
                    elif k == "source":
                        v = "sim"
                    else:
                        v = ""
                    table = vocab[k]
                    codes.append(table.setdefault(v, len(table)))
                cat.append(codes)
                fl = []
                for k, i in zip(FLAGS, f_idx):
                    if i is not None:
                        fl.append(1 if row[i] in ("1", "True", "true") else 0)
                    elif k == "airborne":               # pre-Stage-5 recording: derive from the mode
                        fl.append(1 if i_mode is not None and row[i_mode] in AIRBORNE_MODES else 0)
                    else:
                        fl.append(0)
                flags.append(fl)
        if not t:
            raise ReplayError(f"{path.parent.name}: telemetry has no rows yet")
        num64 = np.asarray(num, dtype=np.float64)
        ll = num64[:, [NUMERIC.index("lat"), NUMERIC.index("lon")]]      # float32 would cost ~0.4 m
        num_a = num64.astype(np.float32)
        del num64
        agl = NUMERIC.index("altitude_agl")
        missing = np.isnan(num_a[:, agl])
        num_a[missing, agl] = num_a[missing, NUMERIC.index("z")]
        t_a = np.asarray(t, dtype=np.float64)
        order = np.lexsort((np.asarray(ids), t_a))    # rows are written sorted; be robust anyway
        return _Columns(t_a[order], np.asarray(ids, dtype=np.int32)[order], num_a[order], ll[order],
                        np.asarray(cat, dtype=np.int16)[order], np.asarray(flags, dtype=np.int8)[order],
                        {k: list(v) for k, v in vocab.items()})

    # ------------------------------------------------------------------ queries
    @property
    def duration(self) -> float:
        return float(self.frame_times[-1]) if len(self.frame_times) else 0.0

    def drones(self) -> list[dict[str, Any]]:
        c = self._cols
        names, sources = c.vocab["name"], c.vocab["source"]
        ni, si = CATEGORICAL.index("name"), CATEGORICAL.index("source")
        out: dict[int, dict[str, Any]] = {}
        for did in np.unique(c.ids):
            k = int(np.argmax(c.ids == did))
            out[int(did)] = {"id": int(did), "name": names[c.cat[k, ni]], "source": sources[c.cat[k, si]]}
        return [out[k] for k in sorted(out)]

    def events(self) -> list[dict[str, Any]]:
        """Every recorded event (mission + collision files), by time."""
        if self._events is None:
            items = read_json_array(self.directory / "mission_events.json")
            items += read_json_array(self.directory / "collision_events.json")
            items.sort(key=lambda e: (e.get("time", 0.0), e.get("seq", 0)))
            self._events = items
        return self._events

    def markers(self, limit: int) -> list[dict[str, Any]]:
        """Events worth a scrubber marker: warnings, criticals, commands and mission events."""
        keep = [e for e in self.events()
                if e.get("severity") != "INFO" or e.get("category") in ("COMMAND", "MISSION")]
        if len(keep) > limit:                          # keep every critical, thin out the rest evenly
            crit = [e for e in keep if e.get("severity") == "CRITICAL"][:limit]
            rest = [e for e in keep if e.get("severity") != "CRITICAL"]
            step = max(1, len(rest) // max(1, limit - len(crit)))
            keep = sorted(crit + rest[::step][:limit - len(crit)], key=lambda e: e.get("time", 0.0))
        return [{"seq": e.get("seq"), "time": e.get("time"), "category": e.get("category"), "kind": e.get("kind"),
                 "severity": e.get("severity"), "drone_id": e.get("drone_id"), "message": e.get("message")}
                for e in keep]

    def meta(self, max_events: int) -> dict[str, Any]:
        sim = self.config.get("simulation", {}) if isinstance(self.config, dict) else {}
        rate = (len(self.frame_times) - 1) / self.duration if self.duration > 0 else 0.0
        return {
            "run_id": self.run_id,
            "started_at": _started_at(self.run_id),
            "duration_s": round(self.duration, 3),
            "frames": int(len(self.frame_times)),
            "record_rate_hz": round(rate, 2),
            "drones": self.drones(),
            "seed": sim.get("seed"),
            "world": self.world.get("world"),
            "geofence": self.world.get("geofence"),
            "event_count": len(self.events()),
            "events": self.markers(max_events),
            "columns": {"numeric": list(NUMERIC), "categorical": list(CATEGORICAL), "flags": list(FLAGS)},
            "vocab": self._cols.vocab,
        }

    def _frame_slice(self, k: int) -> slice:
        return slice(int(self._starts[k]), int(self._starts[k + 1]))

    def frames(self, start: float, end: float) -> dict[str, Any]:
        """Frames with ``start <= t <= end``; each row is ``[id, *NUMERIC, *CATEGORICAL codes, *FLAGS]``."""
        start, end = max(0.0, float(start)), float(end)
        if not (np.isfinite(start) and np.isfinite(end)) or end < start:
            raise ReplayError("frames: need 0 <= start <= end")
        k0 = int(np.searchsorted(self.frame_times, start - 1e-9, side="left"))
        k1 = int(np.searchsorted(self.frame_times, end + 1e-9, side="right"))
        c = self._cols
        out = []
        for k in range(k0, k1):
            s = self._frame_slice(k)
            num = np.round(c.num[s].astype(np.float64), 2)
            num[:, NUMERIC.index("lat")] = np.round(c.latlon[s, 0], 7)
            num[:, NUMERIC.index("lon")] = np.round(c.latlon[s, 1], 7)
            rows = np.column_stack([c.ids[s], num, c.cat[s], c.flags[s]])
            clean = [[None if isinstance(v, float) and v != v else v for v in r] for r in rows.tolist()]
            for r in clean:
                r[0] = int(r[0])
                for j in range(1 + len(NUMERIC), len(r)):
                    r[j] = int(r[j])
            out.append({"t": round(float(self.frame_times[k]), 3), "rows": clean})
        return {"run_id": self.run_id, "start": start, "end": end, "frames": out}

    def tracks(self, max_points: int) -> dict[int, dict[str, np.ndarray]]:
        """Per drone: t, x, y, z, battery, agl, airborne - decimated to ``max_points``."""
        c = self._cols
        out = {}
        for did in np.unique(c.ids):
            rows = np.flatnonzero(c.ids == did)
            step = max(1, int(np.ceil(len(rows) / max_points)))
            sel = rows[::step]
            if sel[-1] != rows[-1]:
                sel = np.append(sel, rows[-1])
            out[int(did)] = {
                "t": c.t[sel], "x": c.num[sel, 0].astype(np.float64), "y": c.num[sel, 1].astype(np.float64),
                "z": c.num[sel, 2].astype(np.float64), "battery": c.num[sel, NUMERIC.index("battery")].astype(np.float64),
                "agl": c.num[sel, NUMERIC.index("altitude_agl")].astype(np.float64),
                "airborne": c.flags[sel, FLAGS.index("airborne")].astype(bool),
            }
        return out

    def per_drone_stats(self) -> list[dict[str, Any]]:
        """Distance flown, max altitude, battery used and final state per drone (full resolution)."""
        c = self._cols
        mode_vocab = c.vocab["mode"]
        mi, bi = CATEGORICAL.index("mode"), NUMERIC.index("battery")
        agl = NUMERIC.index("altitude_agl")
        stats = []
        for d in self.drones():
            rows = np.flatnonzero(c.ids == d["id"])
            p = c.num[rows, :3].astype(np.float64)
            dist = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0
            air = c.flags[rows, FLAGS.index("airborne")].astype(bool)
            dt = np.diff(c.t[rows], append=c.t[rows[-1]])
            stats.append({**d, "distance_m": round(dist, 1), "max_agl_m": round(float(np.nanmax(c.num[rows, agl])), 1),
                          "battery_start": round(float(c.num[rows[0], bi]), 1),
                          "battery_end": round(float(c.num[rows[-1], bi]), 1),
                          "airborne_s": round(float(np.sum(dt[air])), 1),
                          "final_mode": mode_vocab[c.cat[rows[-1], mi]]})
        return stats

    def separation(self) -> np.ndarray:
        """``[frames, 2]``: time, minimum distance between two airborne drones (NaN if fewer than two)."""
        if self._separation is None:
            from scipy.spatial import cKDTree
            c = self._cols
            ai = FLAGS.index("airborne")
            out = np.full((len(self.frame_times), 2), np.nan)
            out[:, 0] = self.frame_times
            for k in range(len(self.frame_times)):
                s = self._frame_slice(k)
                pts = c.num[s, :3][c.flags[s, ai] == 1].astype(np.float64)
                if len(pts) >= 2:
                    d, _ = cKDTree(pts).query(pts, k=2)
                    out[k, 1] = float(np.min(d[:, 1]))
            self._separation = out
        return self._separation


class ReplayStore:
    """Thread-safe LRU cache of parsed runs under one log directory.

    A run that is still being recorded grows on disk; it is re-parsed at most every ``REFRESH_S``.
    """

    REFRESH_S = 15.0

    def __init__(self, log_dir: Path, cache_runs: int = 2) -> None:
        self.log_dir = log_dir
        self.cache_runs = cache_runs
        self._cache: OrderedDict[str, tuple[float, float, RunLog]] = OrderedDict()   # id -> (mtime, parsed at, run)
        self._lock = threading.Lock()

    def runs(self) -> list[dict[str, Any]]:
        return list_runs(self.log_dir)

    def path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or not RUN_ID_RE.match(run_id):
            raise ReplayError("invalid run id")
        d = (self.log_dir / run_id).resolve()
        if d.parent != self.log_dir.resolve() or not d.is_dir():
            raise ReplayError(f"run {run_id} not found")
        return d

    def get(self, run_id: str) -> RunLog:
        d = self.path(run_id)
        tel = d / "telemetry.csv"
        stamp = tel.stat().st_mtime if tel.exists() else 0.0
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(run_id)
            if hit is not None and (hit[0] == stamp or now - hit[1] < self.REFRESH_S):
                self._cache.move_to_end(run_id)
                return hit[2]
        run = RunLog(d)                                             # parse outside the lock
        with self._lock:
            self._cache[run_id] = (stamp, now, run)
            self._cache.move_to_end(run_id)
            while len(self._cache) > self.cache_runs:
                self._cache.popitem(last=False)
        return run
