"""Mission library on disk: ``<mission.directory>/mission_<slug>.json``.

Files are validated against the mission schema on save *and* on load, written atomically, and only
names matching ``mission_[a-z0-9_-]+.json`` are ever read or written (no path traversal).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .model import Mission, MissionError, slugify

FILE_RE = re.compile(r"^mission_[a-z0-9_-]{1,60}\.json$")


class MissionStore:
    def __init__(self, directory: Path, geodetic_to_enu: Callable[[float, float], tuple[float, float]] | None = None) -> None:
        self.directory = Path(directory)
        self._geo = geodetic_to_enu

    def _path(self, filename: str) -> Path:
        if not FILE_RE.match(filename):
            raise MissionError(f"invalid mission file name {filename!r} (expected mission_<name>.json)")
        return self.directory / filename

    def list(self) -> list[dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        out = []
        for path in sorted(self.directory.glob("mission_*.json")):
            if not FILE_RE.match(path.name):
                continue
            entry: dict[str, Any] = {"file": path.name, "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")}
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                mission = Mission.from_dict(data, self._geo)
                entry.update(name=mission.name, tracks=len(mission.tracks), waypoints=mission.waypoint_count, valid=True)
            except (OSError, ValueError, MissionError) as exc:
                entry.update(valid=False, error=str(exc))
            out.append(entry)
        return out

    def save(self, data: dict[str, Any], overwrite: bool = True) -> tuple[str, Mission]:
        mission = Mission.from_dict(data, self._geo)             # schema validation before anything is written
        filename = f"mission_{slugify(mission.name)}.json"
        path = self._path(filename)
        if path.exists() and not overwrite:
            raise MissionError(f"{filename} already exists")
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = mission.to_dict()
        payload["created"] = datetime.now().isoformat(timespec="seconds")
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        return filename, mission

    def load(self, filename: str) -> Mission:
        path = self._path(filename)
        if not path.is_file():
            raise FileNotFoundError(filename)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MissionError(f"{filename} is not valid JSON: {exc}") from None
        return Mission.from_dict(data, self._geo)

    def delete(self, filename: str) -> None:
        path = self._path(filename)
        if not path.is_file():
            raise FileNotFoundError(filename)
        path.unlink()
