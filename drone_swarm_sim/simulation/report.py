"""One-click post-mission report (Stage 5): self-contained HTML and PDF, both offline.

Content: logo (or a GANDIV wordmark), run summary, top-down flight paths, battery curves, minimum
separation over time, separation violations, alerts, the event timeline (with who issued each
command) and a per-drone table.

Charts are built once as simple vector primitives (lines, text, rectangles, circles in a top-left
point coordinate system) and rendered to inline SVG for HTML or to PDF drawing operators, so both
formats show exactly the same plots. The PDF writer is a minimal PDF 1.4 generator (Helvetica,
Flate-compressed pages, optional PNG logo) - no third-party packages.
"""

from __future__ import annotations

import base64
import html
import math
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .replay import RunLog
from .replay import _started_at as started_at

PALETTE = ("#3F4A2C", "#C62828", "#1F6FB2", "#A8874A", "#7B3F99", "#2E8B57", "#D35400", "#5E5054",
           "#008B8B", "#B03060", "#556B2F", "#4169E1")
ACCENT, TEXT, DIM, GRID, PANEL = "#3F4A2C", "#3A3A3A", "#5E5054", "#E2B4BD", "#FFF5F5"
LINK_ACK_KINDS = ("command_ack", "command_lost")


# ============================================================================ report data

def _alert_priority(e: dict[str, Any]) -> str | None:
    """Same rule as simulation.alerts.classify, on a recorded event dict."""
    if e.get("category") == "COMMAND":
        return None
    sev = e.get("severity")
    return sev if sev in ("CRITICAL", "WARNING") else None


def _clock(t: float | None) -> str:
    if t is None:
        return "-"
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:04.1f}"


def build_report_data(run: RunLog, *, title: str = "Post-mission report", max_track_points: int = 600,
                      max_timeline_events: int = 200) -> dict[str, Any]:
    """Everything the renderers need (plain Python / NumPy, no formatting)."""
    world = run.world.get("world") or {}
    swarm_w = world.get("swarm") or {}
    cfg_swarm = run.config.get("swarm", {}) if isinstance(run.config, dict) else {}
    events = run.events()
    drones = run.per_drone_stats()
    sep = run.separation()
    valid = sep[~np.isnan(sep[:, 1])]
    collisions = [e for e in events if e.get("category") == "COLLISION"]
    alerts = [dict(e, priority=_alert_priority(e)) for e in events if _alert_priority(e)]
    # Operator commands (not the per-drone radio acknowledgements of the link model).
    commands = [e for e in events if e.get("category") == "COMMAND" and e.get("kind") not in LINK_ACK_KINDS]
    timeline = [e for e in events if e.get("kind") != "command_ack" and (
        e.get("category") in ("COMMAND", "MISSION") or e.get("severity") != "INFO" or e.get("kind") == "reset")]
    if len(timeline) > max_timeline_events:
        crit = [e for e in timeline if e.get("severity") == "CRITICAL"]
        rest = [e for e in timeline if e.get("severity") != "CRITICAL"]
        step = math.ceil(len(rest) / max(1, max_timeline_events - len(crit)))
        timeline = sorted(crit[:max_timeline_events] + rest[::step], key=lambda e: e.get("time", 0.0))
        timeline = timeline[:max_timeline_events]
    sources: dict[str, int] = {}
    for d in drones:
        sources[d["source"]] = sources.get(d["source"], 0) + 1
    return {
        "title": title,
        "run_id": run.run_id,
        "started_at": started_at(run.run_id),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": run.duration,
        "seed": (run.config.get("simulation") or {}).get("seed") if isinstance(run.config, dict) else None,
        "drones": drones,
        "sources": sources,
        "tracks": run.tracks(max_track_points),
        "separation": sep,
        "min_separation": float(np.min(valid[:, 1])) if len(valid) else None,
        "min_separation_t": float(valid[np.argmin(valid[:, 1]), 0]) if len(valid) else None,
        "separation_distance": swarm_w.get("separation_distance") or cfg_swarm.get("separation_distance"),
        "hard_floor": cfg_swarm.get("min_separation"),
        "battery_thresholds": world.get("battery_thresholds") or {},
        "geofence": run.world.get("geofence"),
        "bounds": world.get("bounds"),
        "collisions": collisions,
        "violations": [e for e in collisions if e.get("kind") != "collision"],
        "crashes": [e for e in collisions if e.get("kind") == "collision"],
        "alerts": alerts,
        "alert_counts": {p: sum(1 for a in alerts if a["priority"] == p) for p in ("CRITICAL", "WARNING")},
        "commands": commands,
        "users": sorted({(e.get("data") or {}).get("user") for e in commands} - {None}),
        "timeline": timeline,
        "event_count": len(events),
    }


# ============================================================================ vector charts

@dataclass
class Drawing:
    """Vector primitives in points, origin top-left."""

    width: float
    height: float
    items: list[tuple] = field(default_factory=list)

    def line(self, pts: Sequence[tuple[float, float]], color: str = TEXT, width: float = 1.0,
             dash: bool = False) -> None:
        if len(pts) >= 2:
            self.items.append(("line", [(float(x), float(y)) for x, y in pts], color, width, dash))

    def text(self, x: float, y: float, s: str, size: float = 8, color: str = TEXT, anchor: str = "start",
             bold: bool = False) -> None:
        self.items.append(("text", float(x), float(y), str(s), size, color, anchor, bold))

    def rect(self, x: float, y: float, w: float, h: float, stroke: str | None = GRID, fill: str | None = None,
             width: float = 0.8) -> None:
        self.items.append(("rect", float(x), float(y), float(w), float(h), stroke, fill, width))

    def circle(self, x: float, y: float, r: float, fill: str = TEXT) -> None:
        self.items.append(("circle", float(x), float(y), float(r), fill))


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step) * step
    return [round(start + k * step, 10) for k in range(int((hi - start) / step + 1e-9) + 1)]


def _fmt_tick(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 10 or v == int(v) else f"{v:.1f}"


def line_chart(width: float, height: float, series: list[tuple[str, str, np.ndarray, np.ndarray]], *,
               title: str, y_label: str, y_min: float | None = None, y_max: float | None = None,
               hlines: Iterable[tuple[float, str, str]] = (), x_is_time: bool = True) -> Drawing:
    """Time-series chart: ``series = [(label, color, x, y)]``; ``hlines = [(value, color, label)]``."""
    dr = Drawing(width, height)
    left, right, top, bottom = 40, 10, 18, 24
    pw, ph = width - left - right, height - top - bottom
    xs = [s[2] for s in series if len(s[2])]
    ys = [s[3][np.isfinite(s[3])] for s in series if len(s[3])]
    ys = [y for y in ys if len(y)]
    x0 = min((float(np.min(x)) for x in xs), default=0.0)
    x1 = max((float(np.max(x)) for x in xs), default=1.0)
    hl = list(hlines)
    lo = y_min if y_min is not None else min([float(np.min(y)) for y in ys] + [v for v, _, _ in hl], default=0.0)
    hi = y_max if y_max is not None else max([float(np.max(y)) for y in ys] + [v for v, _, _ in hl], default=1.0)
    if hi <= lo:
        hi = lo + 1.0
    if y_max is None:
        hi += 0.05 * (hi - lo)
    if x1 <= x0:
        x1 = x0 + 1.0
    sx = lambda v: left + (v - x0) / (x1 - x0) * pw          # noqa: E731
    sy = lambda v: top + ph - (v - lo) / (hi - lo) * ph      # noqa: E731
    dr.text(left, 11, title, 9, TEXT, bold=True)
    dr.rect(left, top, pw, ph, GRID, "#FFFFFF")
    for v in _nice_ticks(lo, hi, 4):
        dr.line([(left, sy(v)), (left + pw, sy(v))], "#F0DADA", 0.5)
        dr.text(left - 4, sy(v) + 3, _fmt_tick(v), 7, DIM, "end")
    for v in _nice_ticks(x0, x1, 6):
        dr.line([(sx(v), top + ph), (sx(v), top + ph + 3)], DIM, 0.5)
        dr.text(sx(v), top + ph + 12, _clock(v)[:-2] if x_is_time else _fmt_tick(v), 7, DIM, "middle")
    dr.text(left + pw, height - 2, "mission time (mm:ss)" if x_is_time else "", 7, DIM, "end")
    dr.text(4, top - 5, y_label, 7, DIM)
    for value, color, label in hl:
        if lo <= value <= hi:
            dr.line([(left, sy(value)), (left + pw, sy(value))], color, 0.9, dash=True)
            dr.text(left + pw - 2, sy(value) - 2, label, 6.5, color, "end")
    for _, color, x, y in series:
        seg: list[tuple[float, float]] = []
        for xv, yv in zip(x, y):
            if np.isfinite(yv):
                seg.append((sx(xv), sy(min(max(yv, lo), hi))))
            else:
                dr.line(seg, color, 1.1)
                seg = []
        dr.line(seg, color, 1.1)
    return dr


def path_plot(width: float, height: float, data: dict[str, Any]) -> Drawing:
    """Top-down flight paths (East right, North up) with equal axis scaling, geofence and start/end marks."""
    dr = Drawing(width, height)
    tracks = data["tracks"]
    pts = [np.column_stack([t["x"], t["y"]]) for t in tracks.values() if len(t["x"])]
    fence = data.get("geofence") or {}
    poly = fence.get("inclusion") or []
    allp = np.vstack(pts + ([np.asarray(poly, dtype=float)[:, :2]] if poly else [])) if pts else np.zeros((1, 2))
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    span = np.maximum(hi - lo, 20.0)
    centre = (lo + hi) / 2
    left, right, top, bottom = 40, 10, 18, 24
    pw, ph = width - left - right, height - top - bottom
    scale = min(pw / (span[0] * 1.1), ph / (span[1] * 1.1))
    sx = lambda v: left + pw / 2 + (v - centre[0]) * scale      # noqa: E731
    sy = lambda v: top + ph / 2 - (v - centre[1]) * scale       # noqa: E731
    dr.text(left, 11, "Flight paths (top view, metres ENU)", 9, TEXT, bold=True)
    dr.rect(left, top, pw, ph, GRID, "#FFFFFF")
    for v in _nice_ticks(centre[0] - pw / 2 / scale, centre[0] + pw / 2 / scale, 6):
        dr.line([(sx(v), top), (sx(v), top + ph)], "#F3E4E4", 0.5)
        dr.text(sx(v), top + ph + 12, _fmt_tick(v), 7, DIM, "middle")
    for v in _nice_ticks(centre[1] - ph / 2 / scale, centre[1] + ph / 2 / scale, 5):
        dr.line([(left, sy(v)), (left + pw, sy(v))], "#F3E4E4", 0.5)
        dr.text(left - 4, sy(v) + 3, _fmt_tick(v), 7, DIM, "end")
    dr.text(left + pw, height - 2, "East (m)", 7, DIM, "end")
    dr.text(4, top - 5, "North (m)", 7, DIM)
    if poly:
        ring = [(sx(p[0]), sy(p[1])) for p in poly] + [(sx(poly[0][0]), sy(poly[0][1]))]
        dr.line(ring, "#A8874A", 1.2, dash=True)
    for zone in fence.get("exclusions") or []:
        zp = zone.get("polygon") if isinstance(zone, dict) else zone
        if zp:
            dr.line([(sx(p[0]), sy(p[1])) for p in zp] + [(sx(zp[0][0]), sy(zp[0][1]))], "#C62828", 1.0, dash=True)
    for k, (did, t) in enumerate(tracks.items()):
        color = PALETTE[k % len(PALETTE)]
        if not len(t["x"]):
            continue
        dr.line([(sx(x), sy(y)) for x, y in zip(t["x"], t["y"])], color, 1.2)
        dr.circle(sx(t["x"][0]), sy(t["y"][0]), 2.2, color)
        dr.rect(sx(t["x"][-1]) - 2.2, sy(t["y"][-1]) - 2.2, 4.4, 4.4, None, color)
    return dr


def legend(width: float, names: list[str]) -> Drawing:
    per_row = max(1, int(width // 62))
    rows = math.ceil(len(names) / per_row) if names else 0
    dr = Drawing(width, 12 * rows + 4)
    for k, name in enumerate(names):
        x, y = 40 + (k % per_row) * 62, 8 + (k // per_row) * 12
        dr.rect(x, y - 6, 9, 6, None, PALETTE[k % len(PALETTE)])
        dr.text(x + 12, y, name, 7, TEXT)
    return dr


def build_charts(data: dict[str, Any], width: float = 515) -> dict[str, Drawing]:
    tracks = data["tracks"]
    names = [d["name"] for d in data["drones"]]
    th = data["battery_thresholds"]
    battery = line_chart(width, 190, [(n, PALETTE[k % len(PALETTE)], t["t"], t["battery"])
                                      for k, (n, t) in enumerate(zip(names, tracks.values()))],
                         title="Battery", y_label="%", y_min=0, y_max=100,
                         hlines=[(th[k], c, k.replace("_", " ")) for k, c in
                                 (("warning", "#B7860B"), ("return_home", "#D35400"), ("emergency", "#C62828")) if k in th])
    sep = data["separation"]
    hl = []
    if data.get("separation_distance"):
        hl.append((float(data["separation_distance"]), "#B7860B", "separation"))
    if data.get("hard_floor"):
        hl.append((float(data["hard_floor"]), "#C62828", "hard floor"))
    separation = line_chart(width, 170, [("min", TEXT, sep[:, 0], sep[:, 1])], title="Minimum separation (airborne)",
                            y_label="m", y_min=0, hlines=hl)
    altitude = line_chart(width, 170, [(n, PALETTE[k % len(PALETTE)], t["t"], t["agl"])
                                       for k, (n, t) in enumerate(zip(names, tracks.values()))],
                          title="Altitude above ground", y_label="m", y_min=0)
    return {"paths": path_plot(width, 330, data), "legend": legend(width, names), "battery": battery,
            "altitude": altitude, "separation": separation}


# ============================================================================ SVG / HTML

def to_svg(dr: Drawing) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {dr.width:.0f} {dr.height:.0f}" '
           f'width="100%" role="img" font-family="Helvetica, Arial, sans-serif">']
    for it in dr.items:
        kind = it[0]
        if kind == "line":
            _, pts, color, width, dash = it
            d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            dash_attr = ' stroke-dasharray="4 3"' if dash else ""
            out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
                       f'{dash_attr} stroke-linejoin="round"/>')
        elif kind == "text":
            _, x, y, s, size, color, anchor, bold = it
            weight = ' font-weight="bold"' if bold else ""
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'
                       f'{weight}>{html.escape(s)}</text>')
        elif kind == "rect":
            _, x, y, w, h, stroke, fill, width = it
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill or "none"}" '
                       f'stroke="{stroke or "none"}" stroke-width="{width}"/>')
        elif kind == "circle":
            _, x, y, r, fill = it
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}"/>')
    out.append("</svg>")
    return "".join(out)


def _logo_data_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".svg": "image/svg+xml"}.get(
        path.suffix.lower())
    if mime is None:
        return None
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _summary_rows(data: dict[str, Any]) -> list[tuple[str, str]]:
    drones = data["drones"]
    src = ", ".join(f"{n} {k}" for k, n in sorted(data["sources"].items()))
    ms = data["min_separation"]
    return [
        ("Run", data["run_id"]),
        ("Started", (data["started_at"] or "-").replace("T", " ")),
        ("Duration", _clock(data["duration_s"])),
        ("Drones", f"{len(drones)} ({src})" if drones else "0"),
        ("Distance flown (all drones)", f"{sum(d['distance_m'] for d in drones) / 1000:.2f} km"),
        ("Max altitude AGL", f"{max((d['max_agl_m'] for d in drones), default=0):.1f} m"),
        ("Lowest battery at end", f"{min((d['battery_end'] for d in drones), default=0):.0f} %"),
        ("Minimum separation", "-" if ms is None else f"{ms:.2f} m at {_clock(data['min_separation_t'])}"),
        ("Separation violations", str(len(data["violations"]))),
        ("Collisions", str(len(data["crashes"]))),
        ("Alerts", f"{data['alert_counts']['CRITICAL']} critical, {data['alert_counts']['WARNING']} warning"),
        ("Operator commands", f"{len(data['commands'])}" + (f" by {', '.join(data['users'])}" if data["users"] else "")),
        ("Seed", str(data["seed"])),
    ]


def _event_cells(e: dict[str, Any]) -> list[str]:
    did = e.get("drone_id")
    data = e.get("data") or {}
    who = data.get("user") or ""
    drones = f"D{did:02d}" if isinstance(did, int) else ""
    if not drones and data.get("drone_a") is not None:
        drones = f"D{data['drone_a']:02d}/D{data.get('drone_b', 0):02d}"
    return [_clock(e.get("time")), e.get("severity", ""), e.get("category", ""), drones, who, e.get("message", "")]


def render_html(data: dict[str, Any], logo: Path | None = None) -> str:
    charts = build_charts(data, 760)
    uri = _logo_data_uri(logo)
    brand = (f'<img src="{uri}" alt="GANDIV" class="logo">' if uri else
             '<div class="wordmark">GANDIV</div>')
    esc = html.escape

    def table(head: list[str], rows: list[list[str]], empty: str) -> str:
        if not rows:
            return f'<p class="empty">{esc(empty)}</p>'
        th = "".join(f"<th>{esc(h)}</th>" for h in head)
        body = "".join("<tr>" + "".join(f"<td>{esc(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"

    summary = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in _summary_rows(data))
    per_drone = [[d["name"], d["source"], f"{d['distance_m']:.0f}", f"{d['max_agl_m']:.1f}", _clock(d["airborne_s"]),
                  f"{d['battery_start']:.0f} → {d['battery_end']:.0f}", d["final_mode"]] for d in data["drones"]]
    viol = [_event_cells(e) for e in data["collisions"]]
    alerts = [[_clock(a.get("time")), a["priority"], a.get("kind", ""), _event_cells(a)[3], a.get("message", "")]
              for a in data["alerts"]]
    timeline = [_event_cells(e) for e in data["timeline"]]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(data['title'])} - {esc(data['run_id'])}</title>
<style>
:root {{ --bg:{PANEL}; --panel:#F7D6D0; --border:{GRID}; --text:#4A4A4A; --strong:{TEXT}; --accent:{ACCENT}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:13px/1.45 Inter, "Segoe UI", system-ui, sans-serif; }}
main {{ max-width:840px; margin:0 auto; padding:24px 16px 48px; }}
header {{ display:flex; align-items:center; gap:16px; border-bottom:3px solid var(--accent); padding-bottom:12px; }}
.logo {{ height:52px; }} .wordmark {{ font:800 28px/1 Inter, sans-serif; letter-spacing:.12em; color:var(--accent); }}
h1 {{ margin:0; font-size:20px; color:var(--strong); }} .sub {{ color:#5E5054; font-size:12px; }}
h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.06em; color:var(--accent); margin:26px 0 8px;
      border-bottom:1px solid var(--border); padding-bottom:4px; }}
table {{ width:100%; border-collapse:collapse; background:#fff; font-size:12px; }}
th, td {{ text-align:left; padding:4px 8px; border-bottom:1px solid #F0DADA; vertical-align:top; }}
thead th {{ background:var(--panel); color:var(--strong); }}
table.kv th {{ width:38%; background:#FBEAEA; font-weight:600; }}
.chart {{ background:#fff; border:1px solid var(--border); padding:6px; margin:8px 0; }}
.empty {{ color:#5E5054; font-style:italic; }}
footer {{ margin-top:32px; color:#5E5054; font-size:11px; }}
@media print {{ body {{ background:#fff; }} h2 {{ break-after:avoid; }} .chart, tr {{ break-inside:avoid; }} }}
</style></head><body><main>
<header>{brand}<div><h1>{esc(data['title'])}</h1>
<div class="sub">Run {esc(data['run_id'])} · generated {esc(data['generated_at'].replace('T', ' '))}</div></div></header>
<h2>Summary</h2><table class="kv">{summary}</table>
<h2>Flight paths</h2><div class="chart">{to_svg(charts['paths'])}{to_svg(charts['legend'])}</div>
<h2>Battery</h2><div class="chart">{to_svg(charts['battery'])}</div>
<h2>Altitude</h2><div class="chart">{to_svg(charts['altitude'])}</div>
<h2>Separation</h2><div class="chart">{to_svg(charts['separation'])}</div>
<h2>Separation violations ({len(viol)})</h2>
{table(["Time", "Severity", "Category", "Drones", "", "Message"], viol, "No separation violations recorded.")}
<h2>Alerts ({len(alerts)})</h2>
{table(["Time", "Priority", "Kind", "Drone", "Message"], alerts, "No alerts raised.")}
<h2>Event timeline</h2>
{table(["Time", "Severity", "Category", "Drone", "User", "Event"], timeline, "No events recorded.")}
<h2>Drones</h2>
{table(["Drone", "Source", "Distance m", "Max AGL m", "Airborne", "Battery %", "Final mode"], per_drone, "No drones.")}
<footer>GANDIV Swarm GCS · {data['event_count']} events recorded · report generated offline from logs/{esc(data['run_id'])}</footer>
</main></body></html>"""


# ============================================================================ PDF

def _rgb(color: str) -> tuple[float, float, float]:
    c = color.lstrip("#")
    return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _pdf_str(s: str) -> str:
    s = s.replace("→", "->").replace("·", "-").replace("—", "-").replace("–", "-")
    s = s.encode("cp1252", "replace").decode("latin-1")
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_width(s: str, size: float, bold: bool = False) -> float:
    """Approximate Helvetica advance width (good enough for alignment and clipping)."""
    narrow = sum(1 for ch in s if ch in "iljtfI.,:;'!|() -")
    wide = sum(1 for ch in s if ch in "MWmw@%")
    caps = sum(1 for ch in s if ch.isupper() and ch not in "IMW")
    other = len(s) - narrow - wide - caps
    return size * (0.28 * narrow + 0.85 * wide + 0.68 * caps + 0.54 * other) * (1.05 if bold else 1.0)


def _png_for_pdf(path: Path | None) -> dict[str, Any] | None:
    """An 8-bit, non-interlaced grey / RGB PNG can go into a PDF as-is (Flate + PNG predictors)."""
    if path is None or not path.is_file() or path.suffix.lower() != ".png":
        return None
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos, idat, hdr = 8, bytearray(), None
    while pos < len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            hdr = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat.extend(chunk)
    if hdr is None:
        return None
    w, h, depth, color, _, _, interlace = hdr
    if depth != 8 or interlace or color not in (0, 2):
        return None                        # alpha / palette PNGs: the PDF shows the wordmark instead
    colors = 1 if color == 0 else 3
    return {"w": w, "h": h, "colors": colors, "data": bytes(idat)}


class _PdfPage:
    def __init__(self) -> None:
        self.ops: list[str] = []


class PdfDocument:
    """Tiny flowing-layout PDF writer (A4, Helvetica)."""

    W, H, MARGIN = 595.0, 842.0, 40.0

    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: list[_PdfPage] = []
        self.image: dict[str, Any] | None = None
        self.y = 0.0
        self.footer = ""
        self.new_page()

    # -- page / cursor
    def new_page(self) -> None:
        self.page = _PdfPage()
        self.pages.append(self.page)
        self.y = self.MARGIN

    def ensure(self, h: float) -> None:
        if self.y + h > self.H - self.MARGIN - 14:
            self.new_page()

    # -- primitives (top-left coordinates)
    def _color(self, color: str, stroke: bool) -> str:
        r, g, b = _rgb(color)
        return f"{r:.3f} {g:.3f} {b:.3f} {'RG' if stroke else 'rg'}"

    def text(self, x: float, y: float, s: str, size: float = 9, color: str = TEXT, bold: bool = False,
             anchor: str = "start") -> None:
        if anchor != "start":
            w = _text_width(s, size, bold)
            x -= w if anchor == "end" else w / 2
        self.page.ops.append(f"BT {self._color(color, False)} /{'F2' if bold else 'F1'} {size:.1f} Tf "
                             f"{x:.2f} {self.H - y:.2f} Td ({_pdf_str(s)}) Tj ET")

    def draw(self, dr: Drawing, x0: float, y0: float) -> None:
        ops = self.page.ops
        ops.append("q 1 J 1 j")
        for it in dr.items:
            kind = it[0]
            if kind == "line":
                _, pts, color, width, dash = it
                path = " ".join(f"{x0 + x:.2f} {self.H - (y0 + y):.2f} {'m' if k == 0 else 'l'}"
                                for k, (x, y) in enumerate(pts))
                ops.append(f"{self._color(color, True)} {width:.2f} w {'[3 2] 0 d' if dash else '[] 0 d'} {path} S")
            elif kind == "text":
                _, x, y, s, size, color, anchor, bold = it
                self.text(x0 + x, y0 + y, s, size, color, bold, anchor)
            elif kind == "rect":
                _, x, y, w, h, stroke, fill, width = it
                box = f"{x0 + x:.2f} {self.H - (y0 + y + h):.2f} {w:.2f} {h:.2f} re"
                if fill and stroke:
                    ops.append(f"{self._color(fill, False)} {self._color(stroke, True)} {width:.2f} w [] 0 d {box} B")
                elif fill:
                    ops.append(f"{self._color(fill, False)} {box} f")
                elif stroke:
                    ops.append(f"{self._color(stroke, True)} {width:.2f} w [] 0 d {box} S")
            elif kind == "circle":
                _, x, y, r, fill = it
                cx, cy, k = x0 + x, self.H - (y0 + y), 0.5523 * r
                ops.append(f"{self._color(fill, False)} {cx + r:.2f} {cy:.2f} m "
                           f"{cx + r:.2f} {cy + k:.2f} {cx + k:.2f} {cy + r:.2f} {cx:.2f} {cy + r:.2f} c "
                           f"{cx - k:.2f} {cy + r:.2f} {cx - r:.2f} {cy + k:.2f} {cx - r:.2f} {cy:.2f} c "
                           f"{cx - r:.2f} {cy - k:.2f} {cx - k:.2f} {cy - r:.2f} {cx:.2f} {cy - r:.2f} c "
                           f"{cx + k:.2f} {cy - r:.2f} {cx + r:.2f} {cy - k:.2f} {cx + r:.2f} {cy:.2f} c f")
        ops.append("Q")

    # -- flowing blocks
    def heading(self, s: str, keep: float = 30.0) -> None:
        """Section heading; ``keep`` = height of the block that must follow on the same page."""
        self.ensure(28 + keep)
        self.y += 16
        self.text(self.MARGIN, self.y, s.upper(), 10.5, ACCENT, bold=True)
        self.page.ops.append(f"{self._color(GRID, True)} 0.8 w [] 0 d {self.MARGIN:.1f} {self.H - self.y - 4:.1f} m "
                             f"{self.W - self.MARGIN:.1f} {self.H - self.y - 4:.1f} l S")
        self.y += 12

    def paragraph(self, s: str, size: float = 9, color: str = DIM) -> None:
        self.ensure(size + 4)
        self.y += size + 3
        self.text(self.MARGIN, self.y, s, size, color)

    def drawing(self, dr: Drawing) -> None:
        self.ensure(dr.height + 6)
        self.draw(dr, self.MARGIN, self.y)
        self.y += dr.height + 6

    def table(self, head: list[str], rows: list[list[str]], widths: list[float], empty: str,
              size: float = 7.5) -> None:
        if not rows:
            self.paragraph(empty)
            return
        total = self.W - 2 * self.MARGIN
        widths = [w * total / sum(widths) for w in widths]
        line_h = size + 5

        def row(cells: list[str], bold: bool, fill: str | None) -> None:
            self.ensure(line_h)
            if fill:
                self.page.ops.append(f"{self._color(fill, False)} {self.MARGIN:.2f} {self.H - self.y - line_h:.2f} "
                                     f"{total:.2f} {line_h:.2f} re f")
            x = self.MARGIN
            for c, w in zip(cells, widths):
                s = str(c)
                while s and _text_width(s, size, bold) > w - 6:
                    s = s[:-2] + "…" if len(s) > 2 else ""
                self.text(x + 3, self.y + line_h - 3.5, s, size, TEXT, bold)
                x += w
            self.y += line_h

        row(head, True, "#F7D6D0")
        for k, r in enumerate(rows):
            if self.y + line_h > self.H - self.MARGIN - 14:
                self.new_page()
                row(head, True, "#F7D6D0")
            row(r, False, "#FBEAEA" if k % 2 else None)

    # -- output
    def to_bytes(self) -> bytes:
        objects: list[bytes] = []

        def add(body: bytes) -> int:
            objects.append(body)
            return len(objects)

        catalog = add(b"")                         # placeholders, filled once the page ids are known
        pages_id = add(b"")
        f1 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        f2 = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        img_id = None
        if self.image is not None:
            im = self.image
            head = (f"<< /Type /XObject /Subtype /Image /Width {im['w']} /Height {im['h']} "
                    f"/ColorSpace /{'DeviceGray' if im['colors'] == 1 else 'DeviceRGB'} /BitsPerComponent 8 "
                    f"/Filter /FlateDecode /DecodeParms << /Predictor 15 /Colors {im['colors']} /BitsPerComponent 8 "
                    f"/Columns {im['w']} >> /Length {len(im['data'])} >>").encode("ascii")
            img_id = add(head + b"\nstream\n" + im["data"] + b"\nendstream")
        page_ids = []
        n = len(self.pages)
        for k, page in enumerate(self.pages):
            foot = (f"BT {self._color(DIM, False)} /F1 7 Tf {self.MARGIN:.1f} 22 Td "
                    f"({_pdf_str(self.footer)}) Tj ET BT /F1 7 Tf {self.W - self.MARGIN - 40:.1f} 22 Td "
                    f"(page {k + 1} / {n}) Tj ET")
            stream = zlib.compress(("\n".join(page.ops + [foot])).encode("latin-1", "replace"))
            content = add(f"<< /Length {len(stream)} /Filter /FlateDecode >>".encode("ascii")
                          + b"\nstream\n" + stream + b"\nendstream")
            xobj = f" /XObject << /Im1 {img_id} 0 R >>" if img_id else ""
            page_ids.append(add(f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {self.W:.0f} {self.H:.0f}] "
                                f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >>{xobj} >> "
                                f"/Contents {content} 0 R >>".encode("ascii")))
        objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii")
        kids = " ".join(f"{i} 0 R" for i in page_ids)
        objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
        info = add(f"<< /Title ({_pdf_str(self.title)}) /Producer (GANDIV Swarm GCS) >>".encode("latin-1"))
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for i, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode("ascii")
        out += (f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode("ascii")
        return bytes(out)


def render_pdf(data: dict[str, Any], logo: Path | None = None) -> bytes:
    doc = PdfDocument(f"{data['title']} - {data['run_id']}")
    doc.footer = f"GANDIV Swarm GCS - run {data['run_id']} - generated {data['generated_at'].replace('T', ' ')}"
    image = _png_for_pdf(logo)
    m = doc.MARGIN
    if image is not None:
        doc.image = image
        h = 42.0
        w = h * image["w"] / image["h"]
        doc.page.ops.append(f"q {w:.2f} 0 0 {h:.2f} {m:.2f} {doc.H - m - h:.2f} cm /Im1 Do Q")
        tx = m + w + 12
    else:
        doc.text(m, m + 30, "GANDIV", 26, ACCENT, bold=True)
        tx = m + _text_width("GANDIV", 26, True) + 16
    doc.text(tx, m + 18, data["title"], 16, TEXT, bold=True)
    doc.text(tx, m + 34, f"Run {data['run_id']}  -  generated {data['generated_at'].replace('T', ' ')}", 8.5, DIM)
    doc.page.ops.append(f"{doc._color(ACCENT, True)} 2 w {m:.1f} {doc.H - m - 50:.1f} m {doc.W - m:.1f} "
                        f"{doc.H - m - 50:.1f} l S")
    doc.y = m + 56
    charts = build_charts(data, doc.W - 2 * m)
    doc.heading("Summary")
    doc.table(["Item", "Value"], [list(r) for r in _summary_rows(data)], [0.38, 0.62], "")
    doc.heading("Flight paths", charts["paths"].height)
    doc.drawing(charts["paths"])
    doc.drawing(charts["legend"])
    doc.heading("Battery", charts["battery"].height)
    doc.drawing(charts["battery"])
    doc.heading("Altitude and separation", charts["altitude"].height)
    doc.drawing(charts["altitude"])
    doc.drawing(charts["separation"])
    doc.heading(f"Separation violations ({len(data['collisions'])})")
    doc.table(["Time", "Severity", "Category", "Drones", "User", "Message"],
              [_event_cells(e) for e in data["collisions"]], [0.09, 0.1, 0.12, 0.12, 0.07, 0.5],
              "No separation violations recorded.")
    doc.heading(f"Alerts ({len(data['alerts'])})")
    doc.table(["Time", "Priority", "Kind", "Drone", "Message"],
              [[_clock(a.get("time")), a["priority"], a.get("kind", ""), _event_cells(a)[3], a.get("message", "")]
               for a in data["alerts"]], [0.09, 0.1, 0.16, 0.1, 0.55], "No alerts raised.")
    doc.heading("Event timeline")
    doc.table(["Time", "Severity", "Category", "Drone", "User", "Event"],
              [_event_cells(e) for e in data["timeline"]], [0.09, 0.1, 0.12, 0.08, 0.1, 0.51], "No events recorded.")
    doc.heading("Drones")
    doc.table(["Drone", "Source", "Distance m", "Max AGL m", "Airborne", "Battery %", "Final mode"],
              [[d["name"], d["source"], f"{d['distance_m']:.0f}", f"{d['max_agl_m']:.1f}", _clock(d["airborne_s"]),
                f"{d['battery_start']:.0f} -> {d['battery_end']:.0f}", d["final_mode"]] for d in data["drones"]],
              [0.12, 0.12, 0.14, 0.14, 0.14, 0.16, 0.18], "No drones.")
    return doc.to_bytes()
