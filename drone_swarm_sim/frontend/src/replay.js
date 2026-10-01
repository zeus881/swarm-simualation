// REPLAY tab (Stage 5): plays a recorded run (logs/<run_id>) back through the normal FLY rendering.
//
// Frames come from /api/replay/runs/<id>/frames in windows of `chunk_s` seconds (the current window plus
// the next one are kept, older ones dropped), are turned into telemetry-shaped snapshots and handed to
// Dashboard.onTelemetry - so the 3D view, fleet list, inspector, HUD, charts and minimap all work in
// replay exactly as they do live. Live telemetry keeps arriving meanwhile and is shown again on exit.
import { auth } from './auth.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
export const REPLAY_SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const RENDER_PERIOD_S = 0.05;          // hand at most 20 snapshots per second to the dashboard

function clock(t) {
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, '0')}:${(t - m * 60).toFixed(1).padStart(4, '0')}`;
}

function markerColor(e) {
  if (e.severity === 'CRITICAL') return '#C62828';
  if (e.severity === 'WARNING') return '#B7860B';
  if (e.category === 'COMMAND') return '#3F4A2C';
  return '#1F6FB2';
}

export class ReplayController {
  constructor(ui) {
    this.ui = ui;
    this.active = false;
    this.meta = null;
    this.time = 0;
    this.speed = 1;
    this.playing = false;
    this.chunks = new Map();          // chunk index -> Promise<frames[]> | frames[]
    this.lastLive = null;
    this._renderClock = 0;
    this._shownT = null;
    this._eventCursor = 0;
    this._lowest = null;
    this._wire();
  }

  // ------------------------------------------------------------------ lifecycle
  async enter() {
    this.active = true;
    document.body.classList.add('replaying');
    $('replay-bar').hidden = false;
    this.ui.history.reset();
    await this.refreshRuns();
    if (!this.meta && $('rp-run').value) await this.load($('rp-run').value);
    else if (this.meta) this._render(true);
  }

  exit() {
    this.pause();
    this.active = false;
    document.body.classList.remove('replaying');
    $('replay-bar').hidden = true;
    this.ui.history.reset();
    if (this.lastLive) this.ui.onTelemetry(this.lastLive);
  }

  onLive(snap) { this.lastLive = snap; this.ui.lastLiveRunId = snap.run_id; }

  async refreshRuns() {
    let runs = [];
    try {
      const res = await fetch('/api/replay/runs');
      if (res.ok) runs = await res.json();
    } catch { /* offline: keep the list */ }
    const sel = $('rp-run');
    const keep = this.meta?.run_id ?? sel.value;
    sel.innerHTML = runs.length ? runs.map((r) => `<option value="${esc(r.run_id)}"${r.has_telemetry ? '' : ' disabled'}>`
      + `${esc(r.run_id)} · ${clock(r.duration_s)} · ${r.drones} drones${r.current ? ' (recording)' : ''}${r.has_telemetry ? '' : ' (no telemetry)'}</option>`).join('')
      : '<option value="">No recorded runs</option>';
    const firstPlayable = runs.find((r) => r.has_telemetry && !r.current) ?? runs.find((r) => r.has_telemetry);
    sel.value = runs.some((r) => r.run_id === keep) ? keep : (firstPlayable?.run_id ?? '');
  }

  async load(runId) {
    if (!runId) return;
    this.pause();
    const res = await fetch(`/api/replay/runs/${encodeURIComponent(runId)}`);
    if (!res.ok) { this.ui.toast(`Replay: ${(await res.json().catch(() => ({}))).detail ?? res.status}`, 'error'); return; }
    this.meta = await res.json();
    this.chunks.clear();
    this.homes = new Map();
    this._lowest = null;
    const n = this.meta.columns.numeric;
    this._idx = { n: n.length, c: this.meta.columns.categorical.length, pos: n.indexOf('x') };
    $('rp-scrub').max = String(this.meta.duration_s);
    $('rp-total').textContent = clock(this.meta.duration_s);
    this._drawMarkers();
    await this._homeFrame();
    this.seek(0);
  }

  // ------------------------------------------------------------------ transport
  play() {
    if (!this.meta) return;
    if (this.time >= this.meta.duration_s - 1e-3) this.seek(0);
    this.playing = true;
    this._syncButtons();
  }

  pause() { this.playing = false; this._syncButtons(); }

  toggle() { if (this.playing) this.pause(); else this.play(); }

  setSpeed(s) {
    this.speed = s;
    $('rp-speed').value = String(s);
  }

  seek(t) {
    if (!this.meta) return;
    const back = t < this.time - 1e-6;
    this.time = Math.max(0, Math.min(this.meta.duration_s, t));
    if (back || t === 0) {
      this.ui.resetEvents();
      this.ui.history.reset();
      this._eventCursor = 0;
      this._lowest = null;
    }
    this._render(true);
  }

  step(dt) { this.seek(this.time + dt); }

  animate(dt) {
    if (!this.active || !this.meta) return;
    if (this.playing) {
      this.time += dt * this.speed;
      if (this.time >= this.meta.duration_s) { this.time = this.meta.duration_s; this.pause(); }
      this._renderClock += dt;
      if (this._renderClock >= RENDER_PERIOD_S) { this._renderClock = 0; this._render(false); }
    }
    if (document.activeElement !== $('rp-scrub')) $('rp-scrub').value = String(this.time);
    $('rp-time').textContent = clock(this.time);
  }

  // ------------------------------------------------------------------ frames
  _chunkIndex(t) { return Math.floor(t / this.meta.chunk_s); }

  _chunk(k) {
    if (k < 0 || k * this.meta.chunk_s > this.meta.duration_s + 1e-6) return null;
    let c = this.chunks.get(k);
    if (!c) {
      const cs = this.meta.chunk_s;
      const id = encodeURIComponent(this.meta.run_id);
      c = fetch(`/api/replay/runs/${id}/frames?start=${(k * cs).toFixed(3)}&end=${((k + 1) * cs).toFixed(3)}`)
        .then((r) => (r.ok ? r.json() : { frames: [] }))
        .then((body) => { this.chunks.set(k, body.frames); return body.frames; })
        .catch(() => { this.chunks.delete(k); return []; });
      this.chunks.set(k, c);
    }
    return c;
  }

  async _homeFrame() {
    // Home pads: first recorded position of every drone.
    const frames = await this._chunk(0);
    for (const f of frames ?? []) {
      for (const r of f.rows) {
        if (!this.homes.has(r[0])) this.homes.set(r[0], { x: r[1 + this._idx.pos], y: r[2 + this._idx.pos], z: 0 });
      }
    }
  }

  _frameAt(t) {
    const k = this._chunkIndex(t);
    for (const kk of [...this.chunks.keys()]) if (kk < k - 1 || kk > k + 1) this.chunks.delete(kk);
    const c = this._chunk(k);
    if (t > (k + 0.5) * this.meta.chunk_s) this._chunk(k + 1);          // prefetch
    if (!Array.isArray(c)) return null;
    let lo = 0; let hi = c.length - 1; let best = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (c[mid].t <= t + 1e-6) { best = c[mid]; lo = mid + 1; } else hi = mid - 1;
    }
    return best ?? c[0] ?? null;
  }

  async _render(force) {
    let frame = this._frameAt(this.time);
    if (!frame && force) {
      await this._chunk(this._chunkIndex(this.time));
      frame = this._frameAt(this.time);
    }
    if (!frame || (!force && frame.t === this._shownT)) return;
    this._shownT = frame.t;
    this.ui.onTelemetry(this._snapshot(frame));
  }

  _snapshot(frame) {
    const m = this.meta;
    const { n, c } = this._idx;
    const cols = m.columns;
    const drones = frame.rows.map((r) => {
      const v = {};
      cols.numeric.forEach((k, i) => { v[k] = r[1 + i]; });
      cols.categorical.forEach((k, i) => { v[k] = m.vocab[k][r[1 + n + i]]; });
      cols.flags.forEach((k, i) => { v[k] = !!r[1 + n + c + i]; });
      return {
        drone_id: r[0], name: v.name, source: v.source,
        position: { x: v.x, y: v.y, z: v.z }, gps: { lat: v.lat, lon: v.lon, alt: v.alt },
        altitude_agl: v.altitude_agl, velocity: { x: v.vx, y: v.vy, z: v.vz }, acceleration: null,
        speed: v.speed, heading: v.heading, roll: v.roll, pitch: v.pitch, yaw_rate: null,
        battery: v.battery, battery_state: v.battery_state || 'NORMAL', battery_voltage: null, power_w: null,
        mode: v.mode, armed: v.armed, airborne: v.airborne, health: v.health, communication: v.communication,
        task: v.task, target: null, home: this.homes.get(r[0]) ?? { x: v.x, y: v.y, z: 0 }, neighbors: [],
        collision_state: v.collision_state, nearest_distance: v.nearest_distance, flight_time: null,
        distance_travelled: null, link_quality: null, gps_fix: '—', satellites: null, hdop: null,
        time_left_s: null, telemetry_age_s: null, est_position: null, pos_error: null, pos_sigma: null, failures: [],
      };
    });
    // Swarm summary for the panels (minimum separation among airborne drones, O(n²) is fine at replay rates).
    const air = drones.filter((d) => d.airborne);
    let minSep = null;
    for (let i = 0; i < air.length; i += 1) {
      for (let j = i + 1; j < air.length; j += 1) {
        const a = air[i].position; const b = air[j].position;
        const d = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
        if (minSep === null || d < minSep) minSep = d;
      }
    }
    if (minSep !== null) this._lowest = this._lowest === null ? minSep : Math.min(this._lowest, minSep);
    const batt = drones.map((d) => d.battery).filter((b) => b != null);
    const past = m.events.filter((e) => e.time <= frame.t + 1e-6);
    const fresh = past.slice(this._eventCursor);
    this._eventCursor = past.length;
    return {
      type: 'telemetry', replay: true, run_id: `replay:${m.run_id}`, sim_time: frame.t, tick: 0, state: 'REPLAY',
      stats: { sim_rate_hz: 0, target_rate_hz: 0, step_ms: 0 },
      world: m.world ?? this.lastLive?.world ?? this.ui.snapshot?.world,
      wind: { enabled: false, speed: 0, direction: 0 },
      summary: {
        total: drones.length, airborne: air.length, min_separation: minSep, lowest_separation: this._lowest,
        average_battery: batt.length ? batt.reduce((a, b) => a + b, 0) / batt.length : null,
        min_battery: batt.length ? Math.min(...batt) : null, flight_time_left_s: null,
        separation_violations: past.filter((e) => e.category === 'COLLISION' && e.kind !== 'collision').length,
        collisions: past.filter((e) => e.kind === 'collision').length,
        failed: drones.filter((d) => d.health === 'FAILED').length, min_separation_breaches: 0,
      },
      swarm_control: null, missions: { geofence: m.geofence, groups: {} }, alerts: null, failures: null,
      drones, events: fresh.map((e, k) => ({ ...e, seq: e.seq ?? (this._eventCursor - fresh.length + k + 1) })),
      last_event_seq: fresh.length ? fresh[fresh.length - 1].seq : 0,
    };
  }

  // ------------------------------------------------------------------ UI
  _drawMarkers() {
    const track = $('rp-markers');
    const dur = this.meta.duration_s || 1;
    track.innerHTML = this.meta.events.map((e) => `<i style="left:${(100 * e.time / dur).toFixed(3)}%;background:${markerColor(e)}"`
      + ` data-t="${e.time}" title="${clock(e.time)} · ${esc(e.category)}${e.drone_id != null ? ` · D${String(e.drone_id).padStart(2, '0')}` : ''} · ${esc(e.message)}"></i>`).join('');
    $('rp-count').textContent = `${this.meta.events.length} markers`;
  }

  _syncButtons() {
    const b = $('rp-play');
    b.querySelector('use').setAttribute('href', this.playing ? '#i-pause' : '#i-play');
    b.setAttribute('aria-pressed', String(this.playing));
    b.dataset.tip = this.playing ? 'Pause replay — Space' : 'Play replay — Space';
  }

  report(format) {
    const id = this.meta?.run_id ?? $('rp-run').value;
    if (!id) { this.ui.toast('No recorded run selected', 'warn'); return; }
    openReport(id, format);
  }

  _wire() {
    $('rp-speed').innerHTML = REPLAY_SPEEDS.map((s) => `<option value="${s}"${s === 1 ? ' selected' : ''}>${s}×</option>`).join('');
    $('rp-speed').onchange = (e) => this.setSpeed(Number(e.target.value));
    $('rp-run').onchange = (e) => this.load(e.target.value);
    $('rp-refresh').onclick = () => this.refreshRuns();
    $('rp-play').onclick = () => this.toggle();
    $('rp-back').onclick = () => this.step(-10);
    $('rp-fwd').onclick = () => this.step(10);
    $('rp-scrub').oninput = (e) => this.seek(Number(e.target.value));
    $('rp-markers').addEventListener('click', (e) => {
      const m = e.target.closest('[data-t]');
      if (m) this.seek(Math.max(0, Number(m.dataset.t) - 1));
    });
    $('rp-report-html').onclick = () => this.report('html');
    $('rp-report-pdf').onclick = () => this.report('pdf');
  }
}

/** Open the HTML report in a new tab, or download the PDF (token in the URL: plain links cannot send headers). */
export function openReport(runId, format = 'html') {
  const url = auth.withToken(`/api/replay/runs/${encodeURIComponent(runId)}/report?format=${format}`);
  if (format === 'html') { window.open(url, '_blank', 'noopener'); return; }
  const a = document.createElement('a');
  a.href = url;
  a.download = `gandiv-report-${runId}.pdf`;
  document.body.appendChild(a);
  a.click();
  a.remove();
}
