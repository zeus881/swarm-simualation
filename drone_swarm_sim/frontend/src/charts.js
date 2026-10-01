// Lightweight time-series charts for the dock (no library: plain canvas 2D, works offline).
// TelemetryHistory keeps the last WINDOW_S seconds per drone in fixed-size ring buffers, sampled on
// simulation time so a paused simulation freezes the charts and memory never grows.

export const WINDOW_S = 120;
const RATE_HZ = 5;
const N = WINDOW_S * RATE_HZ;
export const SERIES_COLORS = ['#3F4A2C', '#C62828', '#1F6FB5', '#A8874A', '#8E44AD', '#138D75', '#D35400', '#5D6D7E'];

class Ring {
  constructor() {
    this.t = new Float64Array(N);
    this.v = new Float32Array(N);
    this.head = 0;           // next write index
    this.count = 0;
  }

  push(t, v) {
    this.t[this.head] = t;
    this.v[this.head] = v;
    this.head = (this.head + 1) % N;
    this.count = Math.min(this.count + 1, N);
  }

  /** Iterate samples oldest -> newest. */
  *[Symbol.iterator]() {
    const start = (this.head - this.count + N) % N;
    for (let i = 0; i < this.count; i += 1) {
      const k = (start + i) % N;
      yield [this.t[k], this.v[k]];
    }
  }

  last() { return this.count ? this.v[(this.head - 1 + N) % N] : null; }
}

export class TelemetryHistory {
  constructor() { this.reset(); }

  reset() {
    this.drones = new Map();     // id -> {alt, speed, battery, nearest}
    this.swarm = { alt: new Ring(), speed: new Ring(), battery: new Ring(), sep: new Ring() };
    this.lastT = -Infinity;
    this.now = 0;
  }

  push(snap) {
    const t = snap.sim_time;
    if (t < this.lastT) this.reset();                  // simulation reset
    this.now = t;
    if (t - this.lastT < 1 / RATE_HZ - 1e-6) return;
    this.lastT = t;
    let sa = 0; let ss = 0; let sb = 0; let n = 0;
    const seen = new Set();
    for (const d of snap.drones) {
      seen.add(d.drone_id);
      let h = this.drones.get(d.drone_id);
      if (!h) { h = { alt: new Ring(), speed: new Ring(), battery: new Ring(), nearest: new Ring() }; this.drones.set(d.drone_id, h); }
      h.alt.push(t, d.altitude_agl);
      h.speed.push(t, d.speed);
      h.battery.push(t, d.battery);
      h.nearest.push(t, d.nearest_distance ?? NaN);
      if (d.airborne) { sa += d.altitude_agl; ss += d.speed; n += 1; }
      sb += d.battery;
    }
    for (const id of [...this.drones.keys()]) if (!seen.has(id)) this.drones.delete(id);
    this.swarm.alt.push(t, n ? sa / n : 0);
    this.swarm.speed.push(t, n ? ss / n : 0);
    this.swarm.battery.push(t, snap.drones.length ? sb / snap.drones.length : NaN);
    this.swarm.sep.push(t, snap.summary.min_separation ?? NaN);
  }
}

export class LineChart {
  constructor(canvas, { title, unit, min = null, max = null, decimals = 1 }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    Object.assign(this, { title, unit, min, max, decimals });
  }

  /**
   * @param series [{label, color, ring, dashed?}]
   * @param now    current simulation time [s]
   * @param lines  horizontal reference lines [{value, color, label}]
   */
  draw(series, now, lines = []) {
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pad = { l: 34, r: 8, t: 18, b: 16 };
    const t0 = now - WINDOW_S;
    // value range
    let lo = Infinity; let hi = -Infinity;
    for (const s of series) for (const [t, v] of s.ring) if (t >= t0 && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const l of lines) { lo = Math.min(lo, l.value); hi = Math.max(hi, l.value); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (this.min != null) lo = Math.min(lo, this.min);
    if (this.max != null) hi = Math.max(hi, this.max);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    const span = hi - lo;
    lo -= span * 0.05; hi += span * 0.08;
    const X = (t) => pad.l + ((t - t0) / WINDOW_S) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    // grid + axes
    g.font = '10px ui-monospace, Consolas, monospace';
    g.fillStyle = '#5E5054';
    g.strokeStyle = '#F0E0E2';
    g.lineWidth = 1;
    const ticks = 4;
    g.textAlign = 'right';
    for (let i = 0; i <= ticks; i += 1) {
      const v = lo + (hi - lo) * i / ticks;
      const y = Y(v);
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(w - pad.r, y); g.stroke();
      g.fillText(v.toFixed(Math.abs(hi - lo) < 10 ? 1 : 0), pad.l - 4, y + 3);
    }
    g.textAlign = 'center';
    for (let s = 0; s <= WINDOW_S; s += 30) {
      const x = X(t0 + s);
      g.fillText(s === WINDOW_S ? 'now' : `-${WINDOW_S - s}s`, x, h - 3);
    }
    // reference lines
    lines.forEach((l, k) => {
      g.strokeStyle = l.color;
      g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(pad.l, Y(l.value)); g.lineTo(w - pad.r, Y(l.value)); g.stroke();
      g.setLineDash([]);
      g.fillStyle = l.color;
      // alternate left / right so labels of nearby lines do not overlap
      g.textAlign = k % 2 ? 'right' : 'left';
      g.fillText(l.label, k % 2 ? w - pad.r - 3 : pad.l + 3, Y(l.value) + (k % 2 ? 11 : -3));
    });
    // series
    g.save();
    g.beginPath(); g.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b); g.clip();
    for (const s of series) {
      g.strokeStyle = s.color;
      g.lineWidth = 1.6;
      g.setLineDash(s.dashed ? [5, 3] : []);
      g.beginPath();
      let pen = false;
      for (const [t, v] of s.ring) {
        if (t < t0 || !Number.isFinite(v)) { pen = false; continue; }
        const x = X(t); const y = Y(v);
        if (pen) g.lineTo(x, y); else g.moveTo(x, y);
        pen = true;
      }
      g.stroke();
    }
    g.restore();
    g.setLineDash([]);
    // title + latest value of the first series
    g.textAlign = 'left';
    g.fillStyle = '#3A3A3A';
    g.font = '700 11px Inter, system-ui, sans-serif';
    const last = series[0]?.ring.last();
    g.fillText(`${this.title}${last != null && Number.isFinite(last) ? `  ${last.toFixed(this.decimals)} ${this.unit}` : ''}`, pad.l, 12);
  }
}
