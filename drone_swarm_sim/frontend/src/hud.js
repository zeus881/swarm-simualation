// Head-up display for the selected drone (Mission Planner style), drawn on a 2D canvas.
// Values are exponentially smoothed so the HUD animates at display rate between 20 Hz telemetry frames.
import { batteryColor } from './state.js';

const W = 420;
const H = 260;
const TOP = 28;          // heading tape height
const BOTTOM = 40;       // status strip height
const SIDE = 58;         // speed / altitude tape width
const PPD = 3.2;         // pixels per degree of pitch
const MONO = 'ui-monospace, "Cascadia Mono", Consolas, monospace';

function wrap360(a) { return ((a % 360) + 360) % 360; }
function angleLerp(a, b, k) { let d = wrap360(b - a); if (d > 180) d -= 360; return wrap360(a + d * k); }

export class Hud {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.state = null;        // smoothed values
    this.target = null;       // latest telemetry
    this.thresholds = null;
  }

  /** Latest telemetry of the drone to display (or null to hide). */
  setDrone(d, thresholds) {
    this.target = d;
    this.thresholds = thresholds;
    if (d && (!this.state || this.state.id !== d.drone_id)) {
      this.state = { id: d.drone_id, roll: d.roll, pitch: d.pitch, heading: d.heading, speed: this._gs(d),
        alt: d.altitude_agl, vz: d.velocity.z };
    }
  }

  _gs(d) { return Math.hypot(d.velocity.x, d.velocity.y); }

  draw(dt) {
    const d = this.target;
    if (!d || !this.state) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.canvas.width !== W * dpr) { this.canvas.width = W * dpr; this.canvas.height = H * dpr; }
    const s = this.state;
    const k = 1 - Math.exp(-dt * 10);
    s.roll += (d.roll - s.roll) * k;
    s.pitch += (d.pitch - s.pitch) * k;
    s.heading = angleLerp(s.heading, d.heading, k);
    s.speed += (this._gs(d) - s.speed) * k;
    s.alt += (d.altitude_agl - s.alt) * k;
    s.vz += (d.velocity.z - s.vz) * k;

    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    this._horizon(g, s);
    this._rollArc(g, s);
    this._aircraft(g);
    this._headingTape(g, s);
    this._tape(g, s.speed, 'left', 'GS m/s', 2, 1);
    this._tape(g, s.alt, 'right', 'ALT m', 5, 0);
    this._vario(g, s.vz);
    this._status(g, d);
  }

  _horizon(g, s) {
    const cx = W / 2;
    const cy = TOP + (H - TOP - BOTTOM) / 2;
    g.save();
    g.beginPath();
    g.rect(0, TOP, W, H - TOP - BOTTOM);
    g.clip();
    g.translate(cx, cy);
    g.rotate(-s.roll * Math.PI / 180);
    g.translate(0, s.pitch * PPD);
    const big = 800;
    const sky = g.createLinearGradient(0, -big / 2, 0, 0);
    sky.addColorStop(0, '#1d4f82'); sky.addColorStop(1, '#4a90c8');
    g.fillStyle = sky;
    g.fillRect(-big, -big, big * 2, big);
    const ground = g.createLinearGradient(0, 0, 0, big / 2);
    ground.addColorStop(0, '#7a5a2e'); ground.addColorStop(1, '#4a3518');
    g.fillStyle = ground;
    g.fillRect(-big, 0, big * 2, big);
    g.strokeStyle = '#fff';
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(-big, 0); g.lineTo(big, 0); g.stroke();
    // pitch ladder
    g.lineWidth = 1.2;
    g.font = `600 10px ${MONO}`;
    g.fillStyle = '#fff';
    g.textAlign = 'right';
    for (let p = -30; p <= 30; p += 5) {
      if (p === 0) continue;
      const y = -p * PPD;
      const half = p % 10 === 0 ? 34 : 18;
      g.beginPath(); g.moveTo(-half, y); g.lineTo(half, y); g.stroke();
      if (p % 10 === 0) { g.fillText(String(p), -half - 4, y + 3); }
    }
    g.restore();
  }

  _rollArc(g, s) {
    const cx = W / 2;
    const cy = TOP + (H - TOP - BOTTOM) / 2;
    const r = 88;
    g.save();
    g.translate(cx, cy);
    g.strokeStyle = 'rgba(255,255,255,0.9)';
    g.lineWidth = 1.5;
    g.beginPath(); g.arc(0, 0, r, (-90 - 60) * Math.PI / 180, (-90 + 60) * Math.PI / 180); g.stroke();
    for (const a of [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]) {
      const len = a % 30 === 0 ? 10 : 6;
      const t = (a - 90) * Math.PI / 180;
      g.beginPath(); g.moveTo(Math.cos(t) * r, Math.sin(t) * r); g.lineTo(Math.cos(t) * (r - len), Math.sin(t) * (r - len)); g.stroke();
    }
    g.rotate(-s.roll * Math.PI / 180);
    g.fillStyle = '#F5C542';
    g.beginPath(); g.moveTo(0, -r + 2); g.lineTo(-6, -r + 12); g.lineTo(6, -r + 12); g.closePath(); g.fill();
    g.restore();
  }

  _aircraft(g) {
    const cx = W / 2;
    const cy = TOP + (H - TOP - BOTTOM) / 2;
    g.save();
    g.strokeStyle = '#F5C542';
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(cx - 44, cy); g.lineTo(cx - 14, cy); g.lineTo(cx - 7, cy + 7); g.lineTo(cx, cy); g.lineTo(cx + 7, cy + 7);
    g.lineTo(cx + 14, cy); g.lineTo(cx + 44, cy);
    g.stroke();
    g.fillStyle = '#F5C542';
    g.beginPath(); g.arc(cx, cy, 2.5, 0, Math.PI * 2); g.fill();
    g.restore();
  }

  _headingTape(g, s) {
    g.save();
    g.fillStyle = 'rgba(20,20,20,0.82)';
    g.fillRect(0, 0, W, TOP);
    g.beginPath(); g.rect(0, 0, W, TOP); g.clip();
    const ppd = 3;                      // px per degree
    const cx = W / 2;
    g.strokeStyle = '#ddd';
    g.fillStyle = '#eee';
    g.font = `700 11px ${MONO}`;
    g.textAlign = 'center';
    const start = Math.floor((s.heading - 80) / 5) * 5;
    for (let h = start; h <= s.heading + 80; h += 5) {
      const x = cx + (h - s.heading) * ppd;
      const hh = wrap360(h);
      const major = hh % 30 === 0;
      g.beginPath(); g.moveTo(x, TOP); g.lineTo(x, TOP - (major ? 9 : 5)); g.stroke();
      if (major) {
        const label = { 0: 'N', 90: 'E', 180: 'S', 270: 'W' }[hh] ?? String(hh / 10);
        g.fillStyle = label.length === 1 && isNaN(label) ? '#F5C542' : '#eee';
        g.fillText(label, x, 12);
      }
    }
    g.restore();
    // current heading box
    g.fillStyle = '#111';
    g.strokeStyle = '#F5C542';
    g.lineWidth = 1.5;
    g.fillRect(W / 2 - 24, 2, 48, 18);
    g.strokeRect(W / 2 - 24, 2, 48, 18);
    g.fillStyle = '#fff';
    g.font = `800 12px ${MONO}`;
    g.textAlign = 'center';
    g.fillText(String(Math.round(s.heading) % 360).padStart(3, '0') + '°', W / 2, 15);
  }

  _tape(g, value, side, label, step, decimals) {
    const top = TOP + 6;
    const bottom = H - BOTTOM - 6;
    const x0 = side === 'left' ? 4 : W - SIDE - 4;
    const mid = (top + bottom) / 2;
    const ppu = side === 'left' ? 9 : 3.5;            // px per unit
    g.save();
    g.fillStyle = 'rgba(20,20,20,0.6)';
    g.fillRect(x0, top, SIDE, bottom - top);
    g.beginPath(); g.rect(x0, top, SIDE, bottom - top); g.clip();
    g.strokeStyle = '#ddd';
    g.fillStyle = '#eee';
    g.font = `600 10px ${MONO}`;
    g.textAlign = side === 'left' ? 'right' : 'left';
    const range = (bottom - top) / 2 / ppu;
    const first = Math.floor((value - range) / step) * step;
    for (let v = first; v <= value + range; v += step) {
      if (side === 'left' && v < 0) continue;
      const y = mid - (v - value) * ppu;
      const tick = side === 'left' ? [x0 + SIDE - 8, x0 + SIDE] : [x0, x0 + 8];
      g.beginPath(); g.moveTo(tick[0], y); g.lineTo(tick[1], y); g.stroke();
      g.fillText(String(Math.round(v)), side === 'left' ? x0 + SIDE - 11 : x0 + 11, y + 3);
    }
    g.restore();
    // value box
    g.fillStyle = '#111';
    g.strokeStyle = '#F5C542';
    g.lineWidth = 1.5;
    g.fillRect(x0 - 1, mid - 11, SIDE + 2, 22);
    g.strokeRect(x0 - 1, mid - 11, SIDE + 2, 22);
    g.fillStyle = '#fff';
    g.font = `800 13px ${MONO}`;
    g.textAlign = 'center';
    g.fillText(value.toFixed(decimals), x0 + SIDE / 2, mid + 5);
    // caption in its own box so tick labels never run through it
    g.fillStyle = 'rgba(10,10,10,0.9)';
    g.fillRect(x0, top, SIDE, 13);
    g.fillStyle = '#F5C542';
    g.font = `700 9px ${MONO}`;
    g.fillText(label, x0 + SIDE / 2, top + 10);
  }

  _vario(g, vz) {
    // vertical-speed arrow beside the altitude tape
    const x = W - SIDE - 12;
    const mid = TOP + (H - TOP - BOTTOM) / 2;
    const len = Math.max(-50, Math.min(50, vz * 10));
    g.strokeStyle = vz >= 0 ? '#3DDC97' : '#FF8C42';
    g.fillStyle = g.strokeStyle;
    g.lineWidth = 3;
    g.beginPath(); g.moveTo(x, mid); g.lineTo(x, mid - len); g.stroke();
    if (Math.abs(len) > 4) {
      const dir = len > 0 ? -1 : 1;
      g.beginPath(); g.moveTo(x - 5, mid - len - dir * 6); g.lineTo(x + 5, mid - len - dir * 6); g.lineTo(x, mid - len); g.fill();
    }
    g.font = `700 9px ${MONO}`;
    g.textAlign = 'right';
    g.fillText(`${vz >= 0 ? '+' : ''}${vz.toFixed(1)}`, x - 6, mid - len + (len > 0 ? -2 : 10));
  }

  _status(g, d) {
    const y = H - BOTTOM;
    g.fillStyle = 'rgba(20,20,20,0.88)';
    g.fillRect(0, y, W, BOTTOM);
    g.font = `800 12px ${MONO}`;
    g.textAlign = 'left';
    // name + mode
    g.fillStyle = '#fff';
    g.fillText(d.name, 8, y + 16);
    g.fillStyle = d.mode === 'EMERGENCY' ? '#FF4D4D' : '#F5C542';
    g.fillText(d.mode, 8, y + 32);
    g.fillStyle = d.armed ? '#3DDC97' : '#9aa';
    g.font = `700 9px ${MONO}`;
    g.fillText(d.armed ? 'ARMED' : 'DISARMED', 44, y + 16);
    // battery
    const bx = 118;
    const pct = Math.max(0, Math.min(100, d.battery));
    g.strokeStyle = '#ccc';
    g.lineWidth = 1;
    g.strokeRect(bx, y + 8, 60, 13);
    g.fillRect(bx + 60, y + 12, 3, 5);
    g.fillStyle = batteryColor(d.battery, this.thresholds);
    g.fillRect(bx + 1.5, y + 9.5, 57 * pct / 100, 10);
    g.fillStyle = '#fff';
    g.font = `800 11px ${MONO}`;
    g.fillText(`${Math.round(d.battery)}%`, bx + 66, y + 19);
    g.font = `600 10px ${MONO}`;
    g.fillStyle = '#ccc';
    const left = d.time_left_s == null ? '—' : `${Math.floor(d.time_left_s / 60)}:${String(Math.floor(d.time_left_s % 60)).padStart(2, '0')}`;
    g.fillText(`${d.battery_voltage.toFixed(1)} V · ${left} left`, bx, y + 34);
    // GPS
    const gx = 262;
    const fixColor = d.gps_fix === '3D' ? '#3DDC97' : d.gps_fix === '2D' ? '#F5C542' : '#FF4D4D';
    g.fillStyle = fixColor;
    g.font = `800 11px ${MONO}`;
    g.fillText(`GPS ${d.gps_fix}`, gx, y + 16);
    g.fillStyle = '#ccc';
    g.font = `600 10px ${MONO}`;
    g.fillText(`${d.satellites} sat · HDOP ${Number(d.hdop).toFixed(1)}`, gx, y + 32);
    // link quality bars
    const lx = 372;
    const q = d.communication === 'LOST' ? 0 : d.link_quality ?? 0;
    for (let i = 0; i < 4; i += 1) {
      const on = q >= (i + 1) * 25 - 12;
      g.fillStyle = on ? (q > 50 ? '#3DDC97' : q > 25 ? '#F5C542' : '#FF8C42') : '#555';
      g.fillRect(lx + i * 9, y + 20 - i * 4, 6, 6 + i * 4);
    }
    g.fillStyle = d.communication === 'LOST' ? '#FF4D4D' : '#ccc';
    g.font = `700 9px ${MONO}`;
    g.fillText(d.communication === 'LOST' ? 'NO LINK' : `LINK ${Math.round(q)}%`, lx - 4, y + 36);
  }
}
