// Minimap in the corner of the 3D view: drones, home, geofence and the camera's position / view
// direction, auto-framed around the swarm and the camera target. Click to move the camera there.
import { VISUAL_STATES, classify } from './state.js';

const SIZE = 184;
const MIN_EXTENT = 240;      // m, smallest area shown

export class Minimap {
  constructor(canvas, scene, onPick) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.scene = scene;
    this.view = { x: 0, y: 0, extent: 600 };
    this.canvas.addEventListener('click', (e) => {
      const r = canvas.getBoundingClientRect();
      const [x, y] = this._toWorld(e.clientX - r.left, e.clientY - r.top);
      onPick(x, y);
    });
  }

  _toScreen(x, y) {
    const s = SIZE / this.view.extent;
    return [SIZE / 2 + (x - this.view.x) * s, SIZE / 2 - (y - this.view.y) * s];
  }

  _toWorld(sx, sy) {
    const s = SIZE / this.view.extent;
    return [this.view.x + (sx - SIZE / 2) / s, this.view.y - (sy - SIZE / 2) / s];
  }

  draw(snap, selection, dt) {
    if (!snap || this.canvas.hidden) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (this.canvas.width !== SIZE * dpr) { this.canvas.width = SIZE * dpr; this.canvas.height = SIZE * dpr; }
    const cam = this.scene.cameraState();              // ENU camera + target
    // frame: the swarm plus the camera target, with a margin; eased so the map does not jump
    const xs = snap.drones.map((d) => d.position.x).concat([cam.target.x, snap.world.home.x]);
    const ys = snap.drones.map((d) => d.position.y).concat([cam.target.y, snap.world.home.y]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
    const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const ext = Math.max(MIN_EXTENT, 1.3 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)));
    const k = 1 - Math.exp(-dt * 3);
    this.view.x += (cx - this.view.x) * k;
    this.view.y += (cy - this.view.y) * k;
    this.view.extent += (ext - this.view.extent) * k;

    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = 'rgba(38,38,38,0.9)';
    g.fillRect(0, 0, SIZE, SIZE);
    // grid every 100 m
    const [wx0, wy1] = this._toWorld(0, 0);
    const [wx1, wy0] = this._toWorld(SIZE, SIZE);
    const step = this.view.extent > 1500 ? 500 : 100;
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    g.beginPath();
    for (let x = Math.ceil(wx0 / step) * step; x < wx1; x += step) { const [sx] = this._toScreen(x, 0); g.moveTo(sx, 0); g.lineTo(sx, SIZE); }
    for (let y = Math.ceil(wy0 / step) * step; y < wy1; y += step) { const [, sy] = this._toScreen(0, y); g.moveTo(0, sy); g.lineTo(SIZE, sy); }
    g.stroke();
    // world bounds
    const b = snap.world.bounds;
    const [bx0, by0] = this._toScreen(b.min.x, b.max.y);
    const [bx1, by1] = this._toScreen(b.max.x, b.min.y);
    g.strokeStyle = 'rgba(91,141,239,0.6)';
    g.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);
    // geofence
    const gf = snap.missions?.geofence;
    const poly = (pts, stroke, fill) => {
      if (!pts?.length) return;
      g.beginPath();
      pts.forEach(([x, y], i) => { const [sx, sy] = this._toScreen(x, y); if (i) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
      g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      g.strokeStyle = stroke; g.stroke();
    };
    if (gf) {
      poly(gf.inclusion, '#F5C542', null);
      for (const z of gf.exclusions ?? []) poly(z.polygon, '#FF4D4D', 'rgba(255,77,77,0.25)');
    }
    // home
    const [hx, hy] = this._toScreen(snap.world.home.x, snap.world.home.y);
    g.strokeStyle = '#5B8DEF';
    g.beginPath(); g.arc(hx, hy, 4, 0, Math.PI * 2); g.stroke();
    // camera view wedge
    const [tx, ty] = this._toScreen(cam.target.x, cam.target.y);
    const [px, py] = this._toScreen(cam.position.x, cam.position.y);
    const ang = Math.atan2(ty - py, tx - px);
    const r = Math.min(40, Math.max(14, Math.hypot(tx - px, ty - py)));
    g.fillStyle = 'rgba(217,188,132,0.22)';
    g.beginPath(); g.moveTo(px, py); g.arc(px, py, r, ang - 0.5, ang + 0.5); g.closePath(); g.fill();
    g.fillStyle = '#D9BC84';
    g.beginPath(); g.arc(px, py, 3, 0, Math.PI * 2); g.fill();
    g.strokeStyle = '#D9BC84';
    g.beginPath(); g.moveTo(tx - 4, ty); g.lineTo(tx + 4, ty); g.moveTo(tx, ty - 4); g.lineTo(tx, ty + 4); g.stroke();
    // drones
    for (const d of snap.drones) {
      const [sx, sy] = this._toScreen(d.position.x, d.position.y);
      g.fillStyle = VISUAL_STATES[classify(d)].color;
      g.beginPath(); g.arc(sx, sy, selection.has(d.drone_id) ? 3.5 : 2.5, 0, Math.PI * 2); g.fill();
      if (selection.has(d.drone_id)) { g.strokeStyle = '#fff'; g.stroke(); }
    }
    // scale + north
    g.fillStyle = 'rgba(242,237,237,0.85)';
    g.font = '600 9px ui-monospace, Consolas, monospace';
    g.fillText(`${Math.round(this.view.extent)} m`, 4, SIZE - 4);
    g.fillText('N', SIZE - 10, 11);
  }
}
