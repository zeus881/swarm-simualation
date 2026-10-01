// PLAN tab: top-down mission planner (waypoints, survey polygons, geofence, files, upload).
// World coordinates are local ENU metres (x = East, y = North), exactly as the backend uses them.
import { VISUAL_STATES, classify } from './state.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d));

export const ACTIONS = ['WAYPOINT', 'LOITER', 'TAKEOFF', 'LAND', 'RTL', 'CHANGE_FORMATION'];
const ACTION_SHORT = { WAYPOINT: 'WP', LOITER: 'LOIT', TAKEOFF: 'T/O', LAND: 'LAND', RTL: 'RTL', CHANGE_FORMATION: 'FORM' };
const ACTION_COLOR = {
  WAYPOINT: '#D9BC84', LOITER: '#7FD1B9', TAKEOFF: '#3DDC97', LAND: '#5B8DEF', RTL: '#FF8C42', CHANGE_FORMATION: '#C9A0FF',
};
const POSITIONAL = new Set(['WAYPOINT', 'LOITER', 'LAND']);
const SHAPES = ['v', 'line', 'column', 'wedge', 'diamond', 'grid', 'circle', 'custom'];
const TRACK_COLORS = ['#D9BC84', '#7FD1B9', '#FF8C42', '#C9A0FF', '#5B8DEF', '#F5C542', '#FF6FAE', '#9BE15D'];
const HIT_PX = 10;
const DRAFT_LABEL = { survey: 'survey area', fence: 'inclusion fence', nfz: 'no-fly zone' };

function dist2(ax, ay, bx, by) { return (ax - bx) ** 2 + (ay - by) ** 2; }

function segDist2(px, py, ax, ay, bx, by) {
  const dx = bx - ax;
  const dy = by - ay;
  const len = dx * dx + dy * dy;
  const t = len ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len)) : 0;
  return dist2(px, py, ax + t * dx, ay + t * dy);
}

/** Compass heading (deg) from a to b in ENU. */
export function bearing(ax, ay, bx, by) {
  return ((90 - Math.atan2(by - ay, bx - ax) * 180 / Math.PI) % 360 + 360) % 360;
}

export class MissionPlanner {
  constructor(dashboard) {
    this.ui = dashboard;
    this.link = dashboard.link;
    this.canvas = $('plan-canvas');
    this.ctx = this.canvas.getContext('2d');
    this.visible = false;
    this.world = null;
    this.defaults = { altitude: 30, speed: 8, loiter_radius: 20 };
    // mission model
    this.mission = { name: 'New mission', tracks: [[]] };
    this.track = 0;
    this.selected = -1;
    this.surveyPolygon = null;
    this.surveyStats = null;
    // fence model (mirrors the server until edited)
    this.fence = { enabled: false, action: 'RTL', inclusion: null, exclusions: [], max_altitude: null };
    this.fenceVersion = -1;
    this.fenceDirty = false;
    // interaction
    this.tool = 'select';
    this.draft = null;            // polygon being drawn: {kind, points}
    this.drag = null;             // {type: 'wp'|'pan'|'vertex', ...}
    this.measure = null;          // {a: [x,y], b: [x,y] | null}
    this.hover = null;
    this.view = { x: 0, y: 0, scale: 0.35 };   // centre (ENU m) and pixels per metre
    this._fitted = false;
    this.activePaths = [];
    this.missionVersion = -1;
    this._dirty = true;
    this._wireCanvas();
    this._wirePanel();
    new ResizeObserver(() => { this._dirty = true; }).observe(this.canvas);
    this._renderTable();
    this._renderFence();
  }

  // ------------------------------------------------------------------ public
  setVisible(on) {
    this.visible = on;
    $('plan-view').hidden = !on;
    $('plan-panel').hidden = !on;
    if (on) {
      this._dirty = true;
      if (!this._fitted && this.world) this.fit();
      this.refreshLibrary();
      this.canvas.focus({ preventScroll: true });
    } else {
      this._closeMenu();
    }
  }

  onTelemetry(snap) {
    this.world = snap.world;
    if (snap.world.mission_defaults) this.defaults = snap.world.mission_defaults;
    const gf = snap.missions?.geofence;
    if (gf && gf.version !== this.fenceVersion) {
      this.fenceVersion = gf.version;
      if (!this.fenceDirty) {
        this.fence = { enabled: gf.enabled, action: gf.action, inclusion: gf.inclusion,
          exclusions: gf.exclusions.map((z) => ({ ...z })), max_altitude: gf.max_altitude };
        this._renderFence();
      }
    }
    const mv = snap.missions?.version ?? 0;
    if (mv !== this.missionVersion) {
      this.missionVersion = mv;
      this._fetchActive();
    }
    this._renderAssign(snap);
    this._renderRunStatus(snap);
    if (this.visible) this._dirty = true;
  }

  animate() {
    if (!this.visible || !this._dirty) return;
    this._dirty = false;
    this._draw();
  }

  /** Tracks of the mission as sent to the backend. */
  missionPayload() {
    const clean = (w) => {
      const out = { x: +w.x.toFixed(2), y: +w.y.toFixed(2), alt: +w.alt, hold: +(w.hold || 0), action: w.action };
      if (w.speed != null && w.speed !== '') out.speed = +w.speed;
      if (w.params && Object.keys(w.params).length) out.params = { ...w.params };
      return out;
    };
    const tracks = this.mission.tracks.filter((t) => t.length);
    const base = { schema: 'gandiv.mission/1', name: this.mission.name || 'mission', altitude_mode: $('pm-altmode').value };
    const payload = tracks.length > 1 ? { ...base, tracks: tracks.map((t) => t.map(clean)) }
      : { ...base, waypoints: (tracks[0] ?? []).map(clean) };
    if (this.surveyStats) payload.survey = this.surveyStats;
    if (this._fenceDefined()) payload.geofence = this._fencePayload();
    return payload;
  }

  loadMission(m) {
    const tracks = m.tracks ?? [m.waypoints ?? []];
    this.mission = { name: m.name ?? 'mission', tracks: tracks.map((t) => t.map((w) => ({ ...w, params: { ...(w.params ?? {}) } }))) };
    this.track = 0;
    this.selected = -1;
    this.surveyStats = m.survey ?? null;
    this.surveyPolygon = m.survey?.polygon ?? null;
    if (m.geofence) {
      this.fence = { enabled: !!m.geofence.enabled, action: m.geofence.action ?? 'RTL', inclusion: m.geofence.inclusion ?? null,
        exclusions: (m.geofence.exclusions ?? []).map((z, k) => (Array.isArray(z) ? { name: `NFZ ${k + 1}`, polygon: z } : { ...z })),
        max_altitude: m.geofence.max_altitude ?? null };
      this._setFenceDirty(true);
    }
    $('pm-name').value = this.mission.name;
    $('pm-altmode').value = m.altitude_mode === 'agl' ? 'agl' : 'relative';
    this._renderTable();
    this._renderFence();
    this.fit();
  }

  fit() {
    const pts = [];
    for (const t of this.mission.tracks) for (const w of t) if (POSITIONAL.has(w.action) || w.action === 'TAKEOFF') pts.push([w.x, w.y]);
    for (const p of this.surveyPolygon ?? []) pts.push(p);
    for (const p of this.fence.inclusion ?? []) pts.push(p);
    if (pts.length < 2 && this.world) {
      const { min, max } = this.world.bounds;
      pts.push([min.x, min.y], [max.x, max.y]);
    }
    if (!pts.length) return;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const w = this.canvas.clientWidth || 800;
    const h = this.canvas.clientHeight || 600;
    this.view.x = (x0 + x1) / 2;
    this.view.y = (y0 + y1) / 2;
    this.view.scale = Math.max(0.02, Math.min(20, 0.85 * Math.min(w / Math.max(x1 - x0, 40), h / Math.max(y1 - y0, 40))));
    this._fitted = true;
    this._dirty = true;
  }

  // ------------------------------------------------------------------ coordinates
  toScreen(x, y) {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    return [w / 2 + (x - this.view.x) * this.view.scale, h / 2 - (y - this.view.y) * this.view.scale];
  }

  toWorld(sx, sy) {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    return [this.view.x + (sx - w / 2) / this.view.scale, this.view.y - (sy - h / 2) / this.view.scale];
  }

  _eventPos(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  get waypoints() { return this.mission.tracks[this.track] ?? (this.mission.tracks[this.track] = []); }

  _newWaypoint(x, y) {
    const prev = this.waypoints[this.waypoints.length - 1];
    return { x, y, alt: prev?.alt ?? this.defaults.altitude, speed: prev?.speed ?? this.defaults.speed, hold: 0,
      action: 'WAYPOINT', params: {} };
  }

  // ------------------------------------------------------------------ hit testing
  _hitWaypoint(sx, sy) {
    const wps = this.waypoints;
    for (let k = wps.length - 1; k >= 0; k -= 1) {
      const [px, py] = this.toScreen(wps[k].x, wps[k].y);
      if (dist2(sx, sy, px, py) <= HIT_PX * HIT_PX && (POSITIONAL.has(wps[k].action) || wps[k].action === 'TAKEOFF')) return k;
    }
    return -1;
  }

  _hitSegment(sx, sy) {
    const pts = this._pathPoints(this.waypoints);
    for (let k = 1; k < pts.length; k += 1) {
      const [ax, ay] = this.toScreen(pts[k - 1].x, pts[k - 1].y);
      const [bx, by] = this.toScreen(pts[k].x, pts[k].y);
      if (segDist2(sx, sy, ax, ay, bx, by) <= 36) return pts[k].index;   // insert before this waypoint
    }
    return -1;
  }

  _hitVertex(sx, sy) {
    const polys = [];
    if (this.surveyPolygon) polys.push({ kind: 'survey', poly: this.surveyPolygon });
    if (this.fence.inclusion) polys.push({ kind: 'fence', poly: this.fence.inclusion });
    this.fence.exclusions.forEach((z, zi) => polys.push({ kind: 'nfz', zone: zi, poly: z.polygon }));
    for (const p of polys) {
      for (let k = 0; k < p.poly.length; k += 1) {
        const [px, py] = this.toScreen(p.poly[k][0], p.poly[k][1]);
        if (dist2(sx, sy, px, py) <= HIT_PX * HIT_PX) return { ...p, index: k };
      }
    }
    return null;
  }

  _pathPoints(wps) {
    // Only waypoints with a position form the drawn path (RTL / formation changes happen in place).
    return wps.map((w, index) => ({ ...w, index })).filter((w) => POSITIONAL.has(w.action) || w.action === 'TAKEOFF');
  }

  // ------------------------------------------------------------------ canvas interaction
  _wireCanvas() {
    const c = this.canvas;
    c.addEventListener('contextmenu', (e) => { e.preventDefault(); this._contextMenu(e); });
    c.addEventListener('pointerdown', (e) => this._pointerDown(e));
    c.addEventListener('pointermove', (e) => this._pointerMove(e));
    c.addEventListener('pointerup', (e) => this._pointerUp(e));
    c.addEventListener('pointercancel', () => { this.drag = null; c.classList.remove('panning'); });
    c.addEventListener('dblclick', (e) => { e.preventDefault(); this._finishDraft(); });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const [sx, sy] = this._eventPos(e);
      const [wx, wy] = this.toWorld(sx, sy);
      const k = Math.exp(-e.deltaY * 0.0015);
      this.view.scale = Math.max(0.02, Math.min(40, this.view.scale * k));
      const [nx, ny] = this.toWorld(sx, sy);
      this.view.x += wx - nx;
      this.view.y += wy - ny;
      this._dirty = true;
    }, { passive: false });
    c.addEventListener('keydown', (e) => this._keyDown(e));
    document.addEventListener('pointerdown', (e) => { if (!$('plan-menu').contains(e.target)) this._closeMenu(); });
    for (const btn of document.querySelectorAll('.plan-tools [data-tool]')) {
      btn.addEventListener('click', () => this.setTool(btn.dataset.tool));
    }
    $('plan-fit').onclick = () => this.fit();
  }

  setTool(tool) {
    if (this.draft && tool !== this.draft.kind) this.draft = null;
    this.tool = tool;
    if (tool !== 'measure') this.measure = null;
    for (const btn of document.querySelectorAll('.plan-tools [data-tool]')) {
      btn.setAttribute('aria-pressed', String(btn.dataset.tool === tool));
    }
    this.canvas.classList.toggle('tool-select', tool === 'select');
    const hints = {
      select: 'Drag waypoints or polygon corners to move them · drag the map to pan · right-click for options',
      waypoint: 'Click to add a waypoint · drag to move · right-click to delete or insert',
      survey: 'Click the corners of the survey area · click the first corner (or double-click) to close',
      fence: 'Click the corners of the inclusion fence · click the first corner (or double-click) to close',
      nfz: 'Click the corners of a no-fly zone · click the first corner (or double-click) to close',
      measure: 'Click two points to measure distance and bearing',
    };
    $('plan-hint').textContent = hints[tool] ?? '';
    this._dirty = true;
  }

  _pointerDown(e) {
    this._closeMenu();
    const [sx, sy] = this._eventPos(e);
    const [wx, wy] = this.toWorld(sx, sy);
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      this._startPan(e, sx, sy);
      return;
    }
    if (e.button !== 0) return;
    this.canvas.setPointerCapture(e.pointerId);
    if (this.tool === 'survey' || this.tool === 'fence' || this.tool === 'nfz') {
      this._addDraftPoint(sx, sy, wx, wy);
      return;
    }
    if (this.tool === 'measure') {
      if (!this.measure || this.measure.b) this.measure = { a: [wx, wy], b: null };
      else this.measure.b = [wx, wy];
      this._dirty = true;
      return;
    }
    const hit = this._hitWaypoint(sx, sy);
    if (hit >= 0) {
      this.select(hit);
      this.drag = { type: 'wp', index: hit, moved: false };
      return;
    }
    const vertex = this.tool === 'select' ? this._hitVertex(sx, sy) : null;
    if (vertex) {
      this.drag = { type: 'vertex', ...vertex };
      return;
    }
    if (this.tool === 'waypoint') {
      this.waypoints.push(this._newWaypoint(wx, wy));
      this.select(this.waypoints.length - 1);
      this.drag = { type: 'wp', index: this.selected, moved: false };
      this._changed();
      return;
    }
    this._startPan(e, sx, sy);
  }

  _startPan(e, sx, sy) {
    this.canvas.setPointerCapture(e.pointerId);
    this.drag = { type: 'pan', sx, sy, vx: this.view.x, vy: this.view.y };
    this.canvas.classList.add('panning');
  }

  _pointerMove(e) {
    const [sx, sy] = this._eventPos(e);
    const [wx, wy] = this.toWorld(sx, sy);
    this._readout(wx, wy);
    const d = this.drag;
    if (!d) {
      if (this.draft || (this.measure && !this.measure.b)) { this.hover = [wx, wy]; this._dirty = true; }
      return;
    }
    if (d.type === 'pan') {
      this.view.x = d.vx - (sx - d.sx) / this.view.scale;
      this.view.y = d.vy + (sy - d.sy) / this.view.scale;
    } else if (d.type === 'wp') {
      const w = this.waypoints[d.index];
      w.x = wx;
      w.y = wy;
      d.moved = true;
    } else if (d.type === 'vertex') {
      d.poly[d.index] = [wx, wy];
      if (d.kind !== 'survey') this._setFenceDirty(true);
    }
    this._dirty = true;
  }

  _pointerUp() {
    const d = this.drag;
    this.drag = null;
    this.canvas.classList.remove('panning');
    if (d?.type === 'wp' && d.moved) this._renderTable();
    if (d?.type === 'vertex') this._renderFence();
  }

  _keyDown(e) {
    const key = e.key;
    if ((key === 'Delete' || key === 'Backspace') && this.selected >= 0) {
      e.preventDefault();
      this.deleteWaypoint(this.selected);
    } else if (key === 'Escape') {
      this.draft = null;
      this.measure = null;
      this._closeMenu();
      this._dirty = true;
    } else if (key === 'Enter' && this.draft) {
      this._finishDraft();
    } else if (!e.ctrlKey && !e.metaKey) {
      const tool = { v: 'select', w: 'waypoint', s: 'survey' }[key.toLowerCase()];
      if (tool) { e.preventDefault(); e.stopPropagation(); this.setTool(tool); }
    }
  }

  _addDraftPoint(sx, sy, wx, wy) {
    if (!this.draft || this.draft.kind !== this.tool) this.draft = { kind: this.tool, points: [] };
    const pts = this.draft.points;
    if (pts.length >= 3) {
      const [fx, fy] = this.toScreen(pts[0][0], pts[0][1]);
      if (dist2(sx, sy, fx, fy) <= HIT_PX * HIT_PX) { this._finishDraft(); return; }
    }
    pts.push([wx, wy]);
    this._dirty = true;
  }

  _finishDraft() {
    const d = this.draft;
    if (!d) return;
    // A double-click also fired two single clicks on the same spot: drop the duplicate.
    const pts = d.points.filter((p, k) => k === 0 || dist2(p[0], p[1], d.points[k - 1][0], d.points[k - 1][1]) > 0.25);
    this.draft = null;
    if (pts.length < 3) { this.ui.toast(`A ${DRAFT_LABEL[d.kind]} needs at least 3 corners`, 'warn'); this._dirty = true; return; }
    if (d.kind === 'survey') {
      this.surveyPolygon = pts;
      this.surveyStats = null;
      $('sv-hint').textContent = `Survey area: ${pts.length} corners. Set the parameters, then Generate.`;
      const n = this.ui.targets()?.length ?? this.ui.snapshot?.drones.length ?? 1;
      $('sv-drones').value = Math.max(1, Math.min(50, n));
    } else if (d.kind === 'fence') {
      this.fence.inclusion = pts;
      this._setFenceDirty(true);
    } else {
      this.fence.exclusions.push({ name: `NFZ ${this.fence.exclusions.length + 1}`, polygon: pts });
      this._setFenceDirty(true);
    }
    this._renderFence();
    this.setTool('select');
  }

  // ------------------------------------------------------------------ context menu
  _contextMenu(e) {
    const [sx, sy] = this._eventPos(e);
    const [wx, wy] = this.toWorld(sx, sy);
    const items = [];
    const wp = this._hitWaypoint(sx, sy);
    const vertex = this._hitVertex(sx, sy);
    if (this.draft) {
      items.push(['Finish polygon', () => this._finishDraft()], ['Cancel drawing', () => { this.draft = null; this._dirty = true; }]);
    } else if (wp >= 0) {
      this.select(wp);
      items.push(
        ['Insert waypoint before', () => this.insertWaypoint(wp, true)],
        ['Insert waypoint after', () => this.insertWaypoint(wp, false)],
        null,
        ...['WAYPOINT', 'LOITER', 'LAND'].filter((a) => a !== this.waypoints[wp].action)
          .map((a) => [`Make ${a.toLowerCase()}`, () => this.setField(wp, 'action', a)]),
        null,
        ['Delete waypoint', () => this.deleteWaypoint(wp), 'danger'],
      );
    } else if (vertex) {
      const label = vertex.kind === 'survey' ? 'survey corner' : vertex.kind === 'fence' ? 'fence corner' : 'no-fly corner';
      items.push([`Delete ${label}`, () => this._deleteVertex(vertex), 'danger']);
      if (vertex.kind === 'nfz') items.push(['Delete this no-fly zone', () => this.removeZone(vertex.zone), 'danger']);
    } else {
      const seg = this._hitSegment(sx, sy);
      if (seg >= 0) items.push(['Insert waypoint here', () => this.insertAt(seg, wx, wy)]);
      items.push(['Add waypoint here', () => { this.waypoints.push(this._newWaypoint(wx, wy)); this.select(this.waypoints.length - 1); this._changed(); }]);
      items.push(['Append RTL', () => { this.waypoints.push({ ...this._newWaypoint(wx, wy), action: 'RTL' }); this._changed(); }]);
      const sel = this.ui.targets();
      if (sel?.length) {
        items.push(null, [`Fly ${sel.length === 1 ? `D${String(sel[0]).padStart(2, '0')}` : `${sel.length} drones`} here now`,
          () => this.ui.command('goto', { position: [wx, wy, this.defaults.altitude], speed: this.defaults.speed })]);
      }
    }
    this._openMenu(sx, sy, items);
  }

  _openMenu(sx, sy, items) {
    const menu = $('plan-menu');
    menu.innerHTML = '';
    for (const item of items) {
      if (!item) { menu.appendChild(document.createElement('hr')); continue; }
      const [label, fn, cls] = item;
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'menuitem');
      b.textContent = label;
      if (cls) b.className = cls;
      b.onclick = () => { this._closeMenu(); fn(); };
      menu.appendChild(b);
    }
    menu.hidden = false;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    menu.style.left = `${Math.min(sx, w - menu.offsetWidth - 4)}px`;
    menu.style.top = `${Math.min(sy, h - menu.offsetHeight - 4)}px`;
    menu.querySelector('button')?.focus();
  }

  _closeMenu() { const m = $('plan-menu'); if (m && !m.hidden) m.hidden = true; }

  // ------------------------------------------------------------------ editing
  select(k) {
    this.selected = k;
    for (const el of $('pm-wps').querySelectorAll('.wp-item')) el.classList.toggle('selected', Number(el.dataset.k) === k);
    this._dirty = true;
  }

  insertWaypoint(k, before) {
    const wps = this.waypoints;
    const a = wps[before ? k - 1 : k];
    const b = wps[before ? k : k + 1];
    let x;
    let y;
    if (a && b) { x = (a.x + b.x) / 2; y = (a.y + b.y) / 2; } else {
      const ref = wps[k];
      x = ref.x + (before ? -20 : 20); y = ref.y;
    }
    this.insertAt(before ? k : k + 1, x, y);
  }

  insertAt(index, x, y) {
    const wp = this._newWaypoint(x, y);
    const ref = this.waypoints[Math.max(0, index - 1)];
    if (ref) { wp.alt = ref.alt; wp.speed = ref.speed; }
    this.waypoints.splice(index, 0, wp);
    this.select(index);
    this._changed();
  }

  deleteWaypoint(k) {
    this.waypoints.splice(k, 1);
    this.selected = Math.min(this.selected, this.waypoints.length - 1);
    this._changed();
  }

  setField(k, key, value) {
    const w = this.waypoints[k];
    if (!w) return;
    if (key === 'action') {
      w.action = value;
      w.params = value === 'CHANGE_FORMATION' ? { shape: w.params?.shape ?? 'line' }
        : value === 'LOITER' ? { radius: w.params?.radius ?? this.defaults.loiter_radius } : {};
      if (value === 'LOITER' && !w.hold) w.hold = 30;
    } else {
      w[key] = value;
    }
    this._changed();
  }

  _deleteVertex(v) {
    if (v.poly.length <= 3) {
      if (v.kind === 'survey') this.surveyPolygon = null;
      else if (v.kind === 'fence') this.fence.inclusion = null;
      else this.fence.exclusions.splice(v.zone, 1);
    } else {
      v.poly.splice(v.index, 1);
    }
    if (v.kind !== 'survey') this._setFenceDirty(true);
    this._renderFence();
    this._dirty = true;
  }

  removeZone(zi) {
    this.fence.exclusions.splice(zi, 1);
    this._setFenceDirty(true);
    this._renderFence();
  }

  _changed() {
    this.surveyStats = this.mission.tracks.length > 1 ? this.surveyStats : null;
    this._renderTable();
    this._dirty = true;
  }

  // ------------------------------------------------------------------ panel
  _wirePanel() {
    $('pm-name').oninput = (e) => { this.mission.name = e.target.value; };
    $('pm-clear').onclick = () => {
      if (!this.mission.tracks.some((t) => t.length) || confirm('Remove all waypoints?')) {
        this.mission.tracks = [[]]; this.track = 0; this.selected = -1; this.surveyStats = null; this._changed();
      }
    };
    $('pm-reverse').onclick = () => { this.waypoints.reverse(); this._changed(); };
    $('pm-validate').onclick = () => this.validate();
    $('pm-fly').onclick = () => this.fly();
    $('pm-pause').onclick = () => this.ui.command('mission_pause', {}, null);
    $('pm-resume').onclick = () => this.ui.command('mission_resume', {}, null);
    $('pm-abort').onclick = () => this.ui.command('mission_abort', {}, null);
    $('sv-generate').onclick = () => this.generateSurvey();
    $('sv-clear').onclick = () => { this.surveyPolygon = null; this.surveyStats = null; $('sv-stats').textContent = ''; $('sv-hint').textContent = 'Draw a polygon with the Survey tool.'; this._dirty = true; };
    $('gf-enabled').onchange = (e) => { this.fence.enabled = e.target.checked; this._setFenceDirty(true); };
    $('gf-action').onchange = (e) => { this.fence.action = e.target.value; this._setFenceDirty(true); };
    $('gf-ceiling').onchange = (e) => { const v = parseFloat(e.target.value); this.fence.max_altitude = Number.isFinite(v) ? v : null; this._setFenceDirty(true); };
    $('gf-apply').onclick = () => this.applyFence();
    $('gf-clear').onclick = () => {
      this.fence.inclusion = null; this.fence.exclusions = []; this.fence.max_altitude = null;
      this._setFenceDirty(true); this._renderFence();
    };
    $('pm-save').onclick = () => this.saveToLibrary();
    $('pm-load').onclick = () => this.loadFromLibrary();
    $('pm-download').onclick = () => this._download(`mission_${this._slug()}.json`, JSON.stringify(this.missionPayload(), null, 2), 'application/json');
    $('pm-export-qgc').onclick = () => this.exportQgc();
    $('pm-import').onclick = () => $('pm-file').click();
    $('pm-file').onchange = (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) this.importFile(f); };
    $('pm-wps').addEventListener('change', (e) => this._tableEdit(e));
    $('pm-wps').addEventListener('click', (e) => {
      const item = e.target.closest('.wp-item');
      if (!item) return;
      const k = Number(item.dataset.k);
      if (e.target.closest('.del')) { this.deleteWaypoint(k); return; }
      this.select(k);
    });
  }

  _tableEdit(e) {
    const item = e.target.closest('.wp-item');
    const key = e.target.dataset.f;
    if (!item || !key) return;
    const k = Number(item.dataset.k);
    const w = this.waypoints[k];
    const raw = e.target.value;
    if (key === 'action') { this.setField(k, 'action', raw); return; }
    if (key === 'shape' || key === 'radius' || key === 'spacing') {
      const v = key === 'shape' ? raw : parseFloat(raw);
      if (key !== 'shape' && !Number.isFinite(v)) delete w.params[key]; else w.params[key] = v;
      this._dirty = true;
      return;
    }
    const v = parseFloat(raw);
    if (key === 'speed' && raw.trim() === '') w.speed = null;
    else if (Number.isFinite(v)) w[key] = v;
    else e.target.value = w[key] ?? '';
    this._dirty = true;
  }

  _renderTable() {
    const wps = this.waypoints;
    const total = this.mission.tracks.reduce((a, t) => a + t.length, 0);
    $('pm-count').textContent = this.mission.tracks.length > 1 ? `${total} · ${this.mission.tracks.length} tracks` : String(total);
    const tabs = $('pm-tracks');
    tabs.hidden = this.mission.tracks.length <= 1;
    if (!tabs.hidden) {
      tabs.innerHTML = this.mission.tracks.map((t, k) => `<button type="button" data-t="${k}" aria-pressed="${k === this.track}"><i style="background:${TRACK_COLORS[k % TRACK_COLORS.length]}"></i>T${k + 1} · ${t.length}</button>`).join('');
      for (const b of tabs.querySelectorAll('button')) b.onclick = () => { this.track = Number(b.dataset.t); this.selected = -1; this._renderTable(); this._dirty = true; };
    }
    const table = $('pm-wps');
    if (!wps.length) { table.innerHTML = '<div class="empty">Pick the Waypoint tool and click the map.</div>'; return; }
    table.innerHTML = wps.map((w, k) => {
      const opts = ACTIONS.map((a) => `<option value="${a}"${a === w.action ? ' selected' : ''}>${ACTION_SHORT[a]}</option>`).join('');
      let params = '';
      if (w.action === 'LOITER') {
        params = `<div class="wp-params"><label>Radius <input type="number" data-f="radius" min="1" step="1" value="${w.params?.radius ?? this.defaults.loiter_radius}"> m</label></div>`;
      } else if (w.action === 'CHANGE_FORMATION') {
        const shapeOpts = SHAPES.map((s) => `<option value="${s}"${s === w.params?.shape ? ' selected' : ''}>${s}</option>`).join('');
        params = `<div class="wp-params"><label>Shape <select data-f="shape">${shapeOpts}</select></label><label>Spacing <input type="number" data-f="spacing" min="6" step="1" placeholder="—" value="${w.params?.spacing ?? ''}"></label></div>`;
      }
      const where = POSITIONAL.has(w.action) ? `(${fmt(w.x, 0)}, ${fmt(w.y, 0)})` : 'in place';
      return `<div class="wp-item${k === this.selected ? ' selected' : ''}" data-k="${k}" title="${ACTION_SHORT[w.action]} ${where}">
        <div class="wp-row"><span class="n">${k + 1}</span><select data-f="action" aria-label="Action of waypoint ${k + 1}">${opts}</select>
        <input type="number" data-f="alt" step="1" value="${fmt(w.alt, 0)}" aria-label="Altitude">
        <input type="number" data-f="speed" step="0.5" min="0.5" value="${w.speed ?? ''}" placeholder="—" aria-label="Speed">
        <input type="number" data-f="hold" step="1" min="0" value="${fmt(w.hold ?? 0, 0)}" aria-label="Hold time">
        <button type="button" class="del" aria-label="Delete waypoint ${k + 1}">×</button></div>${params}</div>`;
    }).join('');
  }

  _renderAssign(snap) {
    const sel = $('pm-assign');
    const groups = Object.keys(snap.missions?.groups ?? {});
    const n = this.ui.selection.size;
    const key = JSON.stringify([n, groups, snap.drones.length]);
    if (key === this._assignKey) return;
    this._assignKey = key;
    const prev = sel.value;
    sel.innerHTML = `<option value="selection"${n ? '' : ' disabled'}>Selected drones (${n})</option>`
      + `<option value="all">Whole swarm (${snap.drones.length})</option>`
      + groups.map((g) => `<option value="group:${esc(g)}">Group ${esc(g)} (${snap.missions.groups[g].length})</option>`).join('');
    sel.value = [...sel.options].some((o) => o.value === prev && !o.disabled) ? prev : (n ? 'selection' : 'all');
  }

  _target() {
    const v = $('pm-assign').value;
    if (v === 'selection') return { ids: this.ui.targets(), params: {} };
    if (v.startsWith('group:')) return { ids: null, params: { group: v.slice(6) } };
    return { ids: null, params: {} };
  }

  _renderRunStatus(snap) {
    const runs = snap.missions?.runs ?? [];
    const run = [...runs].reverse().find((r) => r.state === 'RUNNING' || r.state === 'PAUSED') ?? runs[runs.length - 1];
    for (const id of ['pm-status', 'fly-mission-status']) {
      const el = $(id);
      if (!run) { el.hidden = true; continue; }
      el.hidden = false;
      el.className = `run-status ${run.state}`;
      const tr = run.tracks[0];
      const detail = run.tracks.length === 1 && tr
        ? `WP ${Math.min(tr.index + 1, tr.total)}/${tr.total} · ${tr.action ?? '—'} · ${tr.phase}`
        : `${run.tracks.filter((t) => t.phase !== 'DONE').length}/${run.tracks.length} tracks flying`;
      el.innerHTML = `<b>${esc(run.name)}</b> · ${run.state}<br>${detail}<div class="bar"><i style="width:${Math.round(run.progress * 100)}%"></i></div>`;
    }
  }

  _report(title, data) {
    const el = $('pm-report');
    el.hidden = false;
    const parts = [];
    if (data.errors?.length) parts.push(`<div class="err">${data.errors.length} error(s)</div><ul>${data.errors.map((m) => `<li class="err">${esc(m)}</li>`).join('')}</ul>`);
    if (data.warnings?.length) parts.push(`<div class="warn"><b>${data.warnings.length} warning(s)</b></div><ul>${data.warnings.map((m) => `<li class="warn">${esc(m)}</li>`).join('')}</ul>`);
    if (!data.errors?.length && !data.warnings?.length) parts.push('<div class="ok">✓ No problems found</div>');
    for (const t of data.tracks ?? []) {
      parts.push(`<div class="est">${data.tracks.length > 1 ? `T${t.track + 1}: ` : ''}${fmt(t.distance_m / 1000, 2)} km · ${Math.round(t.duration_s / 60)} min ${Math.round(t.duration_s % 60)} s · ${fmt(t.energy_wh)} Wh</div>`);
    }
    const low = (data.battery ?? []).filter((b) => !b.ok);
    if ((data.battery ?? []).length) parts.push(`<div class="est">Battery: ${data.battery.length - low.length}/${data.battery.length} drones OK</div>`);
    el.innerHTML = `<div><b>${esc(title)}</b></div>${parts.join('')}`;
  }

  // ------------------------------------------------------------------ backend actions
  _hasWaypoints() {
    if (!this.mission.tracks.some((t) => t.length)) { this.ui.toast('The mission has no waypoints', 'warn'); return false; }
    return true;
  }

  async validate() {
    if (!this._hasWaypoints()) return null;
    const { ids, params } = this._target();
    const res = await this.ui.command('mission_validate', { ...params, mission: this.missionPayload(), formation: $('pm-formation').checked }, ids);
    if (res?.data?.warnings) this._report('Validation', res.data);
    return res;
  }

  async fly(force = false) {
    if (!this._hasWaypoints()) return;
    const { ids, params } = this._target();
    const payload = { ...params, mission: this.missionPayload(), formation: $('pm-formation').checked, force,
      apply_geofence: $('pm-apply-fence').checked };
    const res = await this.ui.command('mission_start', payload, ids);
    if (!res) return;
    if (res.data?.warnings) this._report(res.success ? 'Mission uploaded' : 'Upload blocked', res.data);
    if (!res.success && res.data?.needs_confirmation) {
      const ok = await this.ui.confirmDialog({
        title: 'Fly mission with warnings?',
        html: `<p>The pre-flight check found ${res.data.warnings.length} warning(s):</p><ul>${res.data.warnings.map((w) => `<li>${esc(w)}</li>`).join('')}</ul>`,
        okLabel: 'Fly anyway',
      });
      if (ok) this.fly(true);
    }
  }

  async generateSurvey() {
    if (!this.surveyPolygon) { this.ui.toast('Draw the survey area first (Survey tool)', 'warn'); return; }
    const num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : null; };
    const body = {
      polygon: this.surveyPolygon, altitude: num('sv-alt') ?? 30, speed: num('sv-speed'), drones: Math.max(1, Math.round(num('sv-drones') ?? 1)),
      line_spacing: num('sv-spacing'), overlap: num('sv-spacing') ? null : num('sv-overlap'), angle: num('sv-angle'),
      finish: $('sv-finish').value, name: this.mission.name || 'survey',
    };
    const res = await this._fetchJson('/api/missions/survey', body);
    if (!res) return;
    const polygon = this.surveyPolygon;
    this.loadMission(res.mission);
    this.surveyPolygon = polygon;
    this.surveyStats = res.mission.survey;
    const s = res.stats;
    $('sv-stats').textContent = `${s.lines} lines · spacing ${fmt(s.line_spacing)} m · ${fmt(s.area_m2 / 10000, 2)} ha · ${s.tracks} track(s): ${s.track_lengths.map((l) => `${fmt(l / 1000, 2)} km`).join(', ')}`;
    $('pm-formation').checked = false;
    this.ui.toast(`Survey generated: ${s.tracks} track(s), ${s.lines} lines`, 'ok');
  }

  _fenceDefined() { return !!(this.fence.inclusion || this.fence.exclusions.length || this.fence.max_altitude); }

  _fencePayload() {
    return { enabled: this.fence.enabled, action: this.fence.action, inclusion: this.fence.inclusion,
      exclusions: this.fence.exclusions.map((z) => ({ name: z.name, polygon: z.polygon })), max_altitude: this.fence.max_altitude };
  }

  async applyFence() {
    const payload = this._fencePayload();
    const live = this.ui.snapshot?.missions?.geofence;
    // Turning a live fence off (or deleting its polygons) removes a safety net: ask first.
    const disabling = live?.enabled && !payload.enabled;
    const removing = live && ((live.inclusion && !payload.inclusion) || (live.exclusions?.length ?? 0) > payload.exclusions.length);
    if (disabling || removing) {
      const what = disabling ? 'Disable the geofence' : 'Remove geofence zones';
      const ok = await this.ui.confirmDialog({
        title: what, okLabel: disabling ? 'DISABLE FENCE' : 'REMOVE ZONES', danger: true,
        html: disabling ? '<p>The drones will <b>no longer be held inside the fence</b> or kept out of no-fly zones.</p>'
          : `<p>The applied fence loses ${live.inclusion && !payload.inclusion ? 'its <b>inclusion polygon</b>' : ''}${live.inclusion && !payload.inclusion && live.exclusions.length > payload.exclusions.length ? ' and ' : ''}${live.exclusions.length > payload.exclusions.length ? `<b>${live.exclusions.length - payload.exclusions.length} no-fly zone(s)</b>` : ''}.</p>`,
      });
      if (!ok) return;
    }
    const res = await this.ui.command('set_geofence', payload, null);
    if (res?.success) this._setFenceDirty(false);
  }

  _setFenceDirty(on) {
    this.fenceDirty = on;
    $('gf-dirty').hidden = !on;
    this._dirty = true;
  }

  _renderFence() {
    $('gf-enabled').checked = !!this.fence.enabled;
    $('gf-action').value = this.fence.action ?? 'RTL';
    if (document.activeElement !== $('gf-ceiling')) $('gf-ceiling').value = this.fence.max_altitude ?? '';
    const rows = [];
    if (this.fence.inclusion) rows.push(`<div><i style="background:#F5C542"></i><span>Inclusion · ${this.fence.inclusion.length} corners</span><button type="button" data-z="inc" aria-label="Remove inclusion fence">×</button></div>`);
    this.fence.exclusions.forEach((z, k) => rows.push(`<div><i style="background:#FF4D4D"></i><span>${esc(z.name)} · ${z.polygon.length} corners</span><button type="button" data-z="${k}" aria-label="Remove ${esc(z.name)}">×</button></div>`));
    const list = $('gf-zones');
    list.innerHTML = rows.join('');
    for (const b of list.querySelectorAll('button')) {
      b.onclick = () => {
        if (b.dataset.z === 'inc') this.fence.inclusion = null; else this.fence.exclusions.splice(Number(b.dataset.z), 1);
        this._setFenceDirty(true);
        this._renderFence();
      };
    }
    this._dirty = true;
  }

  _slug() { return (this.mission.name || 'mission').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'mission'; }

  async _fetchJson(url, body, method = 'POST') {
    try {
      const res = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: body == null ? undefined : JSON.stringify(body) });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) { this.ui.toast(data.detail ?? `HTTP ${res.status}`, 'error'); return null; }
      return data;
    } catch (err) {
      this.ui.toast(`${url}: ${err.message}`, 'error');
      return null;
    }
  }

  async refreshLibrary() {
    try {
      const res = await fetch('/api/missions/files');
      const files = res.ok ? await res.json() : [];
      const sel = $('pm-library');
      sel.innerHTML = files.length ? files.map((f) => `<option value="${esc(f.file)}"${f.valid ? '' : ' disabled'}>${esc(f.name ?? f.file)}${f.valid ? ` · ${f.waypoints} wp` : ' (invalid)'}</option>`).join('')
        : '<option value="" disabled selected>No saved missions</option>';
    } catch { /* offline backend: the library just stays empty */ }
  }

  async saveToLibrary() {
    if (!this._hasWaypoints()) return;
    const res = await this._fetchJson('/api/missions/files', { mission: this.missionPayload(), overwrite: true });
    if (res) { this.ui.toast(`Saved ${res.file}`, 'ok'); this.refreshLibrary(); }
  }

  async loadFromLibrary() {
    const file = $('pm-library').value;
    if (!file) return;
    try {
      const res = await fetch(`/api/missions/files/${encodeURIComponent(file)}`);
      const data = await res.json();
      if (!res.ok) { this.ui.toast(data.detail ?? `HTTP ${res.status}`, 'error'); return; }
      this.loadMission(data);
      this.ui.toast(`Loaded ${file}`, 'ok');
    } catch (err) { this.ui.toast(err.message, 'error'); }
  }

  async exportQgc() {
    if (!this._hasWaypoints()) return;
    const payload = this.missionPayload();
    const tracks = this.mission.tracks.filter((t) => t.length).length;
    for (let t = 0; t < tracks; t += 1) {
      try {
        const res = await fetch(`/api/missions/export/qgc?track=${t}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (!res.ok) { const d = await res.json().catch(() => ({})); this.ui.toast(d.detail ?? `HTTP ${res.status}`, 'error'); return; }
        const text = await res.text();
        this._download(`${this._slug()}${tracks > 1 ? `_track${t + 1}` : ''}.waypoints`, text, 'text/plain');
      } catch (err) { this.ui.toast(err.message, 'error'); return; }
    }
    this.ui.toast(`Exported ${tracks} QGC WPL 110 file(s)`, 'ok');
  }

  async importFile(file) {
    const text = await file.text();
    if (text.trimStart().startsWith('QGC WPL')) {
      const res = await this._fetchJson('/api/missions/import/qgc', { text, name: file.name.replace(/\.[^.]+$/, '') });
      if (!res) return;
      this.loadMission(res.mission);
      this.ui.toast(`Imported ${file.name}${res.warnings.length ? ` (${res.warnings.length} item(s) skipped)` : ''}`, res.warnings.length ? 'warn' : 'ok');
      return;
    }
    let data;
    try { data = JSON.parse(text); } catch { this.ui.toast(`${file.name} is neither QGC WPL nor JSON`, 'error'); return; }
    const res = await this._fetchJson('/api/missions/normalize', data);     // schema validation on the server
    if (res) { this.loadMission(res); this.ui.toast(`Loaded ${file.name}`, 'ok'); }
  }

  _download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async _fetchActive() {
    try {
      const res = await fetch('/api/missions/active');
      if (!res.ok) return;
      const data = await res.json();
      this.activePaths = data.paths ?? [];
      this.ui.onMissionPaths?.(this.activePaths);
      this._dirty = true;
    } catch { /* ignore: next version change retries */ }
  }

  _readout(wx, wy) {
    let text = `E ${fmt(wx, 1)}  N ${fmt(wy, 1)} m`;
    const m = this.measure;
    if (m) {
      const b = m.b ?? [wx, wy];
      text += `  ·  ${fmt(Math.hypot(b[0] - m.a[0], b[1] - m.a[1]), 1)} m @ ${fmt(bearing(m.a[0], m.a[1], b[0], b[1]), 0)}°`;
    }
    $('plan-readout').textContent = text;
  }

  // ------------------------------------------------------------------ drawing
  _draw() {
    const c = this.canvas;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = c.clientWidth;
    const h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = this.ctx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = '#3a3a3a';
    g.fillRect(0, 0, w, h);
    this._drawTerrain(g);
    this._drawGrid(g, w, h);
    if (this.world) this._drawWorld(g);
    this._drawObstacles(g);
    this._drawFence(g);
    this._drawPolygon(g, this.surveyPolygon, '#7FD1B9', 'rgba(127,209,185,0.10)', [6, 4]);
    for (const p of this.activePaths) this._drawActive(g, p);
    this.mission.tracks.forEach((t, k) => this._drawTrack(g, t, k));
    this._drawDrones(g);
    this._drawDraft(g);
    this._drawMeasure(g);
  }

  /** Hillshade of the terrain grid (built once per scene version, then drawn with the view transform). */
  _drawTerrain(g) {
    const world3d = this.ui.world3d;
    const grid = world3d?.grid;
    if (!grid) return;
    if (this._hillVersion !== world3d.version) {
      this._hillVersion = world3d.version;
      const c = document.createElement('canvas');
      c.width = grid.cols; c.height = grid.rows;
      const ctx = c.getContext('2d');
      const img = ctx.createImageData(grid.cols, grid.rows);
      const H = (r, col) => grid.heights[Math.min(grid.rows - 1, Math.max(0, r)) * grid.cols + Math.min(grid.cols - 1, Math.max(0, col))];
      const dx = grid.size_x / (grid.cols - 1);
      const span = Math.max(grid.max - grid.min, 1);
      for (let r = 0; r < grid.rows; r += 1) {
        for (let col = 0; col < grid.cols; col += 1) {
          const h = H(r, col);
          const sx = (H(r, col + 1) - H(r, col - 1)) / (2 * dx);
          const sy = (H(r + 1, col) - H(r - 1, col)) / (2 * dx);
          const shade = Math.max(0, Math.min(1, 0.55 - 0.9 * sx + 0.9 * sy));   // light from the north-west
          const tint = (h - grid.min) / span;
          const px = ((grid.rows - 1 - r) * grid.cols + col) * 4;                 // image row 0 = north edge
          img.data[px] = 55 + 60 * shade + 40 * tint;
          img.data[px + 1] = 62 + 60 * shade + 25 * tint;
          img.data[px + 2] = 50 + 50 * shade;
          img.data[px + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      this._hillshade = c;
    }
    const [ax, ay] = this.toScreen(grid.x_min, grid.y_min + grid.size_y);
    const [bx, by] = this.toScreen(grid.x_min + grid.size_x, grid.y_min);
    g.imageSmoothingEnabled = true;
    g.globalAlpha = 0.55;
    g.drawImage(this._hillshade, ax, ay, bx - ax, by - ay);
    g.globalAlpha = 1;
  }

  _drawObstacles(g) {
    const obstacles = this.ui.world3d?.obstacles ?? [];
    for (const o of obstacles) {
      if (o.kind === 'building') {
        const r = (o.rotation ?? 0) * Math.PI / 180;
        const c = Math.cos(r); const s = Math.sin(r);
        const hw = o.width / 2; const hd = o.depth / 2;
        g.beginPath();
        [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]].forEach(([a, b], k) => {
          const [sx, sy] = this.toScreen(o.x + c * a - s * b, o.y + s * a + c * b);
          if (k) g.lineTo(sx, sy); else g.moveTo(sx, sy);
        });
        g.closePath();
        g.fillStyle = 'rgba(160,166,176,0.55)';
        g.fill();
        g.strokeStyle = '#d0d4da';
        g.stroke();
      } else {
        const [sx, sy] = this.toScreen(o.x, o.y);
        g.beginPath();
        g.arc(sx, sy, Math.max(2, o.radius * this.view.scale), 0, Math.PI * 2);
        g.fillStyle = o.kind === 'tree' ? 'rgba(90,150,80,0.7)' : 'rgba(220,120,120,0.8)';
        g.fill();
      }
      if (o.kind !== 'tree' && this.view.scale > 0.6) {
        const [sx, sy] = this.toScreen(o.x, o.y);
        g.fillStyle = 'rgba(242,237,237,0.9)';
        g.font = '600 10px ui-monospace, Consolas, monospace';
        g.textAlign = 'center';
        g.fillText(`${o.name} ${Math.round(o.height)} m`, sx, sy + 3);
        g.textAlign = 'left';
      }
    }
  }

  _drawGrid(g, w, h) {
    const target = 80 / this.view.scale;                        // ~80 px between lines
    const steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const step = steps.find((s) => s >= target) ?? 10000;
    const [x0, y1] = this.toWorld(0, 0);
    const [x1, y0] = this.toWorld(w, h);
    g.lineWidth = 1;
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    g.beginPath();
    for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) { const [sx] = this.toScreen(x, 0); g.moveTo(sx + 0.5, 0); g.lineTo(sx + 0.5, h); }
    for (let y = Math.floor(y0 / step) * step; y <= y1; y += step) { const [, sy] = this.toScreen(0, y); g.moveTo(0, sy + 0.5); g.lineTo(w, sy + 0.5); }
    g.stroke();
    // scale bar
    g.fillStyle = 'rgba(242,237,237,0.85)';
    g.font = '600 11px ui-monospace, Consolas, monospace';
    const px = step * this.view.scale;
    g.fillRect(w - px - 16, h - 18, px, 3);
    g.fillText(`${step >= 1000 ? `${step / 1000} km` : `${step} m`}`, w - px - 16, h - 24);
  }

  _drawWorld(g) {
    const { min, max } = this.world.bounds;
    const [ax, ay] = this.toScreen(min.x, max.y);
    const [bx, by] = this.toScreen(max.x, min.y);
    g.strokeStyle = 'rgba(91,141,239,0.55)';
    g.setLineDash([8, 6]);
    g.strokeRect(ax, ay, bx - ax, by - ay);
    g.setLineDash([]);
    const home = this.world.home;
    const [hx, hy] = this.toScreen(home.x, home.y);
    g.strokeStyle = '#5B8DEF';
    g.lineWidth = 2;
    g.beginPath(); g.arc(hx, hy, 9, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.arc(hx, hy, 3, 0, Math.PI * 2); g.fillStyle = '#5B8DEF'; g.fill();
    g.fillStyle = '#E6CFA0';
    g.font = '700 11px ui-monospace, Consolas, monospace';
    g.fillText('HOME', hx + 12, hy + 4);
    // north arrow
    g.fillStyle = 'rgba(242,237,237,0.8)';
    g.fillText('N ↑', 16, this.canvas.clientHeight - 40);
    g.lineWidth = 1;
  }

  _drawPolygon(g, poly, stroke, fill, dash = [], label = null) {
    if (!poly?.length) return;
    g.beginPath();
    poly.forEach(([x, y], k) => { const [sx, sy] = this.toScreen(x, y); if (k) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
    g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    g.setLineDash(dash);
    g.strokeStyle = stroke;
    g.lineWidth = 2;
    g.stroke();
    g.setLineDash([]);
    g.lineWidth = 1;
    if (this.tool === 'select') {
      for (const [x, y] of poly) { const [sx, sy] = this.toScreen(x, y); g.fillStyle = stroke; g.fillRect(sx - 3, sy - 3, 6, 6); }
    }
    if (label) {
      const cx = poly.reduce((a, p) => a + p[0], 0) / poly.length;
      const cy = poly.reduce((a, p) => a + p[1], 0) / poly.length;
      const [sx, sy] = this.toScreen(cx, cy);
      g.fillStyle = stroke;
      g.font = '700 11px ui-monospace, Consolas, monospace';
      g.textAlign = 'center';
      g.fillText(label, sx, sy);
      g.textAlign = 'left';
    }
  }

  _drawFence(g) {
    const f = this.fence;
    const faded = !f.enabled;
    this._drawPolygon(g, f.inclusion, faded ? 'rgba(245,197,66,0.55)' : '#F5C542', null, faded ? [4, 4] : [], null);
    for (const z of f.exclusions) {
      this._drawPolygon(g, z.polygon, faded ? 'rgba(255,77,77,0.6)' : '#FF4D4D', faded ? 'rgba(255,77,77,0.10)' : 'rgba(255,77,77,0.22)', faded ? [4, 4] : [], z.name);
    }
  }

  _drawTrack(g, wps, k) {
    if (!wps.length) return;
    const editing = k === this.track;
    const color = this.mission.tracks.length > 1 ? TRACK_COLORS[k % TRACK_COLORS.length] : '#D9BC84';
    const pts = this._pathPoints(wps);
    const home = this.world?.home;
    g.strokeStyle = color;
    g.globalAlpha = editing ? 1 : 0.55;
    g.lineWidth = editing ? 2 : 1.5;
    g.beginPath();
    pts.forEach((p, i) => { const [sx, sy] = this.toScreen(p.x, p.y); if (i) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
    g.stroke();
    // dashed return leg for an RTL at the end
    if (home && wps[wps.length - 1]?.action === 'RTL' && pts.length) {
      const last = pts[pts.length - 1];
      const [ax, ay] = this.toScreen(last.x, last.y);
      const [bx, by] = this.toScreen(home.x, home.y);
      g.setLineDash([5, 5]);
      g.strokeStyle = ACTION_COLOR.RTL;
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
      g.setLineDash([]);
    }
    // direction arrows at segment midpoints
    for (let i = 1; i < pts.length; i += 1) {
      const [ax, ay] = this.toScreen(pts[i - 1].x, pts[i - 1].y);
      const [bx, by] = this.toScreen(pts[i].x, pts[i].y);
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 40) continue;
      const ang = Math.atan2(by - ay, bx - ax);
      const mx = (ax + bx) / 2;
      const my = (ay + by) / 2;
      g.fillStyle = color;
      g.beginPath();
      g.moveTo(mx + 6 * Math.cos(ang), my + 6 * Math.sin(ang));
      g.lineTo(mx + 5 * Math.cos(ang + 2.5), my + 5 * Math.sin(ang + 2.5));
      g.lineTo(mx + 5 * Math.cos(ang - 2.5), my + 5 * Math.sin(ang - 2.5));
      g.fill();
    }
    // markers (numbered)
    const big = editing && (this.mission.tracks.length === 1 || wps.length < 60);
    for (const p of pts) {
      const [sx, sy] = this.toScreen(p.x, p.y);
      const r = big ? 9 : 4;
      if (p.action === 'LOITER' && editing) {
        const rad = (p.params?.radius ?? this.defaults.loiter_radius) * this.view.scale;
        g.strokeStyle = ACTION_COLOR.LOITER;
        g.setLineDash([3, 3]);
        g.beginPath(); g.arc(sx, sy, rad, 0, Math.PI * 2); g.stroke();
        g.setLineDash([]);
      }
      g.beginPath();
      g.arc(sx, sy, r, 0, Math.PI * 2);
      g.fillStyle = ACTION_COLOR[p.action] ?? color;
      g.fill();
      if (editing && p.index === this.selected) { g.strokeStyle = '#fff'; g.lineWidth = 2.5; g.stroke(); g.lineWidth = 1; }
      if (big) {
        g.fillStyle = '#1e1e1e';
        g.font = '700 10px ui-monospace, Consolas, monospace';
        g.textAlign = 'center';
        g.fillText(String(p.index + 1), sx, sy + 3.5);
        g.textAlign = 'left';
        g.fillStyle = 'rgba(242,237,237,0.85)';
        g.fillText(`${Math.round(p.alt)} m${p.action !== 'WAYPOINT' ? ` ${ACTION_SHORT[p.action]}` : ''}`, sx + 12, sy - 8);
      }
    }
    // in-place items (RTL, formation change) listed next to the preceding marker
    g.globalAlpha = 1;
  }

  _drawActive(g, path) {
    const pts = this._pathPoints(path.waypoints);
    if (pts.length < 2) return;
    g.strokeStyle = 'rgba(61,220,151,0.8)';
    g.lineWidth = 1;
    g.setLineDash([2, 4]);
    g.beginPath();
    pts.forEach((p, i) => { const [sx, sy] = this.toScreen(p.x, p.y); if (i) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
    g.stroke();
    g.setLineDash([]);
  }

  _drawDrones(g) {
    const snap = this.ui.snapshot;
    if (!snap) return;
    const breached = new Set(snap.missions?.geofence?.breached ?? []);
    for (const d of snap.drones) {
      const [sx, sy] = this.toScreen(d.position.x, d.position.y);
      const vis = VISUAL_STATES[classify(d)];
      const ang = (d.heading - 90) * Math.PI / 180;       // compass -> screen angle (0 = East, clockwise)
      g.save();
      g.translate(sx, sy);
      g.rotate(ang);
      g.beginPath();
      g.moveTo(8, 0); g.lineTo(-6, 5); g.lineTo(-3, 0); g.lineTo(-6, -5); g.closePath();
      g.fillStyle = vis.color;
      g.fill();
      g.strokeStyle = this.ui.selection.has(d.drone_id) ? '#fff' : 'rgba(0,0,0,0.6)';
      g.lineWidth = this.ui.selection.has(d.drone_id) ? 2 : 1;
      g.stroke();
      g.restore();
      if (breached.has(d.drone_id)) {
        g.strokeStyle = '#FF4D4D';
        g.beginPath(); g.arc(sx, sy, 12, 0, Math.PI * 2); g.stroke();
      }
      if (this.view.scale > 0.25) {
        g.fillStyle = 'rgba(242,237,237,0.85)';
        g.font = '600 10px ui-monospace, Consolas, monospace';
        g.fillText(d.name, sx + 9, sy + 12);
      }
    }
  }

  _drawDraft(g) {
    const d = this.draft;
    if (!d) return;
    const color = { survey: '#7FD1B9', fence: '#F5C542', nfz: '#FF4D4D' }[d.kind];
    const pts = this.hover ? [...d.points, this.hover] : d.points;
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.setLineDash([6, 4]);
    g.beginPath();
    pts.forEach(([x, y], k) => { const [sx, sy] = this.toScreen(x, y); if (k) g.lineTo(sx, sy); else g.moveTo(sx, sy); });
    g.stroke();
    g.setLineDash([]);
    d.points.forEach(([x, y], k) => {
      const [sx, sy] = this.toScreen(x, y);
      g.fillStyle = k === 0 ? '#fff' : color;
      g.beginPath(); g.arc(sx, sy, k === 0 ? 6 : 4, 0, Math.PI * 2); g.fill();
    });
    g.lineWidth = 1;
  }

  _drawMeasure(g) {
    const m = this.measure;
    if (!m) return;
    const b = m.b ?? this.hover;
    if (!b) return;
    const [ax, ay] = this.toScreen(m.a[0], m.a[1]);
    const [bx, by] = this.toScreen(b[0], b[1]);
    g.strokeStyle = '#fff';
    g.setLineDash([4, 3]);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
    g.setLineDash([]);
    const text = `${fmt(Math.hypot(b[0] - m.a[0], b[1] - m.a[1]), 1)} m · ${fmt(bearing(m.a[0], m.a[1], b[0], b[1]), 0)}°`;
    g.font = '700 12px ui-monospace, Consolas, monospace';
    const tw = g.measureText(text).width;
    g.fillStyle = 'rgba(30,30,30,0.85)';
    g.fillRect((ax + bx) / 2 - tw / 2 - 6, (ay + by) / 2 - 22, tw + 12, 18);
    g.fillStyle = '#fff';
    g.fillText(text, (ax + bx) / 2 - tw / 2, (ay + by) / 2 - 9);
  }
}
