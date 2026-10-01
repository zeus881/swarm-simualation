// Dashboard panels, operator controls and 3D-view interaction.
import { AlertsPanel } from './alerts.js';
import { LineChart, SERIES_COLORS, TelemetryHistory } from './charts.js';
import { Hud } from './hud.js';
import { Minimap } from './minimap.js';
import { bearing } from './plan.js';
import { VISUAL_STATES, batteryColor, classify } from './state.js';

const NATO = ['Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 'Hotel', 'India', 'Juliett', 'Kilo', 'Lima'];
// Swarm health card: operator-facing buckets (a drone counts in the first bucket it matches).
const HEALTH_BUCKETS = [
  ['FAILED', 'Failed', '#c43030'],
  ['COMM_LOST', 'Comm lost', '#8b93a7'],
  ['LOW_BATTERY', 'Low batt', '#ff8c42'],
  ['WARNING', 'Warning', '#f5c542'],
  ['ONLINE', 'Online', '#3ddc97'],
];

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function clock(t) {
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

// Commands that need an explicit confirmation when they address the whole swarm (no selection).
const DANGEROUS_ALL = {
  emergency_stop: { title: 'EMERGENCY STOP — all drones', ok: 'E-STOP ALL',
    html: (n) => `<p>This sends <b>EMERGENCY STOP</b> to <b>all ${n} drones</b>. Airborne drones perform an emergency descent (or cut motors, per configuration).</p><p>Select drones first to stop only some of them.</p>` },
  land: { title: 'Land all drones', ok: 'LAND ALL',
    html: (n) => `<p>Every one of the <b>${n} drones</b> lands where it is now, including any flying a mission.</p><p>Select drones first to land only some of them.</p>` },
};

const MAX_EVENTS = 300;
const LAYOUT_KEY = 'gandiv.gcs.layout.v1';
const VIEWPORT_BG = 0x4a4a4a;
const VIEWPORT_GROUND = 0x3c3c3c;

// Inspector fields: [key, label, wide?]. Values are filled by _inspectorValues().
const INSPECTOR_SECTIONS = [
  ['State', [['mode', 'Mode'], ['task', 'Task'], ['armed', 'Armed'], ['health', 'Health'], ['link', 'Link'], ['collision', 'Collision']]],
  ['Position', [['local', 'Local x / y / z (m)', true], ['lat', 'Latitude'], ['lon', 'Longitude'], ['agl', 'Alt AGL'], ['amsl', 'Alt AMSL'], ['target', 'Target (m)', true]]],
  ['Motion', [['speed', 'Speed'], ['heading', 'Heading'], ['vel', 'Velocity (m/s)', true], ['attitude', 'Roll / Pitch'], ['yawrate', 'Yaw rate'], ['accel', 'Accel (m/s²)', true]]],
  ['Energy & safety', [['battery', 'Battery'], ['voltage', 'Voltage'], ['power', 'Power'], ['nearest', 'Nearest'], ['ftime', 'Flight time'], ['dist', 'Distance'], ['neighbors', 'Neighbours', true], ['home', 'Home (m)', true]]],
  ['Link & navigation', [['linkq', 'Link quality'], ['age', 'Telemetry age'], ['gps', 'GPS', true], ['esterr', 'Estimate error / σ', true], ['failures', 'Injected failures', true]]],
];

export class Dashboard {
  constructor({ link, scene, layer, overlays = null }) {
    this.link = link;
    this.scene = scene;
    this.layer = layer;
    this.overlays = overlays;
    this.world3d = null;          // WorldScene (terrain + obstacles), attached by main.js
    this.planner = null;          // MissionPlanner, attached by main.js
    this.view = 'fly';
    this.snapshot = null;
    this.selection = new Set();
    this.rows = new Map();
    this.events = [];
    this.lastEventSeq = 0;
    this.runId = null;
    this._frames = 0;
    this._fpsTime = 0;
    this._runShown = null;
    this._followShown = null;
    this._inspectorId = null;
    this._inspectorFields = null;
    this._layout = this._loadLayout();
    // Stage 3: operator awareness
    this.hud = new Hud($('hud'));
    this.history = new TelemetryHistory();
    this.charts = {
      alt: new LineChart($('chart-alt'), { title: 'Altitude AGL', unit: 'm', min: 0 }),
      speed: new LineChart($('chart-speed'), { title: 'Speed', unit: 'm/s', min: 0 }),
      battery: new LineChart($('chart-battery'), { title: 'Battery', unit: '%', min: 0, max: 100, decimals: 0 }),
      sep: new LineChart($('chart-sep'), { title: 'Min separation', unit: 'm', min: 0 }),
    };
    this.alertsPanel = new AlertsPanel(this);
    this.minimap = new Minimap($('minimap'), this.scene, (x, y) => { this.setCamera('free'); this.scene.lookAt(x, y); });
    this.dockPane = 'events';
    this.measureMode = false;
    this.measurePts = [];
    this._chartClock = 0;
    this._miniClock = 0;
    this._buildLegend();
    this._wireLogo();
    this._wireControls();
    this._wireAwareness();
    this._wireViewport();
    this._wireKeyboard();
    this._wireLayout();
    this._wireTooltips();
    this._themeViewport();
    this.user = null;
    this.replay = null;           // ReplayController, attached by main.js
    setInterval(() => { if (this.dockPane === 'audit' && !this._layout.dock) this._refreshAudit(); }, 3000);
  }

  // ------------------------------------------------------------------ incoming data
  onConnection({ connected, wasConnected }) {
    const pill = $('conn-pill');
    pill.classList.toggle('ok', connected);
    pill.classList.toggle('bad', !connected);
    $('conn-text').textContent = connected ? 'LINK OK' : 'NO LINK';
    if (connected) $('overlay').hidden = true;
    else {
      $('overlay').hidden = false;
      $('overlay-text').textContent = wasConnected ? 'Connection lost — reconnecting…' : 'Connecting to simulation…';
    }
  }

  onLatency({ rtt }) {
    $('m-latency').textContent = `${rtt.toFixed(0)} ms`;
  }

  onTelemetry(snap) {
    if (!snap.replay) this.lastLiveRunId = snap.run_id;
    if (this.runId !== snap.run_id) {
      if (this.runId !== null) {
        this.layer.reset();
        this.events = [];
        $('event-log').innerHTML = '';
      }
      this.runId = snap.run_id;
      this.lastEventSeq = 0;
    }
    const firstFrame = this.snapshot === null;
    this.snapshot = snap;
    this.scene.buildWorld(snap.world);
    this._themeViewport();
    if (firstFrame) this.scene.resetView();
    const empty = snap.drones.length === 0;
    if (empty !== this._emptyShown) {
      this._emptyShown = empty;
      const wm = $('vp-watermark');
      wm.hidden = !empty || wm.dataset.missing === 'true';
    }
    this.layer.sync(snap.drones, snap.world);
    // Drop selections of drones that no longer exist.
    const ids = new Set(snap.drones.map((d) => d.drone_id));
    const kept = [...this.selection].filter((id) => ids.has(id));
    if (kept.length !== this.selection.size) this.setSelection(kept);

    this._renderStatus(snap);
    this._renderFleet(snap);
    this._renderInspector(snap);
    this._renderMission(snap);
    this._appendEvents(snap.events);
    this.overlays?.setGeofence(snap.missions?.geofence, snap.world);
    if (!snap.replay) this.planner?.onTelemetry(snap);      // a replayed fence must not overwrite the editor
    this.history.push(snap);
    this.alertsPanel.update(snap);
    this._renderHealth(snap);
    this._renderGroups(snap);
    this._renderFailures(snap);
    this.world3d?.sync(snap.world);
    const sel = [...this.selection];
    this.hud.setDrone(sel.length ? snap.drones.find((d) => d.drone_id === sel[0]) ?? null : null, snap.world.battery_thresholds);
  }

  onMissionPaths(paths) {
    this.overlays?.setMissionPaths(paths, this.snapshot?.world);
  }

  /** Clear the event log (replay seeks backwards). */
  resetEvents() {
    this.events = [];
    this.lastEventSeq = 0;
    $('event-log').innerHTML = '';
  }

  // ------------------------------------------------------------------ views (FLY / PLAN / REPLAY)
  setView(view) {
    if (view === this.view) return;
    const wasReplay = this.view === 'replay';
    this.view = view;
    const plan = view === 'plan';
    for (const v of ['fly', 'plan', 'replay']) $(`tab-${v}`).setAttribute('aria-selected', String(v === view));
    $('fly-panel').hidden = plan;
    $('right-rail-title').textContent = plan ? 'Mission plan' : (view === 'replay' ? 'Inspector (replay)' : 'Inspector');
    $('inspector-name').hidden = plan;
    this.planner?.setVisible(plan);
    if (wasReplay) this.replay?.exit();
    if (view === 'replay') this.replay?.enter();
    this._hideTip();
  }

  // ------------------------------------------------------------------ Stage 5: user / role
  setUser(user, securityEnabled) {
    this.user = user;
    const observer = user && !user.can_control;
    document.body.classList.toggle('observer', !!observer);
    $('user-chip').hidden = !securityEnabled || !user;
    if (user) {
      $('user-name').textContent = user.username;
      $('user-role').textContent = observer ? 'OBSERVER · READ-ONLY' : 'OPERATOR';
      $('user-chip').classList.toggle('observer', !!observer);
    }
  }

  get canControl() { return !this.user || !!this.user.can_control; }

  async _refreshAudit() {
    try {
      const res = await fetch('/api/audit?limit=200');
      if (!res.ok) return;
      const rows = await res.json();
      const key = rows.length ? `${rows.length}|${rows[rows.length - 1].time}` : '0';
      if (key === this._auditKey) return;
      this._auditKey = key;
      const target = (t) => (t == null ? '' : Array.isArray(t) ? t.map((i) => `D${String(i).padStart(2, '0')}`).join(' ') : String(t));
      $('audit-list').innerHTML = rows.length ? rows.slice().reverse().map((r) => `<div class="audit-row${r.success === false ? ' bad' : ''}">`
        + `<span class="t">${esc(r.time.slice(11, 23))}</span><span class="u">${esc(r.user ?? '—')}</span><span class="r">${esc(r.role ?? '')}</span>`
        + `<span class="a">${esc(r.action)}</span><span class="g">${esc(target(r.target))}</span>`
        + `<span class="m" title="${esc(r.result ?? '')}">${esc(r.result ?? '')}</span></div>`).join('')
        : '<div class="empty">No audited actions yet.</div>';
    } catch { /* offline */ }
  }

  get planVisible() { return this.view === 'plan'; }

  /** Modal confirmation. Resolves true when the operator confirms. */
  confirmDialog({ title, html, okLabel = 'Confirm', danger = false }) {
    const dlg = $('confirm-dialog');
    $('confirm-title').textContent = title;
    $('confirm-body').innerHTML = html;
    $('confirm-ok').textContent = okLabel;
    dlg.classList.toggle('danger', danger);
    return new Promise((resolve) => {
      dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok'), { once: true });
      dlg.returnValue = 'cancel';
      dlg.showModal();
      $('confirm-cancel').focus();
    });
  }

  animate(dt) {
    this._frames += 1;
    this._fpsTime += dt;
    if (this._fpsTime >= 0.5) {
      $('m-fps').textContent = (this._frames / this._fpsTime).toFixed(0);
      this._frames = 0;
      this._fpsTime = 0;
    }
    if (this.planVisible || !this.snapshot) return;
    // Camera presets: chase the (first) selected drone, orbit / top-view the airborne swarm centre.
    const first = this.selection.size ? [...this.selection][0] : null;
    const pos = first != null ? this.layer.positionOf(first) : null;
    const data = first != null ? this.snapshot.drones.find((d) => d.drone_id === first) : null;
    this.scene.updateCamera(dt, { chase: pos && data ? { position: pos, heading: data.heading } : null, center: this._swarmCentre() });
    // HUD at display rate, only with a selection.
    const hudOn = $('opt-hud').checked && !!this.hud.target;
    if ($('hud').hidden === hudOn) $('hud').hidden = !hudOn;
    if (hudOn) this.hud.draw(dt);
    // Minimap ~15 Hz, charts ~4 Hz (and only while visible).
    this._miniClock += dt;
    if (this._miniClock >= 1 / 15) { this.minimap.draw(this.snapshot, this.selection, this._miniClock); this._miniClock = 0; }
    this._chartClock += dt;
    if (this.dockPane === 'charts' && this._chartClock >= 0.25 && !this._layout.dock) { this._drawCharts(); this._chartClock = 0; }
  }

  _swarmCentre() {
    const pts = [...this.layer.visuals.values()].filter((v) => v.data?.airborne).map((v) => v.anchor.position);
    if (!pts.length) return null;
    const c = pts[0].clone();
    for (let i = 1; i < pts.length; i += 1) c.add(pts[i]);
    return c.multiplyScalar(1 / pts.length);
  }

  // ------------------------------------------------------------------ Stage 3: awareness panels
  _renderHealth(s) {
    const counts = Object.fromEntries(HEALTH_BUCKETS.map(([k]) => [k, 0]));
    for (const d of s.drones) {
      const state = classify(d);
      const bucket = state === 'FAILED' ? 'FAILED' : state === 'COMM_LOST' ? 'COMM_LOST'
        : state === 'LOW_BATTERY' || d.battery_state === 'EMERGENCY' || d.battery_state === 'DEPLETED' ? 'LOW_BATTERY'
          : state === 'EMERGENCY' || state === 'COLLISION' || state === 'WARNING' ? 'WARNING' : 'ONLINE';
      counts[bucket] += 1;
    }
    const html = HEALTH_BUCKETS.map(([k, label, color]) => `<div class="hs${counts[k] ? '' : ' zero'}" style="--c:${color}" title="${label}: ${counts[k]}"><b>${counts[k]}</b><span>${label}</span></div>`).join('');
    if (html !== this._healthHtml) { $('health-states').innerHTML = html; this._healthHtml = html; }
    const sum = s.summary;
    $('hl-avg').textContent = sum.average_battery == null ? '—' : `${fmt(sum.average_battery, 0)}%`;
    $('hl-min').textContent = sum.min_battery == null ? '—' : `${fmt(sum.min_battery, 0)}%`;
    const left = sum.flight_time_left_s;
    $('hl-left').textContent = left == null ? 'on ground' : `${Math.floor(left / 60)} min ${String(Math.floor(left % 60)).padStart(2, '0')} s`;
  }

  _groups() { return this.snapshot?.missions?.groups ?? {}; }

  _renderGroups(s) {
    const groups = s.missions?.groups ?? {};
    const sel = [...this.selection].sort((a, b) => a - b).join(',');
    const key = JSON.stringify([groups, sel]);
    if (key === this._groupsKey) return;
    this._groupsKey = key;
    $('groups').innerHTML = Object.entries(groups).map(([name, ids], k) => {
      const active = [...ids].sort((a, b) => a - b).join(',') === sel;
      return `<button type="button" class="group-chip${active ? ' active' : ''}" data-group="${esc(name)}" title="${ids.map((i) => `D${String(i).padStart(2, '0')}`).join(', ')}">`
        + `${k < 9 ? `<kbd>${k + 1}</kbd>` : ''}${esc(name)}<span class="n">${ids.length}</span><span class="x" data-del="${esc(name)}" role="button" aria-label="Delete group ${esc(name)}">×</span></button>`;
    }).join('');
  }

  selectGroup(index) {
    const entry = Object.entries(this._groups())[index];
    if (!entry) return;
    const alive = new Set(this.snapshot?.drones.map((d) => d.drone_id));
    this.setSelection(entry[1].filter((i) => alive.has(i)));
    this.toast(`Group ${entry[0]} selected`, 'ok');
  }

  async saveGroup() {
    const ids = [...this.selection];
    if (!ids.length) { this.toast('Select drones first, then save them as a group', 'warn'); return; }
    const used = new Set(Object.keys(this._groups()));
    const name = NATO.find((n) => !used.has(n)) ?? `Group ${used.size + 1}`;
    await this.command('define_group', { name }, ids);
  }

  setCamera(mode) {
    if (mode === 'chase' && !this.selection.size) { this.toast('Select a drone to chase', 'warn'); mode = 'free'; }
    this.cameraMode = mode;
    this.scene.setCameraMode(mode);
    for (const b of document.querySelectorAll('.cam-btn')) b.setAttribute('aria-checked', String(b.dataset.cam === mode));
  }

  setDockPane(pane) {
    this.dockPane = pane;
    for (const t of document.querySelectorAll('.dock-tab')) t.setAttribute('aria-selected', String(t.dataset.pane === pane));
    for (const p of document.querySelectorAll('.dock-pane')) p.hidden = p.dataset.pane !== pane;
    for (const el of document.querySelectorAll('[data-pane-tool]')) el.hidden = el.dataset.paneTool !== pane;
    if (pane === 'charts') { this._chartClock = 1; }
    if (pane === 'audit') { this._auditKey = null; this._refreshAudit(); }
    if (this._layout.dock) { this._layout.dock = false; this._applyLayout(); this._saveLayout(); }
  }

  setMeasure(on) {
    this.measureMode = on;
    this.measurePts = [];
    $('btn-measure').setAttribute('aria-pressed', String(on));
    $('measure-readout').hidden = !on;
    $('measure-readout').textContent = 'Measure: click the first point on the ground';
    this.overlays?.setMeasure(null, null);
  }

  _measureClick(p) {
    if (this.measurePts.length >= 2) this.measurePts = [];
    this.measurePts.push(p);
    const [a, b] = this.measurePts;
    this.overlays?.setMeasure(a, b ?? null);
    if (!b) { $('measure-readout').textContent = 'Measure: click the second point'; return; }
    const horiz = Math.hypot(b.x - a.x, b.y - a.y);
    $('measure-readout').textContent = `${fmt(horiz, 1)} m  ·  bearing ${String(Math.round(bearing(a.x, a.y, b.x, b.y)) % 360).padStart(3, '0')}°  ·  Δ E ${fmt(b.x - a.x, 1)} / N ${fmt(b.y - a.y, 1)} m`;
  }

  _drawCharts() {
    const h = this.history;
    const ids = [...this.selection].slice(0, SERIES_COLORS.length);
    const legend = [];
    const series = (key) => {
      if (ids.length) {
        return ids.map((id, k) => {
          const ring = h.drones.get(id)?.[key];
          return ring ? { label: `D${String(id).padStart(2, '0')}`, color: SERIES_COLORS[k], ring } : null;
        }).filter(Boolean);
      }
      return [{ label: 'swarm avg', color: SERIES_COLORS[0], ring: h.swarm[key] }];
    };
    if (ids.length) ids.forEach((id, k) => legend.push(`<span><i style="background:${SERIES_COLORS[k]}"></i>D${String(id).padStart(2, '0')}</span>`));
    else legend.push(`<span><i style="background:${SERIES_COLORS[0]}"></i>swarm average (select drones to compare)</span>`);
    const html = legend.join('');
    if (html !== this._legendHtml) { $('chart-legend').innerHTML = html; this._legendHtml = html; }
    this.charts.alt.draw(series('alt'), h.now);
    this.charts.speed.draw(series('speed'), h.now);
    const th = this.snapshot?.world.battery_thresholds;
    this.charts.battery.draw(series('battery'), h.now, th ? [{ value: th.return_home, color: '#D35400', label: 'RTL' }] : []);
    const sw = this.snapshot?.world.swarm ?? {};
    const floor = this.snapshot?.swarm_control?.avoidance?.min_separation;
    const sepSeries = [{ label: 'swarm min', color: '#3A3A3A', ring: h.swarm.sep, dashed: ids.length > 0 },
      ...ids.map((id, k) => ({ label: `D${id}`, color: SERIES_COLORS[k], ring: h.drones.get(id)?.nearest })).filter((s) => s.ring)];
    this.charts.sep.draw(sepSeries, h.now, [
      ...(sw.separation_distance ? [{ value: sw.separation_distance, color: '#B7860B', label: 'separation' }] : []),
      ...(floor ? [{ value: floor, color: '#C62828', label: 'hard floor' }] : []),
    ]);
  }

  _wireAwareness() {
    for (const t of document.querySelectorAll('.dock-tab')) t.onclick = () => this.setDockPane(t.dataset.pane);
    for (const b of document.querySelectorAll('.cam-btn')) b.onclick = () => this.setCamera(b.dataset.cam);
    $('btn-save-group').onclick = () => this.saveGroup();
    $('groups').addEventListener('click', (e) => {
      const del = e.target.closest('[data-del]');
      if (del) { e.stopPropagation(); this.command('delete_group', { name: del.dataset.del }, null); return; }
      const chip = e.target.closest('.group-chip');
      if (chip) this.selectGroup([...$('groups').children].indexOf(chip));
    });
    $('btn-measure').onclick = () => this.setMeasure(!this.measureMode);
    $('opt-minimap').onchange = (e) => { $('minimap').hidden = !e.target.checked; };
    $('opt-estimate').onchange = (e) => this.layer.setOption('estimate', e.target.checked);
    $('opt-obstacles').onchange = (e) => {
      if (!this.world3d) return;
      this.world3d.obstacleGroup.visible = e.target.checked;
      this.world3d.terrainGroup.visible = e.target.checked;
    };
    this._wireInstructor();
  }

  // ------------------------------------------------------------------ Stage 4: instructor mode
  _wireInstructor() {
    const showFields = () => {
      const t = $('fi-type').value;
      for (const l of document.querySelectorAll('[data-fi]')) l.hidden = !l.dataset.fi.split(' ').includes(t);
    };
    $('fi-type').onchange = showFields;
    showFields();
    $('btn-inject').onclick = async () => {
      const ids = [...this.selection];
      if (!ids.length) { this.toast('Select the drones to fail first', 'warn'); return; }
      const type = $('fi-type').value;
      const params = { type };
      const num = (id) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : null; };
      if (type === 'motor') params.severity = $('fi-severity').value;
      if (['gps_loss', 'comm_loss', 'wind_gust'].includes(type) && num('fi-duration') != null) params.duration = num('fi-duration');
      if (type === 'battery_sag' && num('fi-drop') != null) params.drop = num('fi-drop');
      if (type === 'wind_gust') {
        if (num('fi-speed') != null) params.speed = num('fi-speed');
        if (num('fi-direction') != null) params.direction = num('fi-direction');
      }
      if (type === 'motor' && params.severity === 'total') {
        const names = ids.map((i) => `D${String(i).padStart(2, '0')}`).join(', ');
        const ok = await this.confirmDialog({ title: 'Total motor failure', danger: true, okLabel: 'Crash drones',
          html: `<p>${names} will lose all motors and fall. This cannot be undone in this run.</p>` });
        if (!ok) return;
      }
      this.command('inject_failure', params, ids);
    };
    $('btn-clear-failures').onclick = () => this.command('clear_failure', {}, null);
    $('active-failures').addEventListener('click', (e) => {
      const b = e.target.closest('[data-clear]');
      if (!b) return;
      this.command('clear_failure', { type: b.dataset.clear }, [Number(b.dataset.drone)]);
    });
  }

  _renderFailures(s) {
    const active = s.failures?.active ?? [];
    const label = { motor: 'Motor', gps_loss: 'GPS loss', comm_loss: 'Comm loss', battery_sag: 'Battery sag', wind_gust: 'Wind gust' };
    const html = active.map((a) => {
      const left = a.until == null ? 'until cleared' : `${Math.max(0, Math.round(a.until - s.sim_time))} s left`;
      return `<div><b>D${String(a.drone_id).padStart(2, '0')}</b><span>${label[a.type] ?? a.type} · ${left}</span>`
        + `<button type="button" data-clear="${a.type}" data-drone="${a.drone_id}" aria-label="Clear">×</button></div>`;
    }).join('');
    if (html !== this._failHtml) { $('active-failures').innerHTML = html; this._failHtml = html; }
  }

  // ------------------------------------------------------------------ rendering
  _renderStatus(s) {
    const pill = $('state-pill');
    pill.textContent = s.state;
    pill.className = `pill state ${s.state}`;
    $('m-time').textContent = clock(s.sim_time);
    $('m-rate').textContent = s.state === 'RUNNING' ? `${fmt(s.stats.sim_rate_hz, 0)} Hz` : '—';
    $('m-step').textContent = `${fmt(s.stats.step_ms, 2)} ms`;
    $('m-airborne').textContent = `${s.summary.airborne}/${s.summary.total}`;
    $('m-minsep').textContent = s.summary.min_separation == null ? '—' : `${fmt(s.summary.min_separation)} m`;
    this._setRunButtons(s.state === 'RUNNING');
    const rate = `${s.state === 'RUNNING' ? fmt(s.stats.sim_rate_hz, 0) : '—'} / ${fmt(s.stats.target_rate_hz, 0)} Hz`;
    $('sim-rate-val').textContent = rate;
    $('dock-rate').textContent = rate;
    $('dock-time').textContent = clock(s.sim_time);

    const w = s.wind;
    $('wind-arrow').style.transform = `rotate(${(w.direction + 180) % 360}deg)`;   // arrow points where the wind blows
    $('wind-text').textContent = w.enabled ? `${fmt(w.speed)} m/s from ${String(Math.round(w.direction)).padStart(3, '0')}°` : 'calm';
  }

  _renderFleet(s) {
    const list = $('fleet-list');
    const thresholds = s.world.battery_thresholds;
    $('fleet-count').textContent = s.drones.length;
    const seen = new Set();
    for (const d of s.drones) {
      seen.add(d.drone_id);
      let row = this.rows.get(d.drone_id);
      if (!row) {
        row = document.createElement('div');
        row.className = 'fleet-row';
        row.tabIndex = 0;
        row.innerHTML = '<span class="id"></span><span class="status"><i class="dot"></i><b></b></span><span class="mode"></span><span class="num alt"></span><span class="batt"><span class="batt-bar"><i></i></span><span class="pct"></span></span>';
        row.addEventListener('click', (ev) => this._clickDrone(d.drone_id, ev.ctrlKey || ev.metaKey));
        row.addEventListener('keydown', (ev) => {
          if (ev.key === 'Enter') { ev.preventDefault(); this._clickDrone(d.drone_id, ev.ctrlKey || ev.metaKey); }
        });
        this.rows.set(d.drone_id, row);
        list.appendChild(row);
      }
      const state = classify(d);
      const vis = VISUAL_STATES[state];
      const idHtml = d.source && d.source !== 'sim' ? `${esc(d.name)}<span class="hw-tag" title="Real / SITL vehicle via ${esc(d.source)}">${d.source === 'mavsdk' ? 'SDK' : 'MAV'}</span>` : esc(d.name);
      const idEl = row.querySelector('.id');
      if (idEl.innerHTML !== idHtml) idEl.innerHTML = idHtml;
      const dot = row.querySelector('.dot');
      dot.style.background = vis.color;
      dot.classList.toggle('blink', !!vis.blink);
      const label = row.querySelector('.status b');
      // The state colour is carried by the dot; the label stays dark for WCAG contrast on light panels.
      label.textContent = vis.short;
      label.title = vis.label;
      row.querySelector('.mode').textContent = d.mode;
      row.querySelector('.alt').textContent = `${fmt(d.altitude_agl)}`;
      const bar = row.querySelector('.batt-bar i');
      bar.style.width = `${Math.max(0, Math.min(100, d.battery))}%`;
      bar.style.background = batteryColor(d.battery, thresholds);
      row.querySelector('.pct').textContent = `${fmt(d.battery, 0)}%`;
      row.classList.toggle('selected', this.selection.has(d.drone_id));
    }
    for (const [id, row] of this.rows) {
      if (!seen.has(id)) { row.remove(); this.rows.delete(id); }
    }
  }

  _renderInspector(s) {
    const ids = [...this.selection];
    const body = $('inspector');
    const actions = $('inspector-actions');
    if (ids.length === 0) {
      if (this._inspectorId !== 'empty') {
        this._inspectorId = 'empty';
        this._inspectorFields = null;
        $('inspector-name').textContent = '—';
        body.innerHTML = '<div class="empty">Select a drone to inspect its telemetry.</div>';
        actions.hidden = true;
      }
      return;
    }
    const d = s.drones.find((x) => x.drone_id === ids[0]);
    if (!d) return;
    actions.hidden = false;
    $('inspector-name').textContent = ids.length > 1 ? `${d.name} +${ids.length - 1}` : d.name;

    // Build the panel once per inspected drone; afterwards only text nodes are updated (no re-layout churn).
    if (this._inspectorId !== d.drone_id || !this._inspectorFields || !body.contains(this._inspectorFields.chip)) {
      this._inspectorId = d.drone_id;
      body.innerHTML = `<div class="insp-head"><span class="insp-id" data-k="name"></span><span class="state-chip"><i></i><b></b></span></div>`
        + INSPECTOR_SECTIONS.map(([title, fields]) => `<div class="insp-section"><h4>${title}</h4><div class="insp-grid">${
          fields.map(([k, label, wide]) => `<div${wide ? ' class="wide"' : ''}><label>${label}</label><span data-k="${k}"></span>${
            k === 'battery' ? '<span class="mini-bar"><i></i></span>' : ''}</div>`).join('')}</div></div>`).join('');
      const fields = {};
      for (const el of body.querySelectorAll('[data-k]')) fields[el.dataset.k] = el;
      this._inspectorFields = {
        fields,
        chip: body.querySelector('.state-chip'),
        chipDot: body.querySelector('.state-chip i'),
        chipText: body.querySelector('.state-chip b'),
        batt: body.querySelector('.mini-bar i'),
      };
    }
    const f = this._inspectorFields;
    const vis = VISUAL_STATES[classify(d)];
    f.chipDot.style.background = vis.color;
    f.chipDot.classList.toggle('blink', !!vis.blink);
    f.chipText.textContent = vis.label;
    const values = this._inspectorValues(d);
    for (const [k, el] of Object.entries(f.fields)) {
      const v = values[k] ?? '—';
      if (el.textContent !== v) { el.textContent = v; el.title = v; }
    }
    f.batt.style.width = `${Math.max(0, Math.min(100, d.battery))}%`;
    f.batt.style.background = batteryColor(d.battery, s.world.battery_thresholds);
  }

  _inspectorValues(d) {
    const xyz = (v, dgt = 1) => (v ? `${fmt(v.x, dgt)}, ${fmt(v.y, dgt)}, ${fmt(v.z, dgt)}` : '—');
    return {
      name: d.name,
      mode: d.mode,
      task: d.task,
      armed: d.armed ? 'ARMED' : 'DISARMED',
      health: d.health,
      link: d.communication,
      collision: d.collision_state,
      local: xyz(d.position),
      lat: fmt(d.gps.lat, 7),
      lon: fmt(d.gps.lon, 7),
      agl: `${fmt(d.altitude_agl)} m`,
      amsl: `${fmt(d.gps.alt)} m`,
      target: xyz(d.target),
      speed: `${fmt(d.speed)} m/s`,
      heading: `${fmt(d.heading, 0)}°`,
      vel: xyz(d.velocity),
      attitude: `${fmt(d.roll)}° / ${fmt(d.pitch)}°`,
      yawrate: `${fmt(d.yaw_rate)}°/s`,
      accel: xyz(d.acceleration),
      battery: `${fmt(d.battery)}% · ${d.battery_state}`,
      voltage: `${fmt(d.battery_voltage, 2)} V`,
      power: `${fmt(d.power_w, 0)} W`,
      nearest: d.nearest_distance == null ? '—' : `${fmt(d.nearest_distance)} m`,
      ftime: clock(d.flight_time),
      dist: `${fmt(d.distance_travelled, 0)} m`,
      neighbors: d.neighbors.length ? d.neighbors.map((n) => `D${String(n).padStart(2, '0')}`).join(' ') : '—',
      home: xyz(d.home, 0),
      linkq: d.link_quality == null ? '—' : `${fmt(d.link_quality, 0)} % · ${d.communication}`,
      age: d.telemetry_age_s == null ? 'live' : `${fmt(d.telemetry_age_s, 2)} s`,
      gps: `${d.gps_fix} · ${d.satellites} sat · HDOP ${fmt(d.hdop, 1)}`,
      esterr: d.pos_error == null ? 'sensor model off' : `${fmt(d.pos_error, 2)} m / ±${fmt(d.pos_sigma, 2)} m`,
      failures: d.failures?.length ? d.failures.join(', ') : 'none',
    };
  }

  _renderMission(s) {
    const sc = s.swarm_control;
    const mode = sc?.mode ?? 'BASIC';
    const f = sc?.formation;
    $('ms-mode').textContent = mode === 'BASIC' ? 'BASIC SWARM' : mode;
    $('ms-formation').textContent = f
      ? `${f.shape.toUpperCase()} · ${fmt(f.spacing, 0)} m${f.layers > 1 ? ` · ${f.layers} layers` : ''} · hdg ${fmt(f.heading, 0)}°${f.heading_locked ? ' (fixed)' : ''}`
        + (f.transition_progress < 1 ? ` · transition ${Math.round(f.transition_progress * 100)}%` : '')
      : (mode === 'FLOCKING' ? 'FLOCK' : '—');
    const badge = f ? `${f.shape.toUpperCase()} · err ${fmt(f.max_slot_error)} m` : (mode === 'FLOCKING' ? 'FLOCKING' : '');
    if ($('formation-badge').textContent !== badge) $('formation-badge').textContent = badge;
    const hint = mode === 'BASIC' ? 'Click: select · Ctrl+click: multi-select · Shift+click ground: fly to'
      : `Click: select · Shift+click ground: move the ${mode === 'FLOCKING' ? 'flock' : 'formation'}`;
    if ($('vp-hint').textContent !== hint) $('vp-hint').textContent = hint;
    const av = sc?.avoidance;
    const avoidText = av ? (av.enabled ? `${av.method === 'orca' ? 'ORCA' : 'PF'} · ${av.active_pairs} pair${av.active_pairs === 1 ? '' : 's'}` : 'OFF') : '—';
    if ($('ms-avoidance').textContent !== avoidText) $('ms-avoidance').textContent = avoidText;
    const avoid = $('opt-avoidance');
    if (av && document.activeElement !== avoid && avoid.checked !== av.enabled) avoid.checked = av.enabled;
    const method = $('avoidance-method');
    if (av && document.activeElement !== method && method.value !== av.method) method.value = av.method;
    if (sc?.flocking_weights) {
      for (const [key, value] of Object.entries(sc.flocking_weights)) {
        const input = $(`w-${key}`);
        if (!input || document.activeElement === input || Number(input.value) === value) continue;
        input.value = value;
        $(`w-${key}-val`).textContent = Number(value).toFixed(1);
      }
    }
    const custom = $('custom-offsets');
    if (sc?.custom_offsets?.length && !custom.value && document.activeElement !== custom && !this._customSeeded) {
      this._customSeeded = true;      // seed once from the configured custom formation; the operator edits from there
      custom.value = sc.custom_offsets.map((p) => p.map((v) => Number(v).toFixed(1)).join(', ')).join('\n');
    }
    $('ms-leader').textContent = f?.reference === 'leader' && f.leader_id != null
      ? `D${String(f.leader_id).padStart(2, '0')}${f.promotions ? ` (${f.promotions} promoted)` : ''}` : '—';
    this.layer.setFormation?.(f);
    const sum = s.summary;
    $('ms-floor').textContent = av ? `${fmt(av.min_separation)} m / ${sum.min_separation_breaches ?? 0}✕` : '—';
    $('ms-lowest').textContent = sum.lowest_separation == null ? '—' : `${fmt(sum.lowest_separation, 2)} m`;
    let status = 'IDLE';
    if (sum.failed) status = 'DEGRADED';
    else if (sum.airborne) status = 'ACTIVE';
    if (s.state === 'PAUSED') status = 'PAUSED';
    $('ms-status').textContent = status;
    $('ms-battery').textContent = sum.average_battery == null ? '—' : `${fmt(sum.average_battery)}%`;
    $('ms-violations').textContent = sum.separation_violations;
    $('ms-collisions').textContent = sum.collisions;
  }

  _appendEvents(events) {
    const fresh = events.filter((e) => e.seq > this.lastEventSeq);
    if (!fresh.length) return;
    this.lastEventSeq = fresh[fresh.length - 1].seq;
    const log = $('event-log');
    const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
    for (const e of fresh) {
      this.events.push(e);
      log.appendChild(this._eventRow(e));
    }
    while (this.events.length > MAX_EVENTS) {
      this.events.shift();
      log.firstChild?.remove();
    }
    if (atBottom) log.scrollTop = log.scrollHeight;
  }

  _eventRow(e) {
    const row = document.createElement('div');
    row.className = `event ${e.severity}`;
    row.dataset.category = e.category;
    row.hidden = !this._eventVisible(e.category, e.severity);
    row.innerHTML = `<span class="t">${clock(e.time)}</span><span class="c">${esc(e.category)}</span><span class="d">${e.drone_id != null ? `D${String(e.drone_id).padStart(2, '0')}` : ''}</span><span class="m" title="${esc(e.message)}">${esc(e.message)}</span>`;
    row.dataset.severity = e.severity;
    return row;
  }

  _eventVisible(category, severity) {
    const f = $('event-filter').value;
    if (f === 'all') return true;
    if (f === 'warn') return severity !== 'INFO';
    return category === f;
  }

  _buildLegend() {
    $('legend').innerHTML = Object.values(VISUAL_STATES)
      .map((v) => `<div><span class="swatch" style="background:${v.color}"></span>${v.label}</div>`).join('');
  }

  toast(message, kind = 'ok') {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.textContent = message;
    $('toasts').appendChild(el);
    setTimeout(() => el.remove(), kind === 'ok' ? 2500 : 5000);
    while ($('toasts').children.length > 5) $('toasts').firstChild.remove();
  }

  // ------------------------------------------------------------------ selection
  setSelection(ids) {
    this.selection = new Set(ids);
    this.layer.setSelection(this.selection);
    const n = this.selection.size;
    const names = [...this.selection].sort((a, b) => a - b).map((i) => `D${String(i).padStart(2, '0')}`);
    $('target-label').textContent = n === 0 ? `ALL DRONES${this.snapshot ? ` (${this.snapshot.drones.length})` : ''}`
      : n <= 6 ? names.join(', ') : `${n} DRONES SELECTED`;
    for (const [id, row] of this.rows) row.classList.toggle('selected', this.selection.has(id));
    if (this.snapshot) this._renderInspector(this.snapshot);
  }

  _clickDrone(id, additive) {
    if (additive) {
      const next = new Set(this.selection);
      if (next.has(id)) next.delete(id); else next.add(id);
      this.setSelection(next);
    } else {
      this.setSelection(this.selection.size === 1 && this.selection.has(id) ? [] : [id]);
    }
  }

  /** Drone ids that commands address: the selection, or null for the whole swarm. */
  targets() {
    return this.selection.size ? [...this.selection] : null;
  }

  // ------------------------------------------------------------------ commands
  async command(type, params = {}, ids = this.targets()) {
    if (!this.canControl) { this.toast('Observer login: read-only — commands are disabled', 'warn'); return null; }
    const danger = DANGEROUS_ALL[type];
    if (danger && ids === null) {
      const n = this.snapshot?.drones.length ?? 0;
      const ok = await this.confirmDialog({ title: danger.title, html: danger.html(n), okLabel: danger.ok, danger: true });
      if (!ok) { this.toast(`${danger.ok} cancelled`, 'warn'); return null; }
    }
    try {
      const res = await this.link.command(type, ids, params);
      this.toast(`${type.replace(/_/g, ' ')}: ${res.message}`, res.success ? (Object.keys(res.details ?? {}).length ? 'warn' : 'ok') : 'error');
      return res;
    } catch (err) {
      this.toast(`${type}: ${err.message}`, 'error');
      return null;
    }
  }

  async simulation(action) {
    if (!this.canControl) { this.toast('Observer login: read-only', 'warn'); return; }
    try {
      const res = await this.link.simulation(action);
      this.toast(res.message, res.success ? 'ok' : 'error');
    } catch (err) {
      this.toast(`${action}: ${err.message}`, 'error');
    }
  }

  _num(id, fallback) {
    const v = parseFloat($(id).value);
    return Number.isFinite(v) ? v : fallback;
  }

  /** set_formation with every formation parameter from the Swarm section. */
  _flyFormation(shape) {
    const params = {
      shape,
      spacing: this._num('formation-spacing', 15),
      reference: $('formation-reference').value,
      layers: Math.round(this._num('formation-layers', 1)),
      layer_spacing: this._num('formation-layer-gap', 6),
    };
    const heading = parseFloat($('formation-heading').value);
    if (Number.isFinite(heading)) params.heading = heading;
    if (params.reference === 'leader' && this.selection.size) params.leader_id = Math.min(...this.selection);
    if (shape === 'custom') {
      const text = $('custom-offsets').value.trim();
      if (text) {
        const offsets = this._parseOffsets(text);
        if (!offsets) return;
        params.offsets = offsets;
      }
    }
    this.command('set_formation', params);
  }

  /** "fwd, left, up" per line -> [[f, l, u], ...]; toasts and returns null on a syntax error. */
  _parseOffsets(text) {
    const rows = [];
    const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith('#'));
    for (const [n, line] of lines.entries()) {
      const nums = line.replace(/[[\]]/g, '').split(/[\s,;]+/).filter(Boolean).map(Number);
      if (nums.length === 2) nums.push(0);
      if (nums.length !== 3 || nums.some((v) => !Number.isFinite(v))) {
        this.toast(`Custom offsets, line ${n + 1}: expected "forward, left, up"`, 'error');
        return null;
      }
      rows.push(nums);
    }
    if (!rows.length) { this.toast('Custom offsets: no slots given', 'error'); return null; }
    return rows;
  }

  /** Current selection -> custom offsets in the formation frame (forward = North, left = West). */
  _customFromSelection() {
    const drones = (this.snapshot?.drones ?? []).filter((d) => this.selection.has(d.drone_id));
    if (drones.length < 2) { this.toast('Select at least two drones to capture a shape', 'warn'); return; }
    const c = drones.reduce((a, d) => ({ x: a.x + d.position.x / drones.length, y: a.y + d.position.y / drones.length,
      z: a.z + d.position.z / drones.length }), { x: 0, y: 0, z: 0 });
    $('custom-offsets').value = drones.map((d) => [d.position.y - c.y, -(d.position.x - c.x), d.position.z - c.z]
      .map((v) => v.toFixed(1)).join(', ')).join('\n');
    $('custom-box').open = true;
    $('formation-shape').value = 'custom';
  }

  _setRunButtons(running) {
    if (running === this._runShown) return;
    this._runShown = running;
    for (const id of ['btn-run', 'btn-run-dock']) {
      const btn = $(id);
      btn.querySelector('use').setAttribute('href', running ? '#i-pause' : '#i-play');
      btn.querySelector('.lbl').textContent = running ? 'Pause' : 'Start';
      btn.setAttribute('aria-pressed', String(running));
    }
  }

  _toggleRun() {
    this.simulation(this.snapshot?.state === 'RUNNING' ? 'pause' : 'start');
  }

  _confirmReset() {
    if (confirm('Reset the simulation? All drones return to their pads and a new run is recorded.')) this.simulation('reset');
  }

  _wireControls() {
    $('tab-fly').onclick = () => this.setView('fly');
    $('tab-plan').onclick = () => this.setView('plan');
    $('tab-replay').onclick = () => this.setView('replay');
    $('btn-report').onclick = () => {
      const id = this.lastLiveRunId ?? this.snapshot?.run_id;
      if (id && !String(id).startsWith('replay:')) import('./replay.js').then((m) => m.openReport(id, 'html'));
    };
    $('btn-run').onclick = () => this._toggleRun();
    $('btn-run-dock').onclick = () => this._toggleRun();
    $('btn-reset').onclick = () => this._confirmReset();
    $('btn-reset-dock').onclick = () => this._confirmReset();
    $('btn-hover').onclick = () => this.command('hover');
    $('btn-formation').onclick = () => this._flyFormation($('formation-shape').value);
    $('btn-custom-apply').onclick = () => { $('formation-shape').value = 'custom'; this._flyFormation('custom'); };
    $('btn-custom-from-sel').onclick = () => this._customFromSelection();
    $('btn-flock').onclick = () => this.command('start_flocking');
    $('btn-release').onclick = () => this.command('release_swarm', {}, null);
    $('opt-avoidance').onchange = (e) => this.command('set_avoidance', { enabled: e.target.checked }, null);
    $('avoidance-method').onchange = (e) => this.command('set_avoidance', { method: e.target.value }, null);
    for (const key of ['separation', 'alignment', 'cohesion', 'goal']) {
      const input = $(`w-${key}`);
      input.oninput = () => { $(`w-${key}-val`).textContent = Number(input.value).toFixed(1); };
      input.onchange = () => this.command('set_flocking_weights', { [key]: Number(input.value) }, null);
    }
    $('btn-takeoff').onclick = () => this.command('takeoff', { altitude: this._num('in-takeoff-alt', 20) });
    $('btn-land').onclick = () => this.command('land');
    $('btn-rth').onclick = () => this.command('return_to_home');
    $('btn-estop').onclick = () => this.command('emergency_stop');
    $('btn-add').onclick = () => this.command('add_drone', { count: 1 }, null);
    $('btn-remove').onclick = () => {
      let ids = [...this.selection];
      if (!ids.length && this.snapshot?.drones.length) ids = [this.snapshot.drones[this.snapshot.drones.length - 1].drone_id];
      if (!ids.length) return;
      const names = ids.map((i) => `D${String(i).padStart(2, '0')}`).join(', ');
      if (confirm(`Remove ${names}?`)) this.command('remove_drone', {}, ids);
    };
    $('btn-select-all').onclick = () => this.setSelection(this.snapshot?.drones.map((d) => d.drone_id) ?? []);
    $('btn-select-none').onclick = () => this.setSelection([]);
    $('btn-view-reset').onclick = () => { this.setCamera('free'); this.scene.resetView(); };
    for (const name of ['trails', 'targets', 'links', 'safety', 'labels']) {
      $(`opt-${name}`).onchange = (e) => this.layer.setOption(name, e.target.checked);
    }
    const applySize = (value) => {
      this.layer.setOption('size', value);
      $('opt-size').value = value;
      $('opt-size-val').textContent = `${value}×`;
    };
    let savedSize = 1;
    try { savedSize = parseFloat(localStorage.getItem('swarm.droneSize')) || 1; } catch { /* storage unavailable */ }
    applySize(savedSize);
    $('opt-size').oninput = (e) => {
      const value = parseFloat(e.target.value);
      applySize(value);
      try { localStorage.setItem('swarm.droneSize', String(value)); } catch { /* ignore */ }
    };
    $('event-filter').onchange = () => {
      for (const row of $('event-log').children) row.hidden = !this._eventVisible(row.dataset.category, row.dataset.severity);
    };
    $('inspector-actions').addEventListener('click', (e) => {
      const cmd = e.target.closest('button')?.dataset.cmd;
      if (!cmd) return;
      const params = cmd === 'takeoff' ? { altitude: this._num('in-takeoff-alt', 20) } : {};
      this.command(cmd, params);
    });
  }

  _wireViewport() {
    const canvas = this.scene.renderer.domElement;
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    canvas.addEventListener('pointerup', (e) => {
      if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;   // it was a drag
      down = null;
      const ndc = this.scene.ndc(e);
      if (this.measureMode) {
        const p = this.scene.groundPoint(ndc);
        if (p) this._measureClick(p);
        return;
      }
      if (e.shiftKey) {
        const p = this.scene.groundPoint(ndc);
        if (!p) return;
        const alt = this._num('in-goto-alt', 30);
        // The Go-to altitude is height above the ground under the target (terrain aware).
        const params = { position: [p.x, p.y, alt], speed: this._num('in-goto-speed', 10), agl: true };
        // With a formation or flock active, Shift+click moves the whole group instead of individual drones.
        const swarmActive = (this.snapshot?.swarm_control?.mode ?? 'BASIC') !== 'BASIC';
        if (swarmActive) this.command('swarm_goto', params, null);
        else this.command('goto', params);
        return;
      }
      const hit = this.scene.pick(ndc, this.layer.pickables());
      if (hit) this._clickDrone(hit.object.userData.droneId, e.ctrlKey || e.metaKey);
      else if (!e.ctrlKey && !e.metaKey) this.setSelection([]);
    });
  }

  _wireKeyboard() {
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement
          || e.target instanceof HTMLTextAreaElement) return;
      const running = this.snapshot?.state === 'RUNNING';
      if (e.key === 'F1' || e.key === 'F2' || e.key === 'F3') {
        e.preventDefault();
        this.setView({ F1: 'fly', F2: 'plan', F3: 'replay' }[e.key]);
        return;
      }
      if (e.target instanceof HTMLButtonElement && e.key === ' ') return;   // Space activates the focused button
      if (this.view === 'replay' && this.replay) {
        // Replay transport keys; flight-command keys stay live except E-STOP (X).
        if (e.key === ' ') { e.preventDefault(); this.replay.toggle(); return; }
        if (e.key === 'ArrowLeft') { this.replay.step(-5); return; }
        if (e.key === 'ArrowRight') { this.replay.step(5); return; }
        if ('tTlLhH'.includes(e.key) && e.key.length === 1) return;
      }
      switch (e.key) {
        case ' ': e.preventDefault(); this.simulation(running ? 'pause' : 'start'); break;
        case 't': case 'T': $('btn-takeoff').click(); break;
        case 'l': case 'L': this.command('land'); break;
        case 'h': case 'H': this.command('return_to_home'); break;
        case 'x': case 'X': this.command('emergency_stop'); break;
        case 'f': case 'F': this.setCamera(this.cameraMode === 'chase' ? 'free' : 'chase'); break;
        case 'r': case 'R': if (!this.planVisible) this.setMeasure(!this.measureMode); break;
        case 'Escape': if (this.measureMode) this.setMeasure(false); else this.setSelection([]); break;
        case 'm': case 'M': this.toggleMaximise(); break;
        default:
          if (e.key >= '1' && e.key <= '9' && !e.ctrlKey && !e.altKey && !e.metaKey) this.selectGroup(Number(e.key) - 1);
          break;
      }
    });
  }

  // ------------------------------------------------------------------ presentation: layout, branding, tooltips
  _loadLayout() {
    const defaults = { left: false, right: false, dock: false, dockH: null, sections: {} };
    try {
      const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY) || 'null');
      return saved && typeof saved === 'object' ? { ...defaults, ...saved, sections: { ...(saved.sections || {}) } } : defaults;
    } catch (e) {
      return defaults;   // storage unavailable or corrupt: fall back to the default layout
    }
  }

  _saveLayout() {
    try {
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(this._layout));
    } catch (e) {
      /* storage unavailable (private mode, blocked): layout simply is not remembered */
    }
  }

  _applyLayout() {
    const app = $('app');
    const L = this._layout;
    app.classList.toggle('left-collapsed', !!L.left);
    app.classList.toggle('right-collapsed', !!L.right);
    app.classList.toggle('dock-collapsed', !!L.dock);
    if (L.dockH) app.style.setProperty('--dock-user', `${L.dockH}px`);
    else app.style.removeProperty('--dock-user');
    const sync = (id, collapsed, what) => {
      const b = $(id);
      b.setAttribute('aria-expanded', String(!collapsed));
      b.dataset.tip = `${collapsed ? 'Expand' : 'Collapse'} ${what}`;
      b.setAttribute('aria-label', b.dataset.tip);
    };
    sync('toggle-left', L.left, 'command rail');
    sync('toggle-right', L.right, 'inspector rail');
    sync('toggle-dock', L.dock, 'event dock');
  }

  _wireLayout() {
    const flip = (key) => { this._layout[key] = !this._layout[key]; this._applyLayout(); this._saveLayout(); };
    $('toggle-left').onclick = () => flip('left');
    $('toggle-right').onclick = () => flip('right');
    $('toggle-dock').onclick = () => flip('dock');
    $('btn-maximise').onclick = () => this.toggleMaximise();

    // Collapsible command-rail sections, remembered per section.
    for (const section of document.querySelectorAll('.rail-left .group[data-section]')) {
      const head = section.querySelector('button.group-head');
      if (!head) continue;
      const name = section.dataset.section;
      const set = (collapsed) => {
        section.classList.toggle('collapsed', collapsed);
        head.setAttribute('aria-expanded', String(!collapsed));
      };
      set(!!this._layout.sections[name]);
      head.addEventListener('click', () => {
        const collapsed = !section.classList.contains('collapsed');
        set(collapsed);
        this._layout.sections[name] = collapsed;
        this._saveLayout();
      });
    }

    // Resizable dock (drag the top edge, or focus it and use the arrow keys).
    const resizer = $('dock-resizer');
    const clampH = (h) => Math.round(Math.max(96, Math.min(window.innerHeight * 0.6, h)));
    const setH = (h) => {
      this._layout.dockH = clampH(h);
      $('app').style.setProperty('--dock-user', `${this._layout.dockH}px`);
    };
    resizer.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      resizer.setPointerCapture(e.pointerId);
      resizer.classList.add('dragging');
      const move = (ev) => setH(window.innerHeight - ev.clientY);
      const up = () => {
        resizer.classList.remove('dragging');
        resizer.removeEventListener('pointermove', move);
        resizer.removeEventListener('pointerup', up);
        resizer.removeEventListener('pointercancel', up);
        this._saveLayout();
      };
      resizer.addEventListener('pointermove', move);
      resizer.addEventListener('pointerup', up);
      resizer.addEventListener('pointercancel', up);
    });
    resizer.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      e.preventDefault();
      e.stopPropagation();
      setH($('dock').getBoundingClientRect().height + (e.key === 'ArrowUp' ? 16 : -16));
      this._saveLayout();
    });

    // Keep the Three.js renderer matched to the viewport whenever the layout changes size
    // (rail/dock collapse, dock resize, maximise, window resize). Coalesced to one resize per frame.
    let pending = false;
    new ResizeObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; this.scene.resize(); });
    }).observe($('viewport'));

    this._applyLayout();
  }

  toggleMaximise() {
    const on = $('app').classList.toggle('maximised');
    const b = $('btn-maximise');
    b.setAttribute('aria-pressed', String(on));
    b.dataset.tip = on ? 'Restore layout — M' : 'Maximise View — M';
    b.setAttribute('aria-label', b.dataset.tip);
    this._hideTip();
  }

  _wireLogo() {
    // The logo is a local asset; if it is missing, hide every <img> that uses it instead of showing a broken image.
    for (const img of document.querySelectorAll('img[src$="gandiv-logo.png"]')) {
      const markMissing = () => { img.dataset.missing = 'true'; img.hidden = true; };
      if (img.complete && img.naturalWidth === 0) markMissing();
      else img.addEventListener('error', markMissing, { once: true });
    }
  }

  _themeViewport() {
    // Presentation only: neutral operator-console backdrop for the existing Three.js scene.
    const scene = this.scene.scene;
    if (!this._vpThemed) {
      scene.background.set(VIEWPORT_BG);
      if (scene.fog) scene.fog.color.set(VIEWPORT_BG);
      this._vpThemed = true;
    }
    const ground = this.scene.world.getObjectByName('ground');
    if (ground && !ground.userData.gcsThemed) {
      ground.material.color.set(VIEWPORT_GROUND);
      ground.userData.gcsThemed = true;
    }
  }

  _wireTooltips() {
    const tip = $('tooltip');
    for (const el of document.querySelectorAll('[data-tip]')) {
      if (!el.textContent.trim() && !el.hasAttribute('aria-label')) el.setAttribute('aria-label', el.dataset.tip);
    }
    const show = (el) => {
      tip.textContent = el.dataset.tip;
      tip.hidden = false;
      const r = el.getBoundingClientRect();
      const t = tip.getBoundingClientRect();
      const gap = 8;
      let x;
      let y;
      if (el.closest('.rail-left')) { x = r.right + gap; y = r.top + (r.height - t.height) / 2; }
      else if (el.closest('.rail-right')) { x = r.left - t.width - gap; y = r.top + (r.height - t.height) / 2; }
      else if (el.closest('.dock')) { x = r.left + (r.width - t.width) / 2; y = r.top - t.height - gap; }
      else { x = r.left + (r.width - t.width) / 2; y = r.bottom + gap; }
      x = Math.max(4, Math.min(window.innerWidth - t.width - 4, x));
      y = Math.max(4, Math.min(window.innerHeight - t.height - 4, y));
      tip.style.left = `${x}px`;
      tip.style.top = `${y}px`;
    };
    document.addEventListener('pointerover', (e) => {
      const el = e.target.closest?.('[data-tip]');
      if (el) show(el); else this._hideTip();
    });
    document.addEventListener('focusin', (e) => {
      const el = e.target.closest?.('[data-tip]');
      if (el && el.matches(':focus-visible')) show(el); else this._hideTip();
    });
    document.addEventListener('focusout', () => this._hideTip());
    document.addEventListener('pointerdown', () => this._hideTip());
  }

  _hideTip() {
    const tip = $('tooltip');
    if (!tip.hidden) tip.hidden = true;
  }
}
