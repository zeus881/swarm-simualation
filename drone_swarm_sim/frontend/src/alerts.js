// Alerts panel: prioritised list, critical banner with acknowledgement, audible beep (WebAudio, no
// audio files so it works offline) and flashing of the drones involved in unacknowledged CRITICAL alerts.
// The alert state itself lives on the server (shared by every GCS client); acknowledgement is a command.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const BEEP_KEY = 'gandiv.gcs.beep';

function clock(t) {
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, '0')}:${(t - m * 60).toFixed(1).padStart(4, '0')}`;
}

export class AlertsPanel {
  constructor(ui) {
    this.ui = ui;
    this.version = -1;
    this.alerts = [];
    this.seen = new Set();          // alert ids already announced
    this.audio = null;
    this.lastBeep = -Infinity;
    this.repeat = 8;
    let on = true;
    try { on = localStorage.getItem(BEEP_KEY) !== 'off'; } catch { /* storage unavailable */ }
    $('opt-beep').checked = on;
    $('opt-beep').onchange = (e) => {
      try { localStorage.setItem(BEEP_KEY, e.target.checked ? 'on' : 'off'); } catch { /* ignore */ }
      if (e.target.checked) this._unlockAudio();
    };
    $('btn-ack-all').onclick = () => this.ack(null);
    $('ab-ack').onclick = () => { const id = Number($('ab-ack').dataset.id); if (id) this.ack([id]); };
    $('alerts-list').addEventListener('click', (e) => {
      const ack = e.target.closest('[data-ack]');
      if (ack) { this.ack([Number(ack.dataset.ack)]); return; }
      const drone = e.target.closest('[data-drone]');
      if (drone) this.ui.setSelection(drone.dataset.drone.split(',').map(Number));
    });
    // Browsers only start audio after a user gesture: unlock on the first interaction.
    const unlock = () => { this._unlockAudio(); window.removeEventListener('pointerdown', unlock); window.removeEventListener('keydown', unlock); };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
  }

  _unlockAudio() {
    if (this.audio) { if (this.audio.state === 'suspended') this.audio.resume(); return; }
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (Ctx) { try { this.audio = new Ctx(); } catch { this.audio = null; } }
  }

  /** Two-tone alarm beep (880 Hz then 660 Hz). */
  beep() {
    if (!$('opt-beep').checked) return;
    this._unlockAudio();
    const ctx = this.audio;
    if (!ctx || ctx.state !== 'running') return;
    const t = ctx.currentTime;
    for (const [freq, start] of [[880, 0], [660, 0.18]]) {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t + start);
      gain.gain.exponentialRampToValueAtTime(0.12, t + start + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + start + 0.16);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t + start);
      osc.stop(t + start + 0.17);
    }
    this.beeps = (this.beeps ?? 0) + 1;
  }

  async ack(ids) {
    const params = ids ? { ids } : { all: true };
    await this.ui.command('ack_alert', params, null);
  }

  update(snap) {
    const a = snap.alerts;
    if (!a) return;
    this.repeat = snap.world.alerts?.critical_repeat_s ?? 8;
    if (a.version !== this.version) {
      this.version = a.version;
      this.alerts = a.active;
      this._renderList();
      const fresh = this.alerts.filter((x) => x.requires_ack && !x.acknowledged && !this.seen.has(x.id));
      for (const x of this.alerts) this.seen.add(x.id);
      if (fresh.length) { this.beep(); this.lastBeep = performance.now() / 1000; }
      this._renderBanner();
      const flash = new Set();
      for (const x of this.alerts) if (x.requires_ack && !x.acknowledged) x.drone_ids.forEach((id) => flash.add(id));
      this.ui.layer.setFlash?.(flash);
    }
    // Repeat the beep while a CRITICAL alert is unacknowledged.
    const now = performance.now() / 1000;
    if (a.unacked_critical > 0 && now - this.lastBeep >= this.repeat) { this.beep(); this.lastBeep = now; }
    const badge = $('alert-badge');
    const active = this.alerts.filter((x) => !x.acknowledged).length;
    badge.hidden = active === 0;
    badge.textContent = a.unacked_critical > 0 ? String(a.unacked_critical) : String(active);
    badge.classList.toggle('warn', a.unacked_critical === 0);
  }

  _renderBanner() {
    const crit = this.alerts.filter((x) => x.requires_ack && !x.acknowledged);
    const banner = $('alert-banner');
    if (!crit.length) { banner.hidden = true; return; }
    const top = crit[0];
    banner.hidden = false;
    $('ab-title').textContent = `${top.title}${top.drone_ids.length ? ` · ${top.drone_ids.map((i) => `D${String(i).padStart(2, '0')}`).join(', ')}` : ''}`;
    $('ab-msg').textContent = top.message;
    $('ab-count').textContent = crit.length > 1 ? `+${crit.length - 1} more` : (top.count > 1 ? `×${top.count}` : '');
    $('ab-ack').dataset.id = String(top.id);
  }

  _renderList() {
    const list = $('alerts-list');
    if (!this.alerts.length) { list.innerHTML = '<div class="empty">No active alerts.</div>'; return; }
    const order = { CRITICAL: 0, WARNING: 1, INFO: 2 };
    const rows = [...this.alerts].sort((x, y) => (x.acknowledged - y.acknowledged) || (order[x.priority] - order[y.priority]) || (y.last_time - x.last_time));
    list.innerHTML = rows.map((x) => {
      const drones = x.drone_ids.map((i) => `D${String(i).padStart(2, '0')}`).join(' ');
      const action = x.acknowledged ? `<span class="done">ack ${esc(x.ack_by ?? '')}</span>`
        : `<button type="button" class="ack${x.requires_ack ? '' : ' soft'}" data-ack="${x.id}">${x.requires_ack ? 'ACK' : 'Dismiss'}</button>`;
      return `<div class="alert-row ${x.priority}${x.acknowledged ? ' acked' : ''}">
        <span class="pr">${x.priority}</span><span class="tm">${clock(x.last_time)}</span>
        <span class="ti" title="${esc(x.title)}">${esc(x.title)}${x.count > 1 ? ` ×${x.count}` : ''}</span>
        <span class="ms" title="${esc(x.message)}">${esc(x.message)}</span>
        <span class="dr"${x.drone_ids.length ? ` data-drone="${x.drone_ids.join(',')}" title="Select"` : ''}>${drones}</span>${action}</div>`;
    }).join('');
  }
}
