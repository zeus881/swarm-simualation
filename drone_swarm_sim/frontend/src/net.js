// WebSocket link to the simulation backend with automatic reconnect,
// request/response correlation and a REST fallback for commands.

import { auth } from './auth.js';

const REQUEST_TIMEOUT_MS = 6000;
const AUTH_CLOSE_CODE = 4401;         // server: login required / session expired

export class SwarmLink extends EventTarget {
  constructor(url) {
    super();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    this.url = url ?? `${proto}://${location.host}/ws/telemetry`;
    this.ws = null;
    this.connected = false;
    this._seq = 0;
    this._pending = new Map();
    this._backoff = 500;
    this._pingTimer = null;
    this._paused = false;             // no reconnects while logged out
    auth.addEventListener('logout', () => { this._paused = true; this.ws?.close(); });
    auth.addEventListener('login', () => { if (this._paused) { this._paused = false; this.connect(); } });
  }

  connect() {
    const ws = new WebSocket(auth.withToken(this.url));
    this.ws = ws;
    ws.onopen = () => {
      this.connected = true;
      this._backoff = 500;
      this._emit('status', { connected: true });
      this._pingTimer = setInterval(() => this._ping(), 2000);
    };
    ws.onmessage = (ev) => this._onMessage(ev.data);
    ws.onclose = (ev) => {
      const wasConnected = this.connected;
      this.connected = false;
      clearInterval(this._pingTimer);
      for (const [, p] of this._pending) p.reject(new Error('connection lost'));
      this._pending.clear();
      if (ev.code === AUTH_CLOSE_CODE) {
        this._paused = true;
        auth.expired(this._authMessage || 'Please log in');
        return;
      }
      if (this._paused || this.ws !== ws) return;
      this._emit('status', { connected: false, wasConnected });
      setTimeout(() => { if (!this._paused) this.connect(); }, this._backoff);
      this._backoff = Math.min(this._backoff * 2, 5000);
    };
    ws.onerror = () => ws.close();
  }

  _emit(type, detail) { this.dispatchEvent(new CustomEvent(type, { detail })); }

  _onMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }
    if (msg.type === 'telemetry') { this._emit('telemetry', msg); return; }
    if (msg.type === 'auth_error') { this._authMessage = msg.message; return; }
    if (msg.type === 'pong') {
      this._emit('latency', { rtt: performance.now() - msg.client_time });
    }
    const pending = msg.id != null ? this._pending.get(msg.id) : null;
    if (pending) {
      this._pending.delete(msg.id);
      clearTimeout(pending.timer);
      pending.resolve(msg);
    } else if (msg.type === 'error') {
      console.warn('server error:', msg.message);
    }
  }

  _ping() {
    if (this.connected) this.ws.send(JSON.stringify({ type: 'ping', id: null, client_time: performance.now() }));
  }

  _request(payload) {
    return new Promise((resolve, reject) => {
      const id = ++this._seq;
      const timer = setTimeout(() => {
        this._pending.delete(id);
        reject(new Error('request timed out'));
      }, REQUEST_TIMEOUT_MS);
      this._pending.set(id, { resolve, reject, timer });
      this.ws.send(JSON.stringify({ ...payload, id }));
    });
  }

  /** Send an operator command. droneIds = null addresses the whole swarm. */
  async command(type, droneIds = null, params = {}) {
    const command = { type, drone_ids: droneIds, params };
    if (this.connected) return this._request({ type: 'command', command });
    const res = await fetch('/api/commands', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(command),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? `HTTP ${res.status}`);
    return res.json();
  }

  /** Simulation control: start | pause | reset. */
  async simulation(action) {
    if (this.connected) return this._request({ type: 'simulation', action });
    const res = await fetch(`/api/simulation/${action}`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }
}
