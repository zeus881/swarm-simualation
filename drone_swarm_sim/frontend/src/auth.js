// Operator / Observer login (Stage 5).
//
// The bearer token lives in sessionStorage (per browser tab, gone when the tab closes). Every same-origin
// /api request gets it automatically (fetch is wrapped once here), the WebSocket passes it as ?token=.
// A 401 anywhere brings the login screen back.

const TOKEN_KEY = 'gandiv.gcs.token';
const $ = (id) => document.getElementById(id);

class AuthSession extends EventTarget {
  constructor() {
    super();
    this.enabled = false;
    this.user = null;            // {username, role, can_control}
    this.token = null;
    try { this.token = sessionStorage.getItem(TOKEN_KEY); } catch { /* storage unavailable */ }
    this._nativeFetch = window.fetch.bind(window);
    window.fetch = (input, init) => this._fetch(input, init);
  }

  get canControl() { return !!this.user?.can_control; }

  async _fetch(input, init = {}) {
    const url = typeof input === 'string' ? input : input.url;
    const ours = this.token && (url.startsWith('/api/') || url.startsWith(`${location.origin}/api/`));
    if (ours) {
      const headers = new Headers(init.headers || (typeof input === 'string' ? undefined : input.headers));
      if (!headers.has('Authorization')) headers.set('Authorization', `Bearer ${this.token}`);
      init = { ...init, headers };
    }
    const res = await this._nativeFetch(input, init);
    if (res.status === 401 && this.enabled && !url.includes('/api/auth/login')) this.expired('Session expired — please log in again');
    return res;
  }

  /** Append the token to a same-origin URL (download links, WebSocket). */
  withToken(url) {
    if (!this.token) return url;
    return `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(this.token)}`;
  }

  _store(token) {
    this.token = token;
    try { if (token) sessionStorage.setItem(TOKEN_KEY, token); else sessionStorage.removeItem(TOKEN_KEY); } catch { /* ignore */ }
  }

  /** Resolves with the user once logged in (immediately when security is disabled or the token is still valid). */
  async ensure() {
    try {
      const cfg = await (await this._nativeFetch('/api/auth/config')).json();
      this.enabled = !!cfg.enabled;
    } catch {
      this.enabled = false;         // backend unreachable: the connection overlay takes over
    }
    if (!this.enabled) {
      this.user = { username: 'local', role: 'operator', can_control: true };
      return this.user;
    }
    if (this.token) {
      const res = await this._fetch('/api/auth/me');
      if (res.ok) { this.user = (await res.json()).user; return this.user; }
      this._store(null);
    }
    return this._prompt();
  }

  _prompt(message = '') {
    const box = $('login');
    box.hidden = false;
    $('login-error').textContent = message;
    $('login-user').focus();
    return new Promise((resolve) => {
      const form = $('login-form');
      form.onsubmit = async (e) => {
        e.preventDefault();
        const btn = $('login-submit');
        btn.disabled = true;
        $('login-error').textContent = '';
        try {
          const res = await this._nativeFetch('/api/auth/login', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: $('login-user').value.trim(), password: $('login-pass').value }),
          });
          const body = await res.json().catch(() => ({}));
          if (!res.ok) { $('login-error').textContent = body.detail ?? `Login failed (HTTP ${res.status})`; return; }
          this._store(body.token);
          this.user = body.user;
          $('login-pass').value = '';
          box.hidden = true;
          form.onsubmit = null;
          resolve(this.user);
          this.dispatchEvent(new CustomEvent('login', { detail: this.user }));
        } catch (err) {
          $('login-error').textContent = `Cannot reach the GCS server (${err.message})`;
        } finally {
          btn.disabled = false;
        }
      };
    });
  }

  /** Token rejected (expired, revoked, server restarted): back to the login screen. */
  expired(message) {
    if (!this.enabled || !$('login').hidden) return;
    this._store(null);
    this.user = null;
    this.dispatchEvent(new CustomEvent('logout', { detail: { message } }));
    this._prompt(message);
  }

  async logout() {
    if (!this.enabled) return;
    try { await this._fetch('/api/auth/logout', { method: 'POST' }); } catch { /* server gone: forget the token anyway */ }
    this._store(null);
    this.user = null;
    this.dispatchEvent(new CustomEvent('logout', { detail: { message: '' } }));
    this._prompt('');
  }
}

export const auth = new AuthSession();
