// Swarm Control Center bootstrap.
import { auth } from './auth.js';
import { SwarmLink } from './net.js';
import { SwarmScene } from './scene.js';
import { DroneLayer } from './drones.js';
import { WorldOverlays } from './overlays.js';
import { MissionPlanner } from './plan.js';
import { ReplayController } from './replay.js';
import { Dashboard } from './ui.js';
import { WorldScene } from './world.js';

const link = new SwarmLink();
const scene = new SwarmScene(document.getElementById('scene'));
const layer = new DroneLayer(scene);
const overlays = new WorldOverlays(scene);
const world3d = new WorldScene(scene);
layer.heightAt = (x, y) => world3d.heightAt(x, y);
const ui = new Dashboard({ link, scene, layer, overlays });
ui.world3d = world3d;
const planner = new MissionPlanner(ui);
ui.planner = planner;
const replay = new ReplayController(ui);
ui.replay = replay;

// Live telemetry drives the dashboard, except while a recorded run is replayed.
link.addEventListener('telemetry', (e) => {
  if (replay.active) replay.onLive(e.detail);
  else ui.onTelemetry(e.detail);
});
link.addEventListener('status', (e) => ui.onConnection(e.detail));
link.addEventListener('latency', (e) => ui.onLatency(e.detail));

auth.addEventListener('login', (e) => ui.setUser(e.detail, auth.enabled));
auth.addEventListener('logout', () => ui.setUser(null, auth.enabled));
document.getElementById('btn-logout').onclick = () => auth.logout();

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  replay.animate(dt);
  layer.animate(dt);
  ui.animate(dt);
  if (ui.planVisible) planner.animate();
  else scene.render(dt);          // the 3D view is hidden under the PLAN map: skip its render
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// Log in first (immediate when security is off or the session is still valid), then connect.
auth.ensure().then((user) => {
  ui.setUser(user, auth.enabled);
  link.connect();
});

// Handy for debugging from the browser console.
window.swarm = { link, scene, layer, overlays, world3d, planner, replay, ui, auth };
