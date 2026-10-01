// Drone visuals: quadrotor model, heading, label, trail, flight path, safety radius, home pad,
// neighbour links. Positions are smoothed between 20 Hz telemetry frames for 60 fps rendering.
import * as THREE from 'three';
import { CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { angleDelta, headingToRotationY, vecToThree } from './coords.js';
import { VISUAL_STATES, classify } from './state.js';

const MODEL_SCALE = 4;        // minimum exaggeration of the 0.5 m airframe
const SCREEN_SCALE = 0.06;    // beyond that, keep a roughly constant on-screen size (scale per metre of camera distance)
const MAX_SCALE = 80;         // all three are multiplied by the user's "Drone size" setting
const TRAIL_POINTS = 500;
const TRAIL_MIN_STEP = 0.8;   // m between trail samples
const SMOOTHING = 12;         // 1/s, exponential smoothing rate

// Shared geometry/materials (one instance for the whole fleet).
const GEO = {
  body: new THREE.BoxGeometry(0.3, 0.1, 0.3),
  arm: new THREE.BoxGeometry(0.75, 0.035, 0.045),
  rotor: new THREE.CylinderGeometry(0.15, 0.15, 0.012, 20),
  nose: new THREE.ConeGeometry(0.06, 0.22, 12).rotateZ(-Math.PI / 2),
  hit: new THREE.SphereGeometry(0.9, 8, 6),
  ring: new THREE.RingGeometry(0.95, 1.12, 40).rotateX(-Math.PI / 2),
  sphere: new THREE.SphereGeometry(1, 24, 16),
  warnRing: new THREE.RingGeometry(0.97, 1, 64).rotateX(-Math.PI / 2),
  pad: new THREE.RingGeometry(1.2, 1.5, 32).rotateX(-Math.PI / 2),
  marker: new THREE.OctahedronGeometry(0.6),
};
const MAT = {
  arm: new THREE.MeshStandardMaterial({ color: 0x2a3440, metalness: 0.4, roughness: 0.6 }),
  rotor: new THREE.MeshBasicMaterial({ color: 0xaac4dd, transparent: true, opacity: 0.35, depthWrite: false }),
  nose: new THREE.MeshBasicMaterial({ color: 0xffffff }),
  hit: new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false }),
  select: new THREE.MeshBasicMaterial({ color: 0x4ea8ff, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }),
  pad: new THREE.MeshBasicMaterial({ color: 0x3a557a, side: THREE.DoubleSide, transparent: true, opacity: 0.8 }),
  stem: new THREE.LineBasicMaterial({ color: 0x6f8aa8, transparent: true, opacity: 0.35 }),
  ghost: new THREE.MeshBasicMaterial({ color: 0x7fd1ff, wireframe: true, transparent: true, opacity: 0.9, depthTest: false }),
  errLine: new THREE.LineBasicMaterial({ color: 0x7fd1ff, transparent: true, opacity: 0.8, depthTest: false }),
};

class DroneVisual {
  constructor(layer, t) {
    this.id = t.drone_id;
    this.layer = layer;
    this.anchor = new THREE.Group();            // position only (labels, rings stay level)
    this.craft = new THREE.Group();             // attitude
    this.craft.rotation.order = 'YZX';          // yaw, then pitch (about lateral), then roll
    this.craft.scale.setScalar(MODEL_SCALE);
    this.anchor.add(this.craft);

    this.bodyMat = new THREE.MeshStandardMaterial({ color: 0x3ddc97, emissive: 0x3ddc97, emissiveIntensity: 0.35, metalness: 0.3, roughness: 0.5 });
    this.craft.add(new THREE.Mesh(GEO.body, this.bodyMat));
    for (const a of [Math.PI / 4, -Math.PI / 4]) {
      const arm = new THREE.Mesh(GEO.arm, MAT.arm);
      arm.rotation.y = a;
      this.craft.add(arm);
    }
    this.rotors = [];
    for (const [x, z] of [[0.265, 0.265], [0.265, -0.265], [-0.265, 0.265], [-0.265, -0.265]]) {
      const r = new THREE.Mesh(GEO.rotor, MAT.rotor);
      r.position.set(x, 0.05, z);
      this.craft.add(r);
      this.rotors.push(r);
    }
    const nose = new THREE.Mesh(GEO.nose, MAT.nose);
    nose.position.set(0.25, 0, 0);               // local +X = forward
    this.craft.add(nose);

    this.hit = new THREE.Mesh(GEO.hit, MAT.hit);
    this.hit.scale.setScalar(MODEL_SCALE);
    this.hit.userData.droneId = this.id;
    this.anchor.add(this.hit);

    this.selectRing = new THREE.Mesh(GEO.ring, MAT.select);
    this.selectRing.scale.setScalar(MODEL_SCALE * 1.2);
    this.selectRing.visible = false;
    this.anchor.add(this.selectRing);

    // Pulsing red ring while the drone is part of an unacknowledged CRITICAL alert.
    this.alertMat = new THREE.MeshBasicMaterial({ color: 0xff3030, side: THREE.DoubleSide, transparent: true, opacity: 0.9, depthTest: false });
    this.alertRing = new THREE.Mesh(GEO.ring, this.alertMat);
    this.alertRing.visible = false;
    this.alertRing.renderOrder = 5;
    this.anchor.add(this.alertRing);
    this.flash = false;

    this.safetyMat = new THREE.MeshBasicMaterial({ color: 0xff4fd8, transparent: true, opacity: 0.08, depthWrite: false });
    this.safety = new THREE.Mesh(GEO.sphere, this.safetyMat);
    this.safety.visible = false;
    this.anchor.add(this.safety);
    this.warnRing = new THREE.Mesh(GEO.warnRing, new THREE.MeshBasicMaterial({ color: 0xf5c542, side: THREE.DoubleSide, transparent: true, opacity: 0.5 }));
    this.warnRing.visible = false;
    this.anchor.add(this.warnRing);

    this.labelEl = document.createElement('div');
    this.labelEl.className = 'drone-label';
    this.label = new CSS2DObject(this.labelEl);
    this.label.position.set(0, 1.4, 0);
    this.anchor.add(this.label);

    // Trail (ring buffer copied into a line strip).
    this.trailPositions = new Float32Array(TRAIL_POINTS * 3);
    this.trailCount = 0;
    this.trailGeo = new THREE.BufferGeometry();
    this.trailGeo.setAttribute('position', new THREE.BufferAttribute(this.trailPositions, 3).setUsage(THREE.DynamicDrawUsage));
    this.trailGeo.setDrawRange(0, 0);
    this.trailMat = new THREE.LineBasicMaterial({ color: 0x3ddc97, transparent: true, opacity: 0.55 });
    this.trail = new THREE.Line(this.trailGeo, this.trailMat);
    this.trail.frustumCulled = false;
    this._lastTrail = null;

    // Flight path: line to the current target plus a marker.
    this.pathGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    this.pathMat = new THREE.LineDashedMaterial({ color: 0x4ea8ff, dashSize: 2, gapSize: 1.5, transparent: true, opacity: 0.7 });
    this.path = new THREE.Line(this.pathGeo, this.pathMat);
    this.path.frustumCulled = false;
    this.marker = new THREE.Mesh(GEO.marker, new THREE.MeshBasicMaterial({ color: 0x4ea8ff, wireframe: true }));
    this.path.visible = this.marker.visible = false;

    this.pad = new THREE.Mesh(GEO.pad, MAT.pad);
    this.pad.scale.setScalar(1.6);

    // Altitude stem: vertical line to the ground, the main depth cue in a perspective view.
    this.stemGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    this.stem = new THREE.Line(this.stemGeo, MAT.stem);
    this.stem.frustumCulled = false;

    // Estimated position (navigation filter) as a ghost, joined to the true position (toggle "Estimate").
    this.ghost = new THREE.Mesh(GEO.marker, MAT.ghost);
    this.ghost.visible = false;
    this.ghostTarget = new THREE.Vector3();
    this.errGeo = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    this.errLine = new THREE.Line(this.errGeo, MAT.errLine);
    this.errLine.frustumCulled = false;
    this.errLine.visible = false;

    layer.root.add(this.anchor, this.trail, this.path, this.marker, this.pad, this.stem, this.ghost, this.errLine);

    this.target = new THREE.Vector3();
    this.goalYaw = 0; this.goalRoll = 0; this.goalPitch = 0;
    this.data = null;
    this._first = true;
    this.update(t);
  }

  update(t) {
    this.data = t;
    vecToThree(t.position, this.target);
    this.goalYaw = headingToRotationY(t.heading);
    this.goalPitch = THREE.MathUtils.degToRad(t.pitch);
    this.goalRoll = THREE.MathUtils.degToRad(t.roll);
    if (this._first) {
      this.anchor.position.copy(this.target);
      this.craft.rotation.set(this.goalRoll, this.goalYaw, this.goalPitch);
      this._first = false;
    }
    vecToThree(t.home, this.pad.position).y += 0.04;

    const state = classify(t);
    const vis = VISUAL_STATES[state];
    this.state = state;
    this.bodyMat.color.set(vis.color);
    this.bodyMat.emissive.set(vis.color);
    this.trailMat.color.set(vis.color);
    const hw = t.source && t.source !== 'sim' ? `<span class="hw-tag">${t.source === 'mavsdk' ? 'SDK' : 'MAV'}</span>` : '';
    this.labelEl.innerHTML = `${t.name}${hw}<span class="bat">${Math.round(t.battery)}%</span>`;
    this.labelEl.style.borderColor = vis.color;

    const opts = this.layer.options;
    const w = this.layer.world;
    const sep = w?.swarm?.separation_distance ?? 5;
    const warn = w?.swarm?.warning_distance ?? 10;
    this.safety.scale.setScalar(sep);
    this.warnRing.scale.setScalar(warn);
    const conflict = t.collision_state === 'AVOIDANCE' || t.collision_state === 'COLLISION';
    this.safety.visible = t.airborne && (opts.safety || conflict || this.selected);
    this.safetyMat.color.set(conflict ? 0xff4d4d : 0xff4fd8);
    this.safetyMat.opacity = conflict ? 0.16 : 0.07;
    this.warnRing.visible = t.airborne && (opts.safety || this.selected || t.collision_state === 'WARNING');

    const showGhost = opts.estimate && t.est_position;
    this.ghost.visible = this.errLine.visible = !!showGhost;
    if (showGhost) {
      vecToThree(t.est_position, this.ghostTarget);
      if (!this._ghostShown) this.ghost.position.copy(this.ghostTarget);
    }
    this._ghostShown = !!showGhost;
    if (t.communication === 'LOST' && t.telemetry_age_s > 1) {
      this.labelEl.innerHTML = `${t.name}<span class="bat">NO LINK ${Math.round(t.telemetry_age_s)}s</span>`;
    }

    const tgt = t.target;
    const showPath = tgt && (opts.targets || this.selected) && t.airborne;
    this.path.visible = this.marker.visible = !!showPath;
    if (showPath) {
      vecToThree(tgt, this.marker.position);
    }
  }

  setSelected(sel) {
    this.selected = sel;
    this.selectRing.visible = sel;
    this.labelEl.classList.toggle('selected', sel);
    if (this.data) this.update(this.data);
  }

  animate(dt, time, camera) {
    const k = 1 - Math.exp(-dt * SMOOTHING);
    this.anchor.position.lerp(this.target, k);

    const f = this.layer.options.size;
    const s = THREE.MathUtils.clamp(camera.position.distanceTo(this.anchor.position) * SCREEN_SCALE * f,
      MODEL_SCALE * f, MAX_SCALE * f);
    this.craft.scale.setScalar(s);
    this.hit.scale.setScalar(s);
    this.selectRing.scale.setScalar(s * 1.2);
    this.label.position.y = 0.55 * s;

    const ap = this.anchor.position;
    const ground = this.layer.heightAt ? this.layer.heightAt(ap.x, -ap.z) : (this.layer.world?.ground_level ?? 0);
    const sa = this.stemGeo.attributes.position;
    sa.setXYZ(0, ap.x, ap.y, ap.z);
    sa.setXYZ(1, ap.x, ground, ap.z);
    sa.needsUpdate = true;
    if (this.ghost.visible) {
      this.ghost.position.lerp(this.ghostTarget, k);
      this.ghost.scale.setScalar(s * 0.25);
      const ea = this.errGeo.attributes.position;
      ea.setXYZ(0, ap.x, ap.y, ap.z);
      ea.setXYZ(1, this.ghost.position.x, this.ghost.position.y, this.ghost.position.z);
      ea.needsUpdate = true;
    }
    this.stem.visible = !!this.data?.airborne;
    const r = this.craft.rotation;
    r.y += angleDelta(r.y, this.goalYaw) * k;
    r.z += (this.goalPitch - r.z) * k;
    r.x += (this.goalRoll - r.x) * k;

    const armed = this.data?.armed;
    if (armed) for (const rotor of this.rotors) rotor.rotation.y += dt * 40;

    // Emergency blink; critical-alert flash (red pulse) until the operator acknowledges.
    const vis = VISUAL_STATES[this.state];
    this.bodyMat.emissiveIntensity = vis?.blink || this.flash ? (Math.sin(time * 10) > 0 ? 0.9 : 0.1) : 0.35;
    this.alertRing.visible = this.flash;
    if (this.flash) {
      const pulse = (time * 1.5) % 1;
      this.alertRing.scale.setScalar(s * (1.3 + 1.7 * pulse));
      this.alertMat.opacity = 0.9 * (1 - pulse);
      this.bodyMat.emissive.set(Math.sin(time * 10) > 0 ? 0xff2020 : vis.color);
    }

    // Trail.
    const p = this.anchor.position;
    if (this.data?.airborne && this.layer.options.trails) {
      if (!this._lastTrail || this._lastTrail.distanceToSquared(p) > TRAIL_MIN_STEP ** 2) {
        this._pushTrail(p);
        this._lastTrail = p.clone();
      }
    }
    this.trail.visible = this.layer.options.trails;

    if (this.path.visible) {
      const a = this.pathGeo.attributes.position;
      a.setXYZ(0, p.x, p.y, p.z);
      a.setXYZ(1, this.marker.position.x, this.marker.position.y, this.marker.position.z);
      a.needsUpdate = true;
      this.path.computeLineDistances();
      this.marker.rotation.y += dt;
    }
    this.label.visible = this.layer.options.labels || this.selected;
  }

  _pushTrail(p) {
    const buf = this.trailPositions;
    if (this.trailCount === TRAIL_POINTS) {
      buf.copyWithin(0, 3);
      this.trailCount -= 1;
    }
    buf.set([p.x, p.y, p.z], this.trailCount * 3);
    this.trailCount += 1;
    this.trailGeo.setDrawRange(0, this.trailCount);
    this.trailGeo.attributes.position.needsUpdate = true;
  }

  clearTrail() {
    this.trailCount = 0;
    this._lastTrail = null;
    this.trailGeo.setDrawRange(0, 0);
  }

  dispose() {
    this.layer.root.remove(this.anchor, this.trail, this.path, this.marker, this.pad, this.stem, this.ghost, this.errLine);
    this.errGeo.dispose();
    this.anchor.remove(this.label);
    this.labelEl.remove();
    this.bodyMat.dispose(); this.trailMat.dispose(); this.pathMat.dispose(); this.safetyMat.dispose(); this.alertMat.dispose();
    this.trailGeo.dispose(); this.pathGeo.dispose(); this.stemGeo.dispose();
    this.marker.material.dispose(); this.warnRing.material.dispose();
  }
}

export class DroneLayer {
  constructor(swarmScene) {
    this.sceneRef = swarmScene;
    this.root = new THREE.Group();
    swarmScene.scene.add(this.root);
    this.visuals = new Map();
    this.world = null;
    this.selection = new Set();
    this.options = { trails: true, targets: true, links: false, safety: false, labels: true, size: 1, estimate: false };
    this.heightAt = null;        // terrain sampler (x, y) -> ENU z, set by main.js
    this._time = 0;

    this.linkGeo = new THREE.BufferGeometry();
    this._linkCapacity = 0;
    this.links = new THREE.LineSegments(this.linkGeo, new THREE.LineBasicMaterial({ color: 0x4ea8ff, transparent: true, opacity: 0.28 }));
    this.links.frustumCulled = false;
    this.links.visible = false;
    this.root.add(this.links);

    // Formation overlay: a cross on every slot plus the reference heading line.
    this.formGeo = new THREE.BufferGeometry();
    this._formCapacity = 0;
    this.formation = new THREE.LineSegments(this.formGeo, new THREE.LineBasicMaterial({ color: 0xd9bc84, transparent: true, opacity: 0.9 }));
    this.formation.frustumCulled = false;
    this.formation.visible = false;
    this.root.add(this.formation);
  }

  /** Show formation slots and reference heading (``f`` = swarm_control.formation, or null). */
  setFormation(f) {
    if (!f || !f.slots?.length) {
      this.formation.visible = false;
      return;
    }
    const pts = [];
    const c = 1.6;   // half-size of the slot cross [m]
    for (const s of f.slots) {
      pts.push(s.x - c, s.z, -s.y, s.x + c, s.z, -s.y);
      pts.push(s.x, s.z, -(s.y - c), s.x, s.z, -(s.y + c));
    }
    const r = f.reference_position;
    const h = THREE.MathUtils.degToRad(f.heading);
    const len = Math.max(10, f.spacing);
    pts.push(r.x, r.z, -r.y, r.x + Math.sin(h) * len, r.z, -(r.y + Math.cos(h) * len));
    if (pts.length > this._formCapacity) {
      this._formCapacity = Math.max(pts.length, this._formCapacity * 2, 240);
      this.formGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this._formCapacity), 3));
    }
    const attr = this.formGeo.attributes.position;
    attr.array.set(pts);
    attr.needsUpdate = true;
    this.formGeo.setDrawRange(0, pts.length / 3);
    this.formation.visible = true;
  }

  sync(drones, world) {
    this.world = world;
    const seen = new Set();
    for (const t of drones) {
      seen.add(t.drone_id);
      const v = this.visuals.get(t.drone_id);
      if (v) v.update(t);
      else {
        const nv = new DroneVisual(this, t);
        nv.setSelected(this.selection.has(t.drone_id));
        nv.flash = this.flashIds?.has(t.drone_id) ?? false;
        this.visuals.set(t.drone_id, nv);
      }
    }
    for (const [id, v] of this.visuals) {
      if (!seen.has(id)) { v.dispose(); this.visuals.delete(id); }
    }
    this._updateLinks(drones);
  }

  _updateLinks(drones) {
    this.links.visible = this.options.links;
    if (!this.options.links) return;
    const byId = new Map(drones.map((d) => [d.drone_id, d]));
    const pts = [];
    for (const d of drones) {
      if (!d.airborne) continue;
      for (const n of d.neighbors) {
        const o = byId.get(n);
        if (n > d.drone_id && o?.airborne) {
          pts.push(d.position.x, d.position.z, -d.position.y, o.position.x, o.position.z, -o.position.y);
        }
      }
    }
    if (pts.length > this._linkCapacity) {
      this._linkCapacity = Math.max(pts.length, this._linkCapacity * 2, 600);
      this.linkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this._linkCapacity), 3));
    }
    if (this._linkCapacity) {
      const attr = this.linkGeo.attributes.position;
      attr.array.set(pts);
      attr.needsUpdate = true;
    }
    this.linkGeo.setDrawRange(0, pts.length / 3);
  }

  reset() {
    for (const v of this.visuals.values()) v.clearTrail();
  }

  setSelection(ids) {
    this.selection = new Set(ids);
    for (const [id, v] of this.visuals) v.setSelected(this.selection.has(id));
  }

  /** Drones to flash (members of unacknowledged CRITICAL alerts). */
  setFlash(ids) {
    this.flashIds = new Set(ids);
    for (const [id, v] of this.visuals) {
      v.flash = this.flashIds.has(id);
      if (!v.flash && v.data) v.update(v.data);     // restore the state colour
    }
  }

  setOption(name, value) {
    this.options[name] = value;
    for (const v of this.visuals.values()) if (v.data) v.update(v.data);
    if (name === 'links') this.links.visible = value;
  }

  pickables() {
    return [...this.visuals.values()].map((v) => v.hit);
  }

  positionOf(id) {
    return this.visuals.get(id)?.anchor.position ?? null;
  }

  animate(dt) {
    this._time += dt;
    const camera = this.sceneRef.camera;
    for (const v of this.visuals.values()) v.animate(dt, this._time, camera);
  }
}
