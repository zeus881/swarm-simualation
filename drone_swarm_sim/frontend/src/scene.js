// Three.js scene: renderer, camera/controls, world geometry (ground, grid, bounds, home base).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { enuToThree, threeToEnu } from './coords.js';

const BG = 0x080c11;

export class SwarmScene {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    Object.assign(this.labels.domElement.style, { position: 'absolute', top: '0', left: '0', pointerEvents: 'none' });
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BG);
    this.scene.fog = new THREE.Fog(BG, 900, 4200);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.3, 20000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 6000;
    this.controls.screenSpacePanning = false;

    this.scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x1a2230, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(300, 600, 200);
    this.scene.add(sun);

    this.world = new THREE.Group();
    this.scene.add(this.world);
    this._worldKey = '';
    this.home = new THREE.Vector3();
    this.groundLevel = 0;
    this._raycaster = new THREE.Raycaster();
    this._followTarget = null;

    this.cameraMode = 'free';
    this._chaseDist = 32;
    this.renderer.domElement.addEventListener('wheel', (e) => {
      if (this.cameraMode === 'chase') this._chaseDist = Math.max(12, Math.min(200, this._chaseDist * Math.exp(e.deltaY * 0.001)));
    }, { passive: true });
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.resetView();
  }

  resize() {
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
  }

  /** (Re)build static world geometry when the world description changes. */
  buildWorld(world) {
    const key = JSON.stringify([world.bounds, world.home, world.max_altitude, world.ground_level]);
    if (key === this._worldKey) return;
    this._worldKey = key;
    this.world.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    this.world.clear();

    const { min, max } = world.bounds;
    const sx = max.x - min.x;
    const sy = max.y - min.y;
    const g = world.ground_level;
    this.groundLevel = g;
    const center = enuToThree((min.x + max.x) / 2, (min.y + max.y) / 2, g);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(sx, sy),
      new THREE.MeshStandardMaterial({ color: 0x0f1820, roughness: 1, metalness: 0 }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.copy(center).setY(g - 0.05);
    ground.name = 'ground';
    this.world.add(ground);

    const size = Math.max(sx, sy);
    const major = new THREE.GridHelper(size, Math.round(size / 100), 0x2b3d4f, 0x1b2835);
    major.position.copy(center).setY(g + 0.01);
    this.world.add(major);

    const home = world.home;
    this.home = enuToThree(home.x, home.y, home.z);
    const minor = new THREE.GridHelper(300, 30, 0x243647, 0x16222d);
    minor.position.copy(this.home).setY(g + 0.02);
    this.world.add(minor);

    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, world.max_altitude, sy)),
      new THREE.LineBasicMaterial({ color: 0x2b4560, transparent: true, opacity: 0.45 }),
    );
    box.position.copy(center).setY(g + world.max_altitude / 2);
    this.world.add(box);

    // Home base: concentric rings + label.
    for (const [r, color, op] of [[5, 0x5b8def, 0.9], [9, 0x5b8def, 0.35]]) {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(r - 0.4, r, 64),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: op, side: THREE.DoubleSide }),
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.copy(this.home).setY(home.z + 0.05);          // home z includes the terrain height
      this.world.add(ring);
    }
    this.world.add(this._label('HOME', 'world-label home', this.home.clone().setY(home.z + 1)));

    // Cardinal markers at the world edge.
    const pad = 20;
    for (const [txt, x, y] of [['N', home.x, max.y - pad], ['S', home.x, min.y + pad], ['E', max.x - pad, home.y], ['W', min.x + pad, home.y]]) {
      this.world.add(this._label(txt, 'world-label', enuToThree(x, y, g + 2)));
    }
  }

  _label(text, cls, position) {
    const el = document.createElement('div');
    el.className = cls;
    el.textContent = text;
    const obj = new CSS2DObject(el);
    obj.position.copy(position);
    return obj;
  }

  /** Normalised device coordinates from a pointer event. */
  ndc(event) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
  }

  pick(ndc, objects) {
    this._raycaster.setFromCamera(ndc, this.camera);
    return this._raycaster.intersectObjects(objects, false)[0] ?? null;
  }

  /** Intersection of the pointer ray with the terrain (or the flat ground plane), in ENU. */
  groundPoint(ndc) {
    this._raycaster.setFromCamera(ndc, this.camera);
    if (this.terrainMesh) {
      const hit = this._raycaster.intersectObject(this.terrainMesh, false)[0];
      if (hit) return threeToEnu(hit.point);
    }
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -this.groundLevel);
    const hit = new THREE.Vector3();
    return this._raycaster.ray.intersectPlane(plane, hit) ? threeToEnu(hit) : null;
  }

  setFollowTarget(vec3OrNull) { this._followTarget = vec3OrNull; }

  // ------------------------------------------------------------------ camera presets
  /** free | top | chase | orbit */
  setCameraMode(mode) {
    this.cameraMode = mode;
    this._followTarget = null;
    this.controls.autoRotate = mode === 'orbit';
    this.controls.autoRotateSpeed = 0.6;
    this.controls.enableRotate = mode === 'free' || mode === 'orbit';
    this.controls.enableZoom = mode !== 'chase';                  // chase zoom is handled by the wheel listener
    if (mode === 'top') {
      const h = Math.max(120, this.camera.position.distanceTo(this.controls.target));
      this.camera.position.copy(this.controls.target).add(new THREE.Vector3(0, h, 0.01));
    }
    this.controls.update();
  }

  /**
   * Per-frame camera automation for the active preset.
   * @param ctx {chase: {position: Vector3, heading: deg} | null, center: Vector3 | null}
   */
  updateCamera(dt, ctx) {
    const mode = this.cameraMode ?? 'free';
    const k = 1 - Math.exp(-dt * 3);
    const target = this.controls.target;
    if (mode === 'chase' && ctx.chase) {
      const p = ctx.chase.position;
      const h = THREE.MathUtils.degToRad(ctx.chase.heading);
      const forward = new THREE.Vector3(Math.sin(h), 0, -Math.cos(h));     // compass heading in Three axes
      const dist = this._chaseDist;
      const desired = p.clone().addScaledVector(forward, -dist).add(new THREE.Vector3(0, dist * 0.38, 0));
      target.lerp(p, 1 - Math.exp(-dt * 8));
      this.camera.position.lerp(desired, k);
    } else if ((mode === 'orbit' || mode === 'top') && ctx.center) {
      const delta = ctx.center.clone().sub(target).multiplyScalar(1 - Math.exp(-dt * 2));
      target.add(delta);
      this.camera.position.add(delta);
      if (mode === 'top') {                                        // stay straight overhead, keep the zoom
        const h = Math.max(20, this.camera.position.y - target.y);
        this.camera.position.set(target.x, target.y + h, target.z + 0.01);
      }
    }
  }

  /** Camera position and target in ENU (for the minimap). */
  cameraState() {
    return { position: threeToEnu(this.camera.position), target: threeToEnu(this.controls.target) };
  }

  /** Move the view (target and camera together) so it looks at ENU point (x, y). */
  lookAt(x, y) {
    const t = enuToThree(x, y, this.controls.target.y);
    const delta = t.sub(this.controls.target).setY(0);
    this.controls.target.add(delta);
    this.camera.position.add(delta);
    this.controls.update();
  }

  resetView() {
    this.controls.target.copy(this.home).setY(this.groundLevel + 15);
    this.camera.position.copy(this.home).add(new THREE.Vector3(-45, 45, 60));
    this.controls.update();
  }

  topView() {
    this.controls.target.copy(this.home).setY(this.groundLevel);
    this.camera.position.copy(this.home).add(new THREE.Vector3(0, 320, 0.01));
    this.controls.update();
  }

  render(dt) {
    if (this._followTarget) {
      // Move camera and target together so the operator keeps their viewing angle.
      const delta = this._followTarget.clone().sub(this.controls.target);
      const k = 1 - Math.exp(-dt * 4);
      delta.multiplyScalar(k);
      this.controls.target.add(delta);
      this.camera.position.add(delta);
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }
}
