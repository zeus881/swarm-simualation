// Terrain and obstacles in the 3D view (fetched from /api/world/scene when world.scene.version changes).
// Also provides a terrain height sampler for the rest of the UI (altitude stems, ground picking).
import * as THREE from 'three';
import { enuToThree } from './coords.js';

const COLORS = {
  building: 0x8a8f98, buildingEdge: 0x2f3338, tower: 0xb9bec6, towerBand: 0xc62828,
  trunk: 0x6b4a2b, crown: 0x3f6e3a,
};

function disposeGroup(group) {
  group.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
  group.clear();
}

export class WorldScene {
  constructor(swarmScene) {
    this.swarmScene = swarmScene;
    this.terrainGroup = new THREE.Group();
    this.terrainGroup.name = 'terrain';
    this.obstacleGroup = new THREE.Group();
    this.obstacleGroup.name = 'obstacles';
    swarmScene.scene.add(this.terrainGroup, this.obstacleGroup);
    this.version = -1;
    this.grid = null;          // {rows, cols, x_min, y_min, size_x, size_y, heights, min, max}
    this.groundLevel = 0;
    this.obstacles = [];
    this.terrainMesh = null;
    this._loading = false;
  }

  /** Called with every telemetry frame; fetches the scene when its version changes. */
  async sync(world) {
    const v = world.scene?.version;
    if (v == null || v === this.version || this._loading) return;
    this._loading = true;
    try {
      const res = await fetch('/api/world/scene');
      if (!res.ok) return;
      const scene = await res.json();
      this.version = v;
      this.build(scene);
    } catch { /* retried on the next frame */ } finally {
      this._loading = false;
    }
  }

  /** Terrain height (ENU z) at (x, y); flat ground level without a terrain model. */
  heightAt(x, y) {
    const g = this.grid;
    if (!g) return this.groundLevel;
    const dx = g.size_x / (g.cols - 1);
    const dy = g.size_y / (g.rows - 1);
    const fx = Math.min(Math.max((x - g.x_min) / dx, 0), g.cols - 1.000001);
    const fy = Math.min(Math.max((y - g.y_min) / dy, 0), g.rows - 1.000001);
    const c = Math.floor(fx); const r = Math.floor(fy);
    const tx = fx - c; const ty = fy - r;
    const h = (rr, cc) => g.heights[rr * g.cols + cc];
    return this.groundLevel + (h(r, c) * (1 - tx) + h(r, c + 1) * tx) * (1 - ty) + (h(r + 1, c) * (1 - tx) + h(r + 1, c + 1) * tx) * ty;
  }

  build(scene) {
    this.groundLevel = scene.ground_level ?? 0;
    this.obstacles = scene.obstacles ?? [];
    this._buildTerrain(scene.terrain);
    this._buildObstacles(this.obstacles);
  }

  _buildTerrain(grid) {
    disposeGroup(this.terrainGroup);
    this.grid = grid;
    this.terrainMesh = null;
    this.swarmScene.terrainMesh = null;
    const flat = this.swarmScene.world.getObjectByName('ground');
    if (!grid) { if (flat) flat.visible = true; return; }
    if (flat) flat.visible = false;
    const geo = new THREE.PlaneGeometry(grid.size_x, grid.size_y, grid.cols - 1, grid.rows - 1);
    geo.rotateX(-Math.PI / 2);                               // plane XY -> Three XZ (row 0 at the north edge)
    const pos = geo.attributes.position;
    const colors = new Float32Array(pos.count * 3);
    const lo = grid.min; const span = Math.max(grid.max - grid.min, 1);
    const cLow = new THREE.Color(0x3d4a3a); const cMid = new THREE.Color(0x5a6448); const cHigh = new THREE.Color(0x8a7f68);
    const tmp = new THREE.Color();
    for (let i = 0; i < pos.count; i += 1) {
      const row = Math.floor(i / grid.cols);                 // 0 = north edge in PlaneGeometry order
      const col = i % grid.cols;
      const h = grid.heights[(grid.rows - 1 - row) * grid.cols + col];
      pos.setY(i, this.groundLevel + h);
      const t = (h - lo) / span;
      tmp.copy(t < 0.5 ? cLow : cMid).lerp(t < 0.5 ? cMid : cHigh, t < 0.5 ? t * 2 : (t - 0.5) * 2);
      colors.set([tmp.r, tmp.g, tmp.b], i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, metalness: 0, flatShading: false }));
    const cx = grid.x_min + grid.size_x / 2;
    const cy = grid.y_min + grid.size_y / 2;
    mesh.position.copy(enuToThree(cx, cy, 0));
    mesh.name = 'terrain-mesh';
    this.terrainGroup.add(mesh);
    this.terrainMesh = mesh;
    this.swarmScene.terrainMesh = mesh;              // ground picking (Shift+click, measure) hits the terrain
  }

  _buildObstacles(list) {
    disposeGroup(this.obstacleGroup);
    const trees = list.filter((o) => o.kind === 'tree');
    for (const o of list) {
      if (o.kind === 'building') {
        const geo = new THREE.BoxGeometry(o.width, o.height, o.depth);
        const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: COLORS.building, roughness: 0.85, metalness: 0.05 }));
        mesh.position.copy(enuToThree(o.x, o.y, o.base + o.height / 2));
        mesh.rotation.y = THREE.MathUtils.degToRad(o.rotation ?? 0);
        mesh.userData = { obstacle: o.name };
        const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: COLORS.buildingEdge }));
        mesh.add(edges);
        this.obstacleGroup.add(mesh);
      } else if (o.kind === 'tower') {
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(o.radius, o.radius * 1.15, o.height, 16),
          new THREE.MeshStandardMaterial({ color: COLORS.tower, roughness: 0.7 }));
        mesh.position.copy(enuToThree(o.x, o.y, o.base + o.height / 2));
        mesh.userData = { obstacle: o.name };
        this.obstacleGroup.add(mesh);
        // red obstruction band at the top
        const band = new THREE.Mesh(new THREE.CylinderGeometry(o.radius * 1.05, o.radius * 1.05, Math.min(4, o.height * 0.1), 16),
          new THREE.MeshBasicMaterial({ color: COLORS.towerBand }));
        band.position.copy(enuToThree(o.x, o.y, o.base + o.height - Math.min(2, o.height * 0.05)));
        this.obstacleGroup.add(band);
      }
    }
    if (trees.length) {
      // Instanced meshes: one draw call for all trunks and one for all crowns.
      const trunkGeo = new THREE.CylinderGeometry(0.25, 0.35, 1, 6);
      const crownGeo = new THREE.ConeGeometry(1, 1, 8);
      const trunks = new THREE.InstancedMesh(trunkGeo, new THREE.MeshStandardMaterial({ color: COLORS.trunk }), trees.length);
      const crowns = new THREE.InstancedMesh(crownGeo, new THREE.MeshStandardMaterial({ color: COLORS.crown, roughness: 0.9 }), trees.length);
      const m = new THREE.Matrix4();
      trees.forEach((t, i) => {
        const trunkH = t.height * 0.3;
        m.makeScale(t.radius * 0.35, trunkH, t.radius * 0.35).setPosition(enuToThree(t.x, t.y, t.base + trunkH / 2));
        trunks.setMatrixAt(i, m);
        const crownH = t.height * 0.75;
        m.makeScale(t.radius, crownH, t.radius).setPosition(enuToThree(t.x, t.y, t.base + t.height - crownH / 2));
        crowns.setMatrixAt(i, m);
      });
      this.obstacleGroup.add(trunks, crowns);
    }
  }
}
