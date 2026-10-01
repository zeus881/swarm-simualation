// 3D world overlays for the FLY view: geofence (inclusion walls, no-fly prisms, ceiling) and the
// waypoint paths of running missions. Rebuilt only when the server-side version changes.
import * as THREE from 'three';
import { enuToThree } from './coords.js';

const FENCE_COLOR = 0xf5c542;
const NFZ_COLOR = 0xff4d4d;
const PATH_COLOR = 0x3ddc97;

function disposeGroup(group) {
  group.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
  group.clear();
}

/** Vertical wall geometry along a closed polygon from z0 to z1 (ENU), as a triangle list. */
function wallGeometry(poly, z0, z1) {
  const pos = [];
  for (let k = 0; k < poly.length; k += 1) {
    const [ax, ay] = poly[k];
    const [bx, by] = poly[(k + 1) % poly.length];
    const a0 = enuToThree(ax, ay, z0); const a1 = enuToThree(ax, ay, z1);
    const b0 = enuToThree(bx, by, z0); const b1 = enuToThree(bx, by, z1);
    pos.push(a0.x, a0.y, a0.z, b0.x, b0.y, b0.z, b1.x, b1.y, b1.z, a0.x, a0.y, a0.z, b1.x, b1.y, b1.z, a1.x, a1.y, a1.z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.computeVertexNormals();
  return g;
}

function outline(poly, z) {
  const pts = poly.map(([x, y]) => enuToThree(x, y, z));
  pts.push(pts[0].clone());
  return new THREE.BufferGeometry().setFromPoints(pts);
}

export class WorldOverlays {
  constructor(swarmScene) {
    this.scene = swarmScene;
    this.fence = new THREE.Group();
    this.fence.name = 'geofence';
    this.paths = new THREE.Group();
    this.paths.name = 'mission-paths';
    this.measure = new THREE.Group();
    this.measure.name = 'measure';
    swarmScene.scene.add(this.fence, this.paths, this.measure);
    this._fenceVersion = -1;
  }

  /** Measurement line between ENU points a and b (b may be null while the second point is pending). */
  setMeasure(a, b) {
    disposeGroup(this.measure);
    if (!a) return;
    const marker = new THREE.SphereGeometry(1.2, 12, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false });
    for (const p of [a, b].filter(Boolean)) {
      const m = new THREE.Mesh(marker, mat);
      m.position.copy(enuToThree(p.x, p.y, p.z + 0.3));
      m.renderOrder = 6;
      this.measure.add(m);
    }
    if (b) {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([enuToThree(a.x, a.y, a.z + 0.3), enuToThree(b.x, b.y, b.z + 0.3)]),
        new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 3, gapSize: 2, depthTest: false }));
      line.computeLineDistances();
      line.renderOrder = 6;
      this.measure.add(line);
    }
  }

  /** ``gf`` = snapshot.missions.geofence; ``world`` = snapshot.world. */
  setGeofence(gf, world) {
    if (!gf || gf.version === this._fenceVersion) return;
    this._fenceVersion = gf.version;
    disposeGroup(this.fence);
    const ground = world.ground_level;
    const top = ground + (gf.max_altitude ?? world.max_altitude);
    const opacity = gf.enabled ? 1 : 0.4;
    if (gf.inclusion?.length) {
      this.fence.add(new THREE.Mesh(wallGeometry(gf.inclusion, ground, top), new THREE.MeshBasicMaterial({
        color: FENCE_COLOR, transparent: true, opacity: 0.08 * opacity, side: THREE.DoubleSide, depthWrite: false })));
      for (const z of [ground + 0.3, top]) {
        this.fence.add(new THREE.Line(outline(gf.inclusion, z), new THREE.LineBasicMaterial({
          color: FENCE_COLOR, transparent: true, opacity: 0.85 * opacity })));
      }
    }
    for (const zone of gf.exclusions ?? []) {
      this.fence.add(new THREE.Mesh(wallGeometry(zone.polygon, ground, top), new THREE.MeshBasicMaterial({
        color: NFZ_COLOR, transparent: true, opacity: 0.16 * opacity, side: THREE.DoubleSide, depthWrite: false })));
      for (const z of [ground + 0.3, top]) {
        this.fence.add(new THREE.Line(outline(zone.polygon, z), new THREE.LineBasicMaterial({
          color: NFZ_COLOR, transparent: true, opacity: 0.9 * opacity })));
      }
      // Floor hatch: a translucent footprint makes the zone readable from above.
      const shape = new THREE.Shape(zone.polygon.map(([x, y]) => new THREE.Vector2(x, y)));
      const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({
        color: NFZ_COLOR, transparent: true, opacity: 0.18 * opacity, side: THREE.DoubleSide, depthWrite: false }));
      floor.rotation.x = -Math.PI / 2;             // shape XY (East, North) -> Three XZ with Z = -North
      floor.position.y = ground + 0.2;
      this.fence.add(floor);
    }
  }

  /** ``paths`` = /api/missions/active paths; drawn as dashed lines with waypoint markers. */
  setMissionPaths(paths, world) {
    disposeGroup(this.paths);
    const ground = world?.ground_level ?? 0;
    for (const p of paths ?? []) {
      const pts = p.waypoints.filter((w) => ['WAYPOINT', 'LOITER', 'LAND', 'TAKEOFF'].includes(w.action))
        .map((w) => enuToThree(w.x, w.y, ground + (w.action === 'LAND' ? 0 : w.alt)));
      if (pts.length < 1) continue;
      if (pts.length > 1) {
        const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({
          color: PATH_COLOR, dashSize: 4, gapSize: 3, transparent: true, opacity: 0.8 }));
        line.computeLineDistances();
        this.paths.add(line);
      }
      const marker = new THREE.OctahedronGeometry(1.2);
      for (const v of pts) {
        const m = new THREE.Mesh(marker, new THREE.MeshBasicMaterial({ color: PATH_COLOR, wireframe: true }));
        m.position.copy(v);
        this.paths.add(m);
      }
    }
  }
}
