// Frame conventions.
// Simulation: ENU (x = East, y = North, z = Up), heading = compass degrees (0 = North, clockwise).
// Three.js:   y-up right-handed. Mapping: three(x, y, z) = (East, Up, -North), a proper rotation.
import * as THREE from 'three';

export function enuToThree(x, y, z, out = new THREE.Vector3()) {
  return out.set(x, z, -y);
}

export function vecToThree(v, out = new THREE.Vector3()) {
  return out.set(v.x, v.z, -v.y);
}

export function threeToEnu(v) {
  return { x: v.x, y: -v.z, z: v.y };
}

/** Compass heading (deg) -> rotation about Three's +Y axis (rad).
 *  ENU yaw = 90deg - heading, and a +Y rotation by yaw turns +X (East) towards -Z (North). */
export function headingToRotationY(headingDeg) {
  return THREE.MathUtils.degToRad(90 - headingDeg);
}

/** Shortest signed angular difference b - a in (-PI, PI]. */
export function angleDelta(a, b) {
  let d = (b - a) % (Math.PI * 2);
  if (d > Math.PI) d -= Math.PI * 2;
  if (d <= -Math.PI) d += Math.PI * 2;
  return d;
}
