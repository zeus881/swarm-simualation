// Visual state classification shared by the 3D view and the panels.
// Priority (highest first) decides which state a drone is drawn in.

export const VISUAL_STATES = {
  EMERGENCY:   { label: 'EMERGENCY',         short: 'EMERGENCY', color: '#ff4d4d', blink: true },
  COLLISION:   { label: 'COLLISION WARNING', short: 'PROXIMITY', color: '#ff4fd8' },
  COMM_LOST:   { label: 'COMM LOST',         short: 'NO LINK',   color: '#8b93a7' },
  LOW_BATTERY: { label: 'LOW BATTERY',       short: 'LOW BATT',  color: '#ff8c42' },
  WARNING:     { label: 'WARNING',           short: 'WARNING',   color: '#f5c542' },
  ONLINE:      { label: 'ONLINE',            short: 'ONLINE',    color: '#3ddc97' },
  GROUNDED:    { label: 'GROUNDED',          short: 'GROUNDED',  color: '#5b8def' },
  FAILED:      { label: 'FAILED',            short: 'FAILED',    color: '#c43030' },
};

export function classify(d) {
  if (d.health === 'FAILED') return 'FAILED';
  if (d.mode === 'EMERGENCY' || d.health === 'CRITICAL'
      || d.battery_state === 'EMERGENCY' || d.battery_state === 'DEPLETED'
      || d.collision_state === 'COLLISION') return 'EMERGENCY';
  if (d.collision_state === 'AVOIDANCE' || d.collision_state === 'WARNING') return 'COLLISION';
  if (d.communication === 'LOST') return 'COMM_LOST';
  if (d.battery_state === 'RETURN_HOME') return 'LOW_BATTERY';
  if (d.health === 'WARNING' || d.battery_state === 'WARNING' || d.communication === 'DEGRADED') return 'WARNING';
  if (!d.airborne) return 'GROUNDED';
  return 'ONLINE';
}

export function batteryColor(pct, thresholds) {
  const t = thresholds ?? { warning: 30, return_home: 20, emergency: 10 };
  if (pct <= t.emergency) return '#ff4d4d';
  if (pct <= t.return_home) return '#ff8c42';
  if (pct <= t.warning) return '#f5c542';
  return '#3ddc97';
}
