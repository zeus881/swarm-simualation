"""Sensor models: GPS (white noise + slowly wandering bias), barometer, compass, IMU (accel, gyro).

Measurements are generated for the whole swarm at once:

* **GPS** at ``gps_rate_hz``: ``z = p + b + n``, ``n ~ N(0, diag(s_h, s_h, s_v)^2)``. The bias ``b`` is a
  first-order Gauss-Markov process (std ``gps_drift``, correlation time ``gps_drift_time_s``)::

      b <- b e^{-dt/T} + gps_drift sqrt(1 - e^{-2 dt/T}) N(0, 1)

  so the fix wanders by metres over minutes, like a real receiver. A GPS loss (failure injection) stops
  the fix: ``fix = NONE``, 0 satellites, HDOP 99.9.
* **Barometer** at ``baro_rate_hz``: altitude with noise and a Gauss-Markov bias (``baro_drift``).
* **Compass**: heading + a constant per-drone bias + noise. **Gyro**: yaw rate + noise.
* **Accelerometer**: acceleration + noise (the Kalman filter's prediction input).

The satellite count and HDOP reported on the HUD vary slowly around a healthy fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SensorsConfig
    from .drone import Drone


@dataclass
class SensorState:
    gps_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    baro_bias: float = 0.0
    compass_bias: float = 0.0              # rad
    gps_lost_until: float = -math.inf
    satellites: float = 14.0


@dataclass
class Measurements:
    ids: list[int]
    gps: np.ndarray                        # (N, 3)
    gps_valid: np.ndarray                  # (N,) bool: a new GPS fix this tick
    gps_fix: np.ndarray                    # (N,) bool: receiver has a fix
    baro: np.ndarray                       # (N,)
    baro_valid: np.ndarray                 # (N,) bool
    accel: np.ndarray                      # (N, 3)
    heading: np.ndarray                    # (N,) compass, rad (ENU yaw)
    yaw_rate: np.ndarray                   # (N,) rad/s


class SensorSuite:
    def __init__(self, config: "SensorsConfig", rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self.states: dict[int, SensorState] = {}
        self._next_gps = 0.0
        self._next_baro = 0.0

    def state(self, drone_id: int) -> SensorState:
        s = self.states.get(drone_id)
        if s is None:
            c = self.cfg
            s = self.states[drone_id] = SensorState(
                gps_bias=self.rng.normal(0.0, 1.0, 3) * np.array([c.gps_drift, c.gps_drift, 0.6 * c.gps_drift]),
                baro_bias=float(self.rng.normal(0.0, c.baro_drift)),
                compass_bias=math.radians(float(self.rng.normal(0.0, c.compass_bias_deg))),
                satellites=float(self.rng.uniform(12, 17)))
        return s

    def set_gps_loss(self, drone_id: int, until: float) -> None:
        self.state(drone_id).gps_lost_until = until

    def clear_gps_loss(self, drone_id: int) -> None:
        self.state(drone_id).gps_lost_until = -math.inf

    def measure(self, t: float, dt: float, drones: list["Drone"]) -> Measurements:
        c = self.cfg
        n = len(drones)
        ids = [d.id for d in drones]
        for gone in [i for i in self.states if i not in set(ids)]:
            del self.states[gone]
        states = [self.state(i) for i in ids]
        pos = np.array([d.position for d in drones]).reshape(n, 3)
        # Gauss-Markov bias evolution (every tick, so the bias is independent of the GPS rate)
        decay = math.exp(-dt / c.gps_drift_time_s)
        kick = math.sqrt(1.0 - decay * decay)
        scale = np.array([c.gps_drift, c.gps_drift, 0.6 * c.gps_drift])
        # vectorised Gauss-Markov updates, written back per drone
        bias = np.array([s.gps_bias for s in states]).reshape(n, 3) * decay + scale * kick * self.rng.standard_normal((n, 3))
        baro_bias = np.array([s.baro_bias for s in states]) * decay + c.baro_drift * kick * self.rng.standard_normal(n)
        sats = np.clip(np.array([s.satellites for s in states]) + self.rng.normal(0.0, 0.02, n), 9, 18)
        for k, s in enumerate(states):
            s.gps_bias, s.baro_bias, s.satellites = bias[k], float(baro_bias[k]), float(sats[k])
        fix = np.array([t >= s.gps_lost_until for s in states], dtype=bool)
        gps_due = t + 1e-9 >= self._next_gps
        if gps_due:
            self._next_gps = t + 1.0 / c.gps_rate_hz
        white = self.rng.standard_normal((n, 3)) * np.array([c.gps_noise_h, c.gps_noise_h, c.gps_noise_v])
        gps = pos + bias + white
        baro_due = t + 1e-9 >= self._next_baro
        if baro_due:
            self._next_baro = t + 1.0 / c.baro_rate_hz
        baro = pos[:, 2] + baro_bias + self.rng.normal(0.0, c.baro_noise, n)
        accel = np.array([d.acceleration for d in drones]).reshape(n, 3) + self.rng.normal(0.0, c.accel_noise, (n, 3))
        yaw = np.array([d.body.yaw for d in drones])
        heading = yaw + np.array([s.compass_bias for s in states]) + np.radians(self.rng.normal(0.0, c.compass_noise_deg, n))
        yaw_rate = np.array([d.body.yaw_rate for d in drones]) + np.radians(self.rng.normal(0.0, c.gyro_noise_deg, n))
        for d, s, has_fix in zip(drones, states, fix):
            d.gps_fix = "3D" if has_fix else "NONE"
            d.satellites = int(round(s.satellites)) if has_fix else 0
            d.hdop = round(0.6 + 6.0 / max(s.satellites, 1.0), 2) if has_fix else 99.9
        return Measurements(ids, gps, fix & gps_due, fix, baro, np.full(n, baro_due), accel, heading, yaw_rate)
