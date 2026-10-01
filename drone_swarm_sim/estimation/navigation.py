"""Navigation filter: fuses the simulated sensors into each drone's estimated state.

Position / velocity: :class:`BatchKalmanFilter` predicted with the accelerometer every tick and
corrected with GPS (x, y, z) and the barometer (z). Heading: a complementary filter
``psi <- psi + r dt + k * wrap(psi_compass - psi)`` (gyro integration corrected by the compass).

The estimate is written back to the drones (``est_position``, ``est_velocity``, ``est_heading``,
``pos_sigma``). With ``sensors.control_source: estimate`` the autopilot flies on it, so GPS drift or a
GPS loss moves the real drone, exactly as on a real vehicle; collision and obstacle avoidance keep using
the true geometry (they stand for onboard relative sensing).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .kalman_filter import BatchKalmanFilter

if TYPE_CHECKING:
    from simulation.config import SensorsConfig
    from simulation.drone import Drone
    from simulation.sensors import Measurements

HEADING_GAIN = 0.05        # complementary-filter compass weight per tick


class Navigator:
    def __init__(self, config: "SensorsConfig") -> None:
        self.cfg = config
        self.ids: list[int] = []
        self.kf = BatchKalmanFilter(np.zeros((0, 3)), np.zeros((0, 3)), process_noise=config.process_noise)
        self.heading = np.zeros(0)

    def _sync(self, drones: list["Drone"]) -> None:
        """Rows follow the swarm: survivors keep their filter, new drones start at their position."""
        ids = [d.id for d in drones]
        if ids == self.ids:
            return
        index = {i: k for k, i in enumerate(self.ids)}
        kf, n = self.kf, len(drones)
        pos_var = self.cfg.gps_noise_h ** 2 + self.cfg.gps_drift ** 2

        def gather(arr: np.ndarray, fresh) -> np.ndarray:
            return np.array([arr[index[d.id]] if d.id in index else fresh(d) for d in drones], dtype=np.float64)

        kf.p = gather(kf.p, lambda d: d.position).reshape(n, 3)
        kf.v = gather(kf.v, lambda d: np.zeros(3)).reshape(n, 3)
        kf.P00 = gather(kf.P00, lambda d: np.full(3, pos_var)).reshape(n, 3)
        kf.P01 = gather(kf.P01, lambda d: np.zeros(3)).reshape(n, 3)
        kf.P11 = gather(kf.P11, lambda d: np.ones(3)).reshape(n, 3)
        self.heading = np.array([self.heading[index[d.id]] if d.id in index else d.body.yaw for d in drones])
        self.ids = ids

    def step(self, dt: float, drones: list["Drone"], meas: "Measurements") -> None:
        if not drones:
            self.ids = []
            return
        self._sync(drones)
        c = self.cfg
        kf = self.kf
        grounded = np.array([not d.airborne for d in drones])
        accel = np.where(grounded[:, None], 0.0, meas.accel)       # resting on the ground: no motion
        kf.predict(accel, dt)
        if np.any(grounded):
            kf.v[grounded] = 0.0
        var = np.array([c.gps_noise_h ** 2, c.gps_noise_h ** 2, c.gps_noise_v ** 2]) + 1e-6
        kf.update_position(meas.gps, var, meas.gps_valid)
        kf.update_position(meas.baro[:, None], c.baro_noise ** 2 + 1e-6, meas.baro_valid, axes=slice(2, 3))
        err = (meas.heading - self.heading + math.pi) % (2 * math.pi) - math.pi
        self.heading = (self.heading + meas.yaw_rate * dt + HEADING_GAIN * err + math.pi) % (2 * math.pi) - math.pi
        sigma = kf.horizontal_sigma()
        for k, d in enumerate(drones):
            d.est_position = kf.p[k].copy()
            d.est_velocity = kf.v[k].copy()
            d.est_heading = float(self.heading[k])
            d.pos_sigma = float(sigma[k])
