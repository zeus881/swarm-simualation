"""Batched linear Kalman filter for drone position / velocity (docs/ARCHITECTURE.md §6.10).

Per drone and per axis, the state is ``x = [p, v]`` with the accelerometer as control input::

    predict:  p <- p + v dt + a dt^2 / 2,   v <- v + a dt
              P <- F P F^T + Q,   F = [[1, dt], [0, 1]],   Q = q^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    update:   S = P00 + r,   K = [P00, P01] / S,   x <- x + K (z - p),   P <- (I - K H) P

(``q`` = acceleration noise density, ``r`` = measurement variance). Axes are independent, so every
2 x 2 covariance is stored as three arrays ``P00, P01, P11`` of shape (N, 3). All drones and axes are
updated with a handful of NumPy expressions: the cost is almost independent of the swarm size.

:class:`Estimator` is the interface, so an EKF / UKF can replace this filter without touching callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Estimator(ABC):
    @abstractmethod
    def predict(self, accel: np.ndarray, dt: float) -> None: ...

    @abstractmethod
    def update_position(self, z: np.ndarray, variance: np.ndarray, mask: np.ndarray, axes: slice = slice(0, 3)) -> None: ...


class BatchKalmanFilter(Estimator):
    """``n`` independent constant-velocity filters, 3 axes each."""

    def __init__(self, positions: np.ndarray, velocities: np.ndarray, pos_var: float = 4.0, vel_var: float = 1.0,
                 process_noise: float = 0.6) -> None:
        self.p = np.array(positions, dtype=np.float64).reshape(-1, 3)
        self.v = np.array(velocities, dtype=np.float64).reshape(-1, 3)
        n = len(self.p)
        self.P00 = np.full((n, 3), pos_var)
        self.P01 = np.zeros((n, 3))
        self.P11 = np.full((n, 3), vel_var)
        self.q2 = process_noise ** 2

    def predict(self, accel: np.ndarray, dt: float) -> None:
        a = np.asarray(accel, dtype=np.float64).reshape(-1, 3)
        self.p += self.v * dt + 0.5 * a * dt * dt
        self.v += a * dt
        q2 = self.q2
        self.P00 += dt * (2.0 * self.P01 + dt * self.P11) + q2 * dt ** 4 / 4.0
        self.P01 += dt * self.P11 + q2 * dt ** 3 / 2.0
        self.P11 += q2 * dt * dt

    def update_position(self, z: np.ndarray, variance: np.ndarray, mask: np.ndarray, axes: slice = slice(0, 3)) -> None:
        """Position measurement ``z`` (N, k) with per-axis variance, for rows where ``mask`` is True."""
        if not np.any(mask):
            return
        rows = np.flatnonzero(mask)
        zz = np.asarray(z, dtype=np.float64).reshape(len(self.p), -1)[rows]
        r = np.broadcast_to(np.asarray(variance, dtype=np.float64), zz.shape)
        P00, P01, P11 = self.P00[rows, axes], self.P01[rows, axes], self.P11[rows, axes]
        s = P00 + r
        k0, k1 = P00 / s, P01 / s
        innov = zz - self.p[rows, axes]
        self.p[rows, axes] += k0 * innov
        self.v[rows, axes] += k1 * innov
        self.P11[rows, axes] = P11 - k1 * P01
        self.P00[rows, axes] = (1.0 - k0) * P00
        self.P01[rows, axes] = (1.0 - k0) * P01

    def horizontal_sigma(self) -> np.ndarray:
        """1-sigma horizontal position uncertainty per drone [m]."""
        return np.sqrt(self.P00[:, 0] + self.P00[:, 1])
