"""Multirotor translational dynamics and the inner velocity-control loop.

Model (docs/ARCHITECTURE.md §6.2)
---------------------------------
The vehicle is a point mass driven by a *specific thrust* vector ``f``
(thrust / mass, m/s^2). The attitude is implied by the direction of ``f``.

    f_des = a_des + g*z_hat                            desired specific thrust
    f_des <- tilt-limited, 0 <= f_z, |f| <= (T/W)*g     physical envelope
    f     <- f + (f_des - f) * (1 - exp(-dt / tau))     first-order motor/attitude lag
    a      = f - g*z_hat - c_d * (v - w)                gravity + linear drag vs air
    v     <- v + a*dt ;  p <- p + v*dt                  semi-implicit (symplectic) Euler

The model keeps the effects that matter for swarm algorithms (acceleration
and tilt limits, response lag, overshoot, drift in wind) at a fraction of the
cost of a rigid-body/rotor model, so 100+ vehicles run in real time. Rotor-level
fidelity comes from the PX4 SITL + Gazebo path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import DroneConfig
from .types import Vector3

GRAVITY = 9.80665
_Z = np.array((0.0, 0.0, 1.0))


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


@dataclass(slots=True)
class DynamicsParams:
    mass_kg: float
    max_horizontal_speed: float
    max_climb_rate: float
    max_descent_rate: float
    max_acceleration: float
    max_tilt_rad: float
    thrust_to_weight: float
    thrust_time_constant: float
    drag_coefficient: float
    max_yaw_rate: float
    yaw_gain: float
    velocity_kp: float
    velocity_ki: float
    velocity_integral_limit: float

    @classmethod
    def from_config(cls, cfg: DroneConfig) -> "DynamicsParams":
        return cls(
            mass_kg=cfg.mass_kg + cfg.payload_kg,
            max_horizontal_speed=cfg.max_horizontal_speed,
            max_climb_rate=cfg.max_climb_rate,
            max_descent_rate=cfg.max_descent_rate,
            max_acceleration=cfg.max_acceleration,
            max_tilt_rad=math.radians(cfg.max_tilt_deg),
            thrust_to_weight=cfg.thrust_to_weight,
            thrust_time_constant=cfg.thrust_time_constant,
            drag_coefficient=cfg.drag_coefficient,
            max_yaw_rate=math.radians(cfg.max_yaw_rate_deg),
            yaw_gain=cfg.yaw_gain,
            velocity_kp=cfg.velocity_kp,
            velocity_ki=cfg.velocity_ki,
            velocity_integral_limit=cfg.velocity_integral_limit,
        )

    @property
    def max_horizontal_accel(self) -> float:
        """Horizontal acceleration available at the tilt limit while holding altitude."""
        return min(self.max_acceleration, GRAVITY * math.tan(self.max_tilt_rad))


@dataclass(slots=True)
class RigidBodyState:
    """Kinematic state of one vehicle in the ENU frame."""

    position: Vector3 = field(default_factory=lambda: np.zeros(3))
    velocity: Vector3 = field(default_factory=lambda: np.zeros(3))
    acceleration: Vector3 = field(default_factory=lambda: np.zeros(3))
    specific_thrust: Vector3 = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0            # rad, positive = right wing down
    pitch: float = 0.0           # rad, positive = nose up
    yaw: float = 0.0             # rad, ENU: counter-clockwise from East
    yaw_rate: float = 0.0        # rad/s
    on_ground: bool = True
    last_touchdown_speed: float = 0.0


def clip_velocity(v: Vector3, max_h: float, max_climb: float, max_descent: float) -> Vector3:
    """Clip a velocity setpoint to the horizontal-speed and climb/descent envelope."""
    out = np.array(v, dtype=np.float64)
    h = math.hypot(out[0], out[1])
    if h > max_h:
        out[:2] *= max_h / h
    out[2] = min(max(out[2], -max_descent), max_climb)
    return out


class VelocityController:
    """PI velocity controller with drag feed-forward and anti-windup.

    ``a_des = Kp * e + Ki * integral(e) + c_d * v_sp`` with ``e = v_sp - v``.

    * The feed-forward term cancels still-air drag at the commanded speed, so the
      integrator only has to reject disturbances (wind, model error).
    * Conditional integration: the integrator is frozen on any axis whose
      output is saturated. This prevents windup during large manoeuvres.
    """

    def __init__(self, params: DynamicsParams) -> None:
        self.p = params
        self._integral = [0.0, 0.0, 0.0]
        self._a_h_max = params.max_horizontal_accel

    def reset(self) -> None:
        self._integral = [0.0, 0.0, 0.0]

    def compute(self, v_sp: Vector3, v: Vector3, dt: float) -> Vector3:
        # Scalar maths: NumPy on 3-vectors costs ~1 us per operation, which dominated the physics step.
        p = self.p
        lim, kp, ki, cd = p.velocity_integral_limit, p.velocity_kp, p.velocity_ki, p.drag_coefficient
        integ = self._integral
        err = [float(v_sp[0]) - float(v[0]), float(v_sp[1]) - float(v[1]), float(v_sp[2]) - float(v[2])]
        cand = [min(max(integ[k] + err[k] * dt, -lim), lim) for k in range(3)]
        a = [kp * err[k] + ki * cand[k] + cd * float(v_sp[k]) for k in range(3)]
        lx, ly, lz = self._limit3(a[0], a[1], a[2])
        limited = (lx, ly, lz)
        # Integrate only on axes that are not saturated (or where the error unwinds it).
        for k in range(3):
            saturated = abs(limited[k] - a[k]) > 1e-9
            unwinding = _sign(err[k]) != _sign(integ[k])
            if not saturated or unwinding:
                integ[k] = cand[k]
        return np.array(limited)

    def _limit3(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        h = math.hypot(x, y)
        if h > self._a_h_max:
            s = self._a_h_max / h
            x, y = x * s, y * s
        m = self.p.max_acceleration
        return x, y, min(max(z, -m), m)

    def _limit(self, a: Vector3) -> Vector3:
        return np.array(self._limit3(float(a[0]), float(a[1]), float(a[2])))


class QuadrotorDynamics:
    """Integrates :class:`RigidBodyState` forward in time."""

    def __init__(self, params: DynamicsParams) -> None:
        self.p = params
        self._tan_tilt = math.tan(params.max_tilt_rad)
        self._f_max = params.thrust_to_weight * GRAVITY
        self._alpha_cache: tuple[float, float] = (-1.0, 0.0)     # (dt, 1 - exp(-dt / tau))

    def set_thrust_scale(self, scale: float) -> None:
        """Scale the available thrust (a partial motor failure leaves ``scale`` < 1)."""
        self._f_max = self.p.thrust_to_weight * GRAVITY * scale

    def _limit3(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        f_max = self._f_max
        z = min(max(z, 0.0), f_max)
        # Tilt limit: the angle between thrust and vertical is at most max_tilt.
        h = math.hypot(x, y)
        h_max = z * self._tan_tilt
        if h > h_max:
            s = h_max / h if h > 0 else 0.0
            x, y = x * s, y * s
        # Total thrust limit.
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > f_max:
            s = f_max / norm
            x, y, z = x * s, y * s, z * s
        return x, y, z

    def _limit_thrust(self, f: Vector3) -> Vector3:
        return np.array(self._limit3(float(f[0]), float(f[1]), float(f[2])))

    def step(
        self,
        state: RigidBodyState,
        accel_cmd: Vector3,
        yaw_setpoint: float | None,
        wind: Vector3,
        dt: float,
        *,
        motors_on: bool,
        ground_z: float,
    ) -> None:
        # Scalar maths throughout (same formulas as the module docstring); arrays only at the boundary.
        p = self.p
        fx, fy, fz = state.specific_thrust.tolist()
        if motors_on:
            dx, dy, dz = self._limit3(float(accel_cmd[0]), float(accel_cmd[1]), float(accel_cmd[2]) + GRAVITY)
            if self._alpha_cache[0] != dt:
                self._alpha_cache = (dt, 1.0 - math.exp(-dt / p.thrust_time_constant))
            alpha = self._alpha_cache[1]
            fx, fy, fz = fx + (dx - fx) * alpha, fy + (dy - fy) * alpha, fz + (dz - fz) * alpha
        else:
            fx = fy = fz = 0.0
        state.specific_thrust = np.array((fx, fy, fz))

        vx, vy, vz = state.velocity.tolist()
        cd = p.drag_coefficient
        ax = fx - cd * (vx - float(wind[0]))
        ay = fy - cd * (vy - float(wind[1]))
        az = fz - cd * (vz - float(wind[2])) - GRAVITY

        if state.on_ground and az <= 0.0:
            # Resting on the ground: the normal force cancels gravity and friction
            # stops sliding, so the vehicle is stationary until thrust exceeds weight.
            nvx = nvy = nvz = 0.0
        else:
            nvx, nvy, nvz = vx + ax * dt, vy + ay * dt, vz + az * dt
        px, py, pz = state.position.tolist()
        px, py, pz = px + nvx * dt, py + nvy * dt, pz + nvz * dt

        if pz <= ground_z:
            if not state.on_ground:
                state.last_touchdown_speed = -nvz
            pz = ground_z
            nvx = nvy = nvz = 0.0
            state.on_ground = True
        else:
            state.on_ground = False

        state.acceleration = np.array(((nvx - vx) / dt, (nvy - vy) / dt, (nvz - vz) / dt))
        state.velocity = np.array((nvx, nvy, nvz))
        state.position = np.array((px, py, pz))
        self._update_attitude(state, yaw_setpoint, dt, motors_on)

    def _update_attitude(self, state: RigidBodyState, yaw_sp: float | None, dt: float, motors_on: bool) -> None:
        p = self.p
        if motors_on and not state.on_ground and yaw_sp is not None:
            err = wrap_angle(yaw_sp - state.yaw)
            rate = min(max(p.yaw_gain * err, -p.max_yaw_rate), p.max_yaw_rate)
            state.yaw_rate = rate
            state.yaw = wrap_angle(state.yaw + rate * dt)
        else:
            state.yaw_rate = 0.0

        fx, fy, fz = state.specific_thrust.tolist()
        if not motors_on or fz <= 1e-6 or state.on_ground:
            state.roll = state.pitch = 0.0
            return
        c, s = math.cos(state.yaw), math.sin(state.yaw)
        f_fwd = fx * c + fy * s
        f_right = fx * s - fy * c
        state.pitch = -math.atan2(f_fwd, fz)
        state.roll = math.atan2(f_right, math.hypot(f_fwd, fz))
