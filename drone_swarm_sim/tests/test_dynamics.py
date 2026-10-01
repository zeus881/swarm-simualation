import math

import numpy as np

from simulation.dynamics import (
    GRAVITY, DynamicsParams, QuadrotorDynamics, RigidBodyState, VelocityController, clip_velocity, wrap_angle)
from simulation.guidance import approach_velocity

from .conftest import make_config


def _setup():
    params = DynamicsParams.from_config(make_config().drone)
    return params, QuadrotorDynamics(params), VelocityController(params)


def _airborne_state(z=20.0):
    s = RigidBodyState(position=np.array([0.0, 0.0, z]), on_ground=False)
    s.specific_thrust = np.array([0.0, 0.0, GRAVITY])
    return s


def _fly(dyn, ctrl, state, v_sp, seconds, wind=np.zeros(3), dt=1 / 60):
    for _ in range(int(seconds / dt)):
        a = ctrl.compute(v_sp, state.velocity, dt)
        dyn.step(state, a, None, wind, dt, motors_on=True, ground_z=0.0)


def test_hover_is_stationary():
    _, dyn, ctrl = _setup()
    s = _airborne_state()
    _fly(dyn, ctrl, s, np.zeros(3), 5.0)
    np.testing.assert_allclose(s.position, [0, 0, 20], atol=1e-6)


def test_velocity_tracking_converges():
    _, dyn, ctrl = _setup()
    s = _airborne_state()
    _fly(dyn, ctrl, s, np.array([8.0, 0.0, 0.0]), 8.0)
    assert abs(s.velocity[0] - 8.0) < 0.1
    assert abs(s.velocity[2]) < 0.05


def test_acceleration_and_tilt_limits_respected():
    params, dyn, ctrl = _setup()
    s = _airborne_state()
    dt = 1 / 60
    max_tilt = 0.0
    for _ in range(120):
        a = ctrl.compute(np.array([15.0, 0.0, 0.0]), s.velocity, dt)
        assert math.hypot(a[0], a[1]) <= params.max_horizontal_accel + 1e-9
        dyn.step(s, a, None, np.zeros(3), dt, motors_on=True, ground_z=0.0)
        max_tilt = max(max_tilt, abs(s.pitch), abs(s.roll))
    assert max_tilt <= params.max_tilt_rad + 1e-6


def test_integral_rejects_steady_wind():
    _, dyn, ctrl = _setup()
    s = _airborne_state()
    _fly(dyn, ctrl, s, np.zeros(3), 20.0, wind=np.array([-6.0, 0.0, 0.0]))
    assert np.linalg.norm(s.velocity) < 0.05      # holds against wind (no steady-state drift)


def test_without_motors_the_vehicle_falls_and_lands():
    _, dyn, _ = _setup()
    s = _airborne_state(z=5.0)
    for _ in range(300):
        dyn.step(s, np.zeros(3), None, np.zeros(3), 1 / 60, motors_on=False, ground_z=0.0)
    assert s.on_ground and s.position[2] == 0.0 and s.last_touchdown_speed > 5.0


def test_resting_on_ground_is_stable():
    _, dyn, _ = _setup()
    s = RigidBodyState()
    for _ in range(60):
        dyn.step(s, np.zeros(3), None, np.array([5.0, 0, 0]), 1 / 60, motors_on=False, ground_z=0.0)
    np.testing.assert_allclose(s.position, 0.0)


def test_nose_down_when_accelerating_forward():
    _, dyn, ctrl = _setup()
    s = _airborne_state()
    s.yaw = 0.0   # facing East
    _fly(dyn, ctrl, s, np.array([10.0, 0, 0]), 0.5)
    assert s.pitch < -0.1 and abs(s.roll) < 1e-3


def test_yaw_is_rate_limited():
    params, dyn, _ = _setup()
    s = _airborne_state()
    dyn.step(s, np.zeros(3), math.pi, np.zeros(3), 0.1, motors_on=True, ground_z=0.0)
    assert abs(s.yaw) <= params.max_yaw_rate * 0.1 + 1e-9


def test_wrap_and_clip_helpers():
    assert abs(wrap_angle(3 * math.pi) - math.pi) < 1e-12
    v = clip_velocity(np.array([30.0, 40.0, 10.0]), 15, 5, 3)
    assert abs(math.hypot(v[0], v[1]) - 15) < 1e-9 and v[2] == 5


def test_approach_velocity_braking_profile():
    v = approach_velocity(np.zeros(3), np.array([100.0, 0, 0]), cruise_speed=10, max_climb=5, max_descent=3,
                          decel=2, gain=1)
    assert abs(v[0] - 10) < 1e-9
    v = approach_velocity(np.zeros(3), np.array([1.0, 0, 0]), cruise_speed=10, max_climb=5, max_descent=3,
                          decel=2, gain=1)
    assert v[0] <= 1.0 + 1e-9
    # Steep climb is scaled uniformly: direction preserved.
    v = approach_velocity(np.zeros(3), np.array([10.0, 0, 100.0]), cruise_speed=10, max_climb=5, max_descent=3,
                          decel=2, gain=1)
    assert abs(v[2] - 5) < 1e-9 and abs(v[0] / v[2] - 0.1) < 1e-9
