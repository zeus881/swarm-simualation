import pytest

from simulation.battery import BatteryModel
from simulation.config import BatteryConfig
from simulation.types import BatteryState


def power(b, **kw):
    base = dict(armed=True, motors_on=True, air_speed=0.0, climb_rate=0.0, accel_magnitude=0.0, mass_kg=1.5)
    base.update(kw)
    return b.compute_power(**base)


def test_power_ordering():
    b = BatteryModel(BatteryConfig())
    disarmed = b.compute_power(armed=False, motors_on=False)
    idle = b.compute_power(armed=True, motors_on=False)
    hover = power(b)
    assert disarmed < idle < hover
    assert power(b, air_speed=10) > hover
    assert power(b, climb_rate=3) > hover
    assert power(b, accel_magnitude=4) > hover
    assert power(b, payload_kg=0.5) > hover


def test_hover_drain_matches_energy_model():
    cfg = BatteryConfig(capacity_wh=100.0)
    b = BatteryModel(cfg, 100.0)
    p = power(b)
    for _ in range(600):   # 60 s
        b.update(0.1, armed=True, motors_on=True, air_speed=0.0, climb_rate=0.0, accel_magnitude=0.0, mass_kg=1.5)
    expected = 100.0 - 100.0 * (p * 60 / 3600) / 100.0
    assert b.percent == pytest.approx(expected, rel=1e-6)


def test_thresholds_and_states():
    cfg = BatteryConfig()
    assert BatteryModel(cfg, 50).state == BatteryState.NORMAL
    assert BatteryModel(cfg, 30).state == BatteryState.WARNING
    assert BatteryModel(cfg, 19).state == BatteryState.RETURN_HOME
    assert BatteryModel(cfg, 9).state == BatteryState.EMERGENCY
    assert BatteryModel(cfg, 0).state == BatteryState.DEPLETED


def test_energy_never_negative_and_voltage_sags_under_load():
    b = BatteryModel(BatteryConfig(), 0.01)
    b.update(3600, armed=True, motors_on=True, mass_kg=1.5)
    assert b.energy_wh == 0 and b.depleted
    full = BatteryModel(BatteryConfig(), 100)
    ocv = full.open_circuit_voltage
    full.update(0.01, armed=True, motors_on=True, mass_kg=1.5)
    assert full.voltage < ocv
    assert ocv == pytest.approx(16.8)


def test_invalid_initial_percent():
    with pytest.raises(ValueError):
        BatteryModel(BatteryConfig(), 120)
