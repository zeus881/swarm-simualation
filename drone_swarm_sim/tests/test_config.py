import pytest

from simulation.config import ConfigError, config_from_dict, load_config, parse_overrides


def test_default_yaml_loads_and_validates():
    cfg = load_config()
    assert cfg.simulation.drone_count == 10
    assert cfg.simulation.simulation_rate == 30
    assert cfg.communication.range == 1200 and cfg.communication.enabled
    assert cfg.swarm.separation_distance < cfg.swarm.warning_distance


def test_overrides_are_typed_and_take_precedence():
    cfg = load_config(overrides=["simulation.drone_count=25", "wind.enabled=false", "simulation.seed=null"])
    assert cfg.simulation.drone_count == 25
    assert cfg.wind.enabled is False
    assert cfg.simulation.seed is None


def test_int_is_coerced_to_float():
    cfg = config_from_dict({"wind": {"speed": 7}})
    assert isinstance(cfg.wind.speed, float)


def test_unknown_key_in_known_section_is_an_error():
    with pytest.raises(ConfigError, match="unknown configuration key"):
        config_from_dict({"simulation": {"drone_cnt": 5}})


def test_unknown_section_is_preserved_as_extension():
    cfg = config_from_dict({"formation_library": {"custom": []}})
    assert "formation_library" in cfg.extensions


@pytest.mark.parametrize("data", [
    {"simulation": {"drone_count": "ten"}},
    {"simulation": {"drone_count": -1}},
    {"battery": {"warning": 10, "return_home": 20, "emergency": 5}},
    {"swarm": {"separation_distance": 12, "warning_distance": 10}},
    {"drone": {"emergency_stop_behavior": "explode"}},
    {"wind": {"enabled": "yes"}},
])
def test_invalid_values_rejected(data):
    with pytest.raises(ConfigError):
        config_from_dict(data)


def test_bad_override_syntax():
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])
    with pytest.raises(ConfigError):
        load_config(overrides={"toplevel": 1})


def test_missing_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")
