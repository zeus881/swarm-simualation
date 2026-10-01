from __future__ import annotations

from typing import Any

import pytest

from simulation.config import SimConfig, load_config
from simulation.engine import SimulationEngine


def make_config(**overrides: Any) -> SimConfig:
    """Default config with deterministic, disturbance-free settings for tests.

    Keyword arguments are dotted keys with '__' instead of '.', e.g. ``wind__enabled=True``.
    """
    base = {
        "wind.enabled": False,
        "simulation.seed": 1,
        "simulation.drone_count": 3,
        "logging.record_telemetry": False,
        # Stage 4 realism models are exercised by their own tests; keep the rest deterministic and flat.
        "terrain.enabled": False,
        "environment.scene_file": None,
        "communication.enabled": False,
        "sensors.enabled": False,
        "security.enabled": False,     # Stage 5 login: tests/test_stage5.py turns it on explicitly
    }
    base.update({k.replace("__", "."): v for k, v in overrides.items()})
    return load_config(overrides=base)


@pytest.fixture
def config() -> SimConfig:
    return make_config()


@pytest.fixture
def engine(config, tmp_path):
    eng = SimulationEngine(config, record=False, log_dir=tmp_path)
    yield eng
    eng.close()


def fly(engine: SimulationEngine, seconds: float) -> None:
    engine.run_for(seconds)
