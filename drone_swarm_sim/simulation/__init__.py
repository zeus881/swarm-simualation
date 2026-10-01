"""Multi-UAV swarm simulation core.

Public entry points::

    from simulation import load_config, SimulationEngine
    engine = SimulationEngine(load_config())
    engine.execute({"type": "takeoff"})
    engine.run_for(10.0)
"""

from .config import SimConfig, load_config
from .drone import Drone, SimulatedDrone
from .drone_interface import CommandResult, DroneInterface, DroneTelemetry
from .engine import SimulationEngine
from .runner import RunState, SimulationRunner

__all__ = [
    "CommandResult",
    "Drone",
    "DroneInterface",
    "DroneTelemetry",
    "RunState",
    "SimConfig",
    "SimulatedDrone",
    "SimulationEngine",
    "SimulationRunner",
    "load_config",
]

__version__ = "0.1.0"
