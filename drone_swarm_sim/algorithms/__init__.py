"""Swarm algorithms. Each behaviour is a :class:`simulation.swarm.VelocityStage` that refines the
desired velocities of the swarm in priority order:

    formation / flocking (20)  ->  collision avoidance (40: ORCA + potential-field fallback + safety filter)

so collision avoidance always has the final word (docs/ARCHITECTURE.md §3.1).
"""

from .collision_avoidance import CollisionAvoidance
from .coordinator import SwarmCoordinator, SwarmMode
from .flocking import FlockingController
from .formation import FormationController, FormationShape, assign_slots, formation_offsets, paths_cross
from .orca import linear_program3, orca_planes

__all__ = [
    "CollisionAvoidance",
    "FlockingController",
    "FormationController",
    "FormationShape",
    "SwarmCoordinator",
    "SwarmMode",
    "assign_slots",
    "formation_offsets",
    "linear_program3",
    "orca_planes",
    "paths_cross",
]
