"""World model: bounds, terrain, obstacles, home base and wind.

Stage 4 adds a terrain height field (:mod:`simulation.terrain`) and static obstacles
(:mod:`simulation.obstacles`); the public methods used by drones (``ground_height``,
``clamp``, ``contains``, ``wind.velocity_at``) stay the same.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .config import PROJECT_ROOT, ConfigError, SimConfig, WindConfig
from .obstacles import ObstacleManager, SceneError, load_scene
from .terrain import Terrain, TerrainError
from .types import Vector3, vec3


class WindModel:
    """Mean wind with altitude shear and Ornstein-Uhlenbeck gusts.

    ``w(z, t) = s(z) * w_mean + g(t)``

    * ``w_mean = -V * [sin D, cos D, 0]`` where D is the meteorological direction,
      i.e. the direction the wind blows *from*, clockwise from North.
    * ``s(z) = clip((z / 10 m) ** (1/7), 0.3, 1.6)``: the 1/7 power law for a
      neutral atmospheric boundary layer, referenced to the standard 10 m height.
    * ``dg = -g / T dt + sigma * sqrt(2 dt / T) * N(0, 1)``: a stationary
      Gauss-Markov process with standard deviation ``sigma`` (gust strength)
      and correlation time ``T = 1 / gust_frequency``, the continuous-time
      equivalent of a first-order Dryden gust filter.
    """

    REFERENCE_HEIGHT = 10.0

    def __init__(self, config: WindConfig, rng: np.random.Generator, ground_level: float = 0.0,
                 ground: Callable[[float, float], float] | None = None) -> None:
        self.config = config
        self._rng = rng
        self._ground = ground_level
        self._ground_fn = ground                 # terrain height (shear is referenced to height above ground)
        self.enabled = config.enabled
        self.speed = config.speed
        self.direction = config.direction
        self.gust_strength = config.gust_strength
        self.gust = np.zeros(3)
        self._mean = np.zeros(3)
        self._update_mean()

    def _update_mean(self) -> None:
        d = math.radians(self.direction)
        self._mean = -self.speed * np.array((math.sin(d), math.cos(d), 0.0))

    def set_wind(self, speed: float | None = None, direction: float | None = None,
                 gust_strength: float | None = None, enabled: bool | None = None) -> None:
        if speed is not None:
            if speed < 0:
                raise ValueError("wind speed must be >= 0")
            self.speed = float(speed)
        if direction is not None:
            self.direction = float(direction) % 360.0
        if gust_strength is not None:
            if gust_strength < 0:
                raise ValueError("gust strength must be >= 0")
            self.gust_strength = float(gust_strength)
        if enabled is not None:
            self.enabled = bool(enabled)
            if not self.enabled:
                self.gust[:] = 0.0
        self._update_mean()

    def step(self, dt: float) -> None:
        if not self.enabled or self.gust_strength <= 0.0:
            self.gust[:] = 0.0
            return
        tau = 1.0 / self.config.gust_frequency
        sigma = self.gust_strength * np.array((1.0, 1.0, self.config.vertical_gust_ratio))
        noise = self._rng.standard_normal(3)
        # Exact discretisation of the OU process (stable for any dt).
        decay = math.exp(-dt / tau)
        self.gust = self.gust * decay + sigma * math.sqrt(1.0 - decay * decay) * noise

    def shear_factor(self, altitude_agl: float) -> float:
        if not self.config.shear:
            return 1.0
        h = max(altitude_agl, 0.5)
        return float(np.clip((h / self.REFERENCE_HEIGHT) ** (1.0 / 7.0), 0.3, 1.6))

    def velocity_at(self, position: Vector3) -> Vector3:
        """Wind velocity vector (ENU, m/s) at a position."""
        if not self.enabled:
            return np.zeros(3)
        ground = self._ground_fn(position[0], position[1]) if self._ground_fn is not None else self._ground
        return self.shear_factor(position[2] - ground) * self._mean + self.gust

    def velocities_at(self, positions: np.ndarray, ground: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`velocity_at` for (N, 3) positions over ground heights ``ground`` (N,)."""
        n = len(positions)
        if not self.enabled:
            return np.zeros((n, 3))
        if self.config.shear:
            h = np.maximum(positions[:, 2] - ground, 0.5)
            s = np.clip((h / self.REFERENCE_HEIGHT) ** (1.0 / 7.0), 0.3, 1.6)
        else:
            s = np.ones(n)
        return s[:, None] * self._mean[None, :] + self.gust[None, :]

    @property
    def mean_velocity(self) -> Vector3:
        return self._mean.copy() if self.enabled else np.zeros(3)

    def snapshot(self) -> dict[str, Any]:
        v = self.mean_velocity + (self.gust if self.enabled else 0.0)
        return {
            "enabled": self.enabled,
            "speed": round(self.speed if self.enabled else 0.0, 2),
            "direction": round(self.direction, 1),
            "gust_strength": round(self.gust_strength, 2),
            "velocity": {"x": round(float(v[0]), 2), "y": round(float(v[1]), 2), "z": round(float(v[2]), 2)},
        }


class Environment:
    """Static world description (bounds, terrain, obstacles) plus time-varying fields (wind)."""

    def __init__(self, config: SimConfig, rng: np.random.Generator) -> None:
        env = config.environment
        self.config = config
        self.ground_level = env.ground_level
        self.max_altitude = env.max_altitude
        self.bounds_min = np.array((-env.size_x / 2.0, -env.size_y / 2.0))
        self.bounds_max = np.array((env.size_x / 2.0, env.size_y / 2.0))
        home_xy = (config.home.position_x, config.home.position_y)
        try:
            self.terrain = Terrain.from_config(config.terrain, env.size_x, env.size_y, home_xy, PROJECT_ROOT)
        except TerrainError as exc:
            raise ConfigError(f"terrain: {exc}") from exc
        self.home_position = vec3(home_xy[0], home_xy[1], self.ground_height(*home_xy))
        self.wind = WindModel(config.wind, rng, ground_level=self.ground_level,
                              ground=self.ground_height if self.terrain is not None else None)
        path = env.scene_path
        try:
            obstacles = load_scene(path, self.ground_height) if path is not None else []
        except SceneError as exc:
            raise ConfigError(f"scene: {exc}") from exc
        self.obstacles = ObstacleManager(obstacles)
        self.scene_version = 1

    def ground_height(self, x: float, y: float) -> float:
        """Terrain height at (x, y) (flat ``ground_level`` without a terrain model)."""
        if self.terrain is None:
            return self.ground_level
        return self.ground_level + self.terrain.height(x, y)

    def ground_heights(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`ground_height`."""
        if self.terrain is None:
            return np.full(np.shape(xs), self.ground_level, dtype=np.float64)
        return self.ground_level + self.terrain.heights(xs, ys)

    def ceiling(self, x: float, y: float) -> float:
        return self.ground_height(x, y) + self.max_altitude

    def contains(self, position: Vector3) -> bool:
        x, y, z = position
        return bool(
            self.bounds_min[0] <= x <= self.bounds_max[0]
            and self.bounds_min[1] <= y <= self.bounds_max[1]
            and self.ground_height(x, y) <= z <= self.ceiling(x, y)
        )

    def clamp(self, position: Vector3, min_agl: float = 0.0) -> Vector3:
        """Clamp a position into the flyable volume (geofence)."""
        p = np.array(position, dtype=np.float64)
        p[:2] = np.clip(p[:2], self.bounds_min, self.bounds_max)
        ground = self.ground_height(p[0], p[1])
        p[2] = float(np.clip(p[2], ground + min_agl, ground + self.max_altitude))
        return p

    def step(self, dt: float) -> None:
        self.wind.step(dt)

    def describe(self) -> dict[str, Any]:
        """Static description for clients (bounds, home, limits). Obstacles and terrain are fetched
        separately from ``/api/world/scene`` when ``scene.version`` changes (they can be large)."""
        return {
            "bounds": {
                "min": {"x": float(self.bounds_min[0]), "y": float(self.bounds_min[1])},
                "max": {"x": float(self.bounds_max[0]), "y": float(self.bounds_max[1])},
            },
            "ground_level": self.ground_level,
            "max_altitude": self.max_altitude,
            "home": {"x": float(self.home_position[0]), "y": float(self.home_position[1]),
                     "z": float(self.home_position[2])},
            "scene": {"version": self.scene_version, "obstacles": len(self.obstacles),
                      "terrain": self.terrain.describe() if self.terrain is not None else {"enabled": False}},
        }

    def scene(self) -> dict[str, Any]:
        """Obstacles and a down-sampled terrain grid for rendering."""
        return {"version": self.scene_version, "obstacles": self.obstacles.to_list(),
                "terrain": self.terrain.sample_grid() if self.terrain is not None else None,
                "ground_level": self.ground_level}
