"""Energy-based battery model.

Electrical power drawn by a multirotor (see docs/ARCHITECTURE.md §6.4)::

    P = P_avionics                                         disarmed
    P = P_avionics + P_idle                                armed, motors idle on ground
    P = P_avionics + P_hover * (m_tot / m) ** 1.5          momentum theory: P ~ T^1.5
        + k_v * |v - w| ** 2                               translational / parasitic
        + m_tot * g * max(v_z, 0) / eta_climb              potential-energy rate
        + k_a * m_tot * |a|                                manoeuvring

Wind enters through the air-relative speed ``|v - w|``. The terminal voltage
uses a linear open-circuit curve with ohmic sag, which is enough for
telemetry realism. State of charge (SoC) is energy-based.
"""

from __future__ import annotations

from .config import BatteryConfig
from .types import BatteryState

GRAVITY = 9.80665
POWER_TAU_S = 5.0          # smoothing time constant of the average power used for endurance predictions


class BatteryModel:
    def __init__(self, config: BatteryConfig, initial_percent: float = 100.0) -> None:
        if not 0.0 <= initial_percent <= 100.0:
            raise ValueError("initial_percent must be in [0, 100]")
        self.config = config
        self.capacity_wh = config.capacity_wh
        self.energy_wh = config.capacity_wh * initial_percent / 100.0
        self.power_w = 0.0
        self.power_avg_w = 0.0                  # exponentially smoothed power (tau POWER_TAU_S) for endurance
        self.consumed_wh = 0.0

    # ------------------------------------------------------------------ state
    @property
    def soc(self) -> float:
        """State of charge in [0, 1]."""
        return max(0.0, self.energy_wh / self.capacity_wh)

    @property
    def percent(self) -> float:
        return 100.0 * self.soc

    @property
    def open_circuit_voltage(self) -> float:
        c = self.config
        return c.cells * (c.cell_voltage_empty + (c.cell_voltage_full - c.cell_voltage_empty) * self.soc)

    @property
    def current_a(self) -> float:
        return self.power_w / max(self.open_circuit_voltage, 1e-3)

    @property
    def voltage(self) -> float:
        return max(0.0, self.open_circuit_voltage - self.current_a * self.config.internal_resistance)

    @property
    def state(self) -> BatteryState:
        pct = self.percent
        c = self.config
        if pct <= 0.0:
            return BatteryState.DEPLETED
        if pct <= c.emergency:
            return BatteryState.EMERGENCY
        if pct <= c.return_home:
            return BatteryState.RETURN_HOME
        if pct <= c.warning:
            return BatteryState.WARNING
        return BatteryState.NORMAL

    @property
    def depleted(self) -> bool:
        return self.energy_wh <= 0.0

    # ------------------------------------------------------------------ model
    def compute_power(
        self,
        *,
        armed: bool,
        motors_on: bool,
        air_speed: float = 0.0,
        climb_rate: float = 0.0,
        accel_magnitude: float = 0.0,
        mass_kg: float = 1.5,
        payload_kg: float = 0.0,
    ) -> float:
        """Instantaneous electrical power [W] for the given flight condition."""
        c = self.config
        if not armed:
            return c.avionics_power_w
        if not motors_on:
            return c.avionics_power_w + c.idle_power_w
        total_mass = mass_kg + payload_kg
        hover = c.hover_power_w * (total_mass / mass_kg) ** 1.5
        translational = c.speed_power_coeff * air_speed * air_speed
        climb = total_mass * GRAVITY * max(climb_rate, 0.0) / c.climb_efficiency
        manoeuvre = c.accel_power_coeff * total_mass * accel_magnitude
        return c.avionics_power_w + hover + translational + climb + manoeuvre

    def update(self, dt: float, **flight_condition: float | bool) -> float:
        """Integrate energy use over ``dt`` seconds. Returns the power drawn [W]."""
        self.power_w = self.compute_power(**flight_condition) * self.config.drain_multiplier
        k = min(1.0, dt / POWER_TAU_S)
        self.power_avg_w = self.power_w if self.power_avg_w == 0.0 else self.power_avg_w + (self.power_w - self.power_avg_w) * k
        used = self.power_w * dt / 3600.0
        used = min(used, max(self.energy_wh, 0.0))
        self.energy_wh -= used
        self.consumed_wh += used
        return self.power_w

    def estimated_endurance_s(self, mass_kg: float = 1.5, payload_kg: float = 0.0) -> float:
        """Remaining hover time at the current charge [s]."""
        hover = self.compute_power(armed=True, motors_on=True, mass_kg=mass_kg, payload_kg=payload_kg)
        hover *= max(self.config.drain_multiplier, 1e-9)
        return 3600.0 * self.energy_wh / hover

    def flight_time_left_s(self, flying: bool, mass_kg: float = 1.5, payload_kg: float = 0.0) -> float:
        """Predicted flight time until the emergency threshold [s].

        In flight this uses the smoothed power actually being drawn (wind, speed and climb included);
        on the ground it assumes hover.
        """
        usable = max(self.energy_wh - self.capacity_wh * self.config.emergency / 100.0, 0.0)
        if flying and self.power_avg_w > self.config.avionics_power_w:
            power = self.power_avg_w
        else:
            power = self.compute_power(armed=True, motors_on=True, mass_kg=mass_kg, payload_kg=payload_kg)
            power *= max(self.config.drain_multiplier, 1e-9)
        return 3600.0 * usable / max(power, 1e-6)
