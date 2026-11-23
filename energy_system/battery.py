"""
Battery Energy Storage System (BESS) Model
Simulates physical battery behavior with charging/discharging efficiencies,
power limits, state-of-charge (SOC) dynamics, and constraint violation tracking.
"""

from typing import Dict, Any, Tuple, Optional
import yaml
import os


class Battery:
    """
    Physical Battery Energy Storage System (BESS).
    
    Sign Convention:
      P_battery > 0 : Discharging (supplying power)
      P_battery < 0 : Charging (absorbing power)
      P_battery = 0 : Idle
    """

    def __init__(
        self,
        capacity_kwh: float = 5.0,
        initial_soc: float = 0.50,
        min_soc: float = 0.20,
        max_soc: float = 0.90,
        max_charge_kw: float = 1.5,
        max_discharge_kw: float = 1.5,
        charge_efficiency: float = 0.95,
        discharge_efficiency: float = 0.95,
        time_step_hours: float = 0.25,
    ):
        self.capacity_kwh = float(capacity_kwh)
        self.initial_soc = float(initial_soc)
        self.soc = float(initial_soc)
        self.min_soc = float(min_soc)
        self.max_soc = float(max_soc)
        self.max_charge_kw = float(max_charge_kw)
        self.max_discharge_kw = float(max_discharge_kw)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.time_step_hours = float(time_step_hours)

        # Tracking metrics
        self.total_energy_charged_kwh: float = 0.0
        self.total_energy_discharged_kwh: float = 0.0
        self.total_throughput_kwh: float = 0.0
        self.soc_violation_count: int = 0
        self.power_limit_violation_count: int = 0

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "Battery":
        """Instantiate Battery from config.yaml."""
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                b_cfg = cfg.get("battery", {})
                sim_cfg = cfg.get("simulation", {})
                return cls(
                    capacity_kwh=b_cfg.get("capacity_kwh", 5.0),
                    initial_soc=b_cfg.get("initial_soc", 0.50),
                    min_soc=b_cfg.get("min_soc", 0.20),
                    max_soc=b_cfg.get("max_soc", 0.90),
                    max_charge_kw=b_cfg.get("max_charge_kw", 1.5),
                    max_discharge_kw=b_cfg.get("max_discharge_kw", 1.5),
                    charge_efficiency=b_cfg.get("charge_efficiency", 0.95),
                    discharge_efficiency=b_cfg.get("discharge_efficiency", 0.95),
                    time_step_hours=sim_cfg.get("time_step_hours", 0.25),
                )
        return cls()

    def reset(self, initial_soc: Optional[float] = None) -> float:
        """Resets battery state for a new simulation episode."""
        if initial_soc is not None:
            self.soc = float(initial_soc)
        else:
            self.soc = float(self.initial_soc)
        self.total_energy_charged_kwh = 0.0
        self.total_energy_discharged_kwh = 0.0
        self.total_throughput_kwh = 0.0
        self.soc_violation_count = 0
        self.power_limit_violation_count = 0
        return self.soc

    def get_max_feasible_discharge_kw(self) -> float:
        """
        Calculates maximum discharge power in kW that will not violate min_soc.
        SOC_{t+1} = SOC_t - (P_discharge * dt) / (eta_d * E_batt) >= min_soc
        => P_discharge <= (SOC_t - min_soc) * eta_d * E_batt / dt
        """
        available_soc = max(0.0, self.soc - self.min_soc)
        max_power_by_energy = (available_soc * self.discharge_efficiency * self.capacity_kwh) / self.time_step_hours
        return min(self.max_discharge_kw, max_power_by_energy)

    def get_max_feasible_charge_kw(self) -> float:
        """
        Calculates maximum charge power in kW that will not violate max_soc.
        SOC_{t+1} = SOC_t + (eta_c * P_charge * dt) / E_batt <= max_soc
        => P_charge <= (max_soc - SOC_t) * E_batt / (eta_c * dt)
        """
        headroom_soc = max(0.0, self.max_soc - self.soc)
        max_power_by_energy = (headroom_soc * self.capacity_kwh) / (self.charge_efficiency * self.time_step_hours)
        return min(self.max_charge_kw, max_power_by_energy)

    def step(self, requested_power_kw: float) -> Tuple[float, float, Dict[str, Any]]:
        """
        Executes one timestep battery update.
        
        Args:
            requested_power_kw:
                > 0: discharge request (kW)
                < 0: charge request (kW, negative number)
                = 0: idle
                
        Returns:
            actual_power_kw: physically bounded battery power executed
            next_soc: new state of charge [0.0, 1.0]
            info: metadata on clipping, violations, throughput
        """
        attempted_power = requested_power_kw
        violation_occurred = False
        reason = "OK"

        # Check raw power rate bounds
        if requested_power_kw > self.max_discharge_kw:
            self.power_limit_violation_count += 1
            requested_power_kw = self.max_discharge_kw
            violation_occurred = True
            reason = "Max discharge rate exceeded"
        elif requested_power_kw < -self.max_charge_kw:
            self.power_limit_violation_count += 1
            requested_power_kw = -self.max_charge_kw
            violation_occurred = True
            reason = "Max charge rate exceeded"

        # Clip against energy/SOC capacity limits
        if requested_power_kw > 0.0:
            # Discharging
            max_feasible_discharge = self.get_max_feasible_discharge_kw()
            if requested_power_kw > max_feasible_discharge:
                self.soc_violation_count += 1
                violation_occurred = True
                reason = "Min SOC boundary reached"
                actual_power = max_feasible_discharge
            else:
                actual_power = requested_power_kw

            # Energy discharged from battery cells
            delta_soc = (actual_power * self.time_step_hours) / (self.discharge_efficiency * self.capacity_kwh)
            self.soc = max(self.min_soc, self.soc - delta_soc)

            energy_discharged = actual_power * self.time_step_hours
            self.total_energy_discharged_kwh += energy_discharged
            self.total_throughput_kwh += energy_discharged

        elif requested_power_kw < 0.0:
            # Charging (power is negative)
            charge_kw = abs(requested_power_kw)
            max_feasible_charge = self.get_max_feasible_charge_kw()
            if charge_kw > max_feasible_charge:
                self.soc_violation_count += 1
                violation_occurred = True
                reason = "Max SOC boundary reached"
                actual_charge_kw = max_feasible_charge
            else:
                actual_charge_kw = charge_kw

            actual_power = -actual_charge_kw
            # Energy added to battery cells
            delta_soc = (self.charge_efficiency * actual_charge_kw * self.time_step_hours) / self.capacity_kwh
            self.soc = min(self.max_soc, self.soc + delta_soc)

            energy_charged = actual_charge_kw * self.time_step_hours
            self.total_energy_charged_kwh += energy_charged
            self.total_throughput_kwh += energy_charged
        else:
            # Idle
            actual_power = 0.0

        # Safety clamp SOC strictly to [min_soc, max_soc]
        self.soc = float(min(self.max_soc, max(self.min_soc, round(self.soc, 6))))

        info = {
            "attempted_power_kw": attempted_power,
            "actual_power_kw": actual_power,
            "soc": self.soc,
            "violation": violation_occurred,
            "reason": reason,
            "throughput_interval_kwh": abs(actual_power) * self.time_step_hours,
            "cumulative_throughput_kwh": self.total_throughput_kwh,
        }

        return actual_power, self.soc, info
