"""
Utility functions and data structures for Energy Management System controllers.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ControllerDecision:
    """Standardized decision output for an EMS controller."""
    controller_name: str
    battery_power_kw: float  # >0: discharge, <0: charge, 0: idle
    action_type: str         # "CHARGE", "DISCHARGE", "IDLE"
    explanation: str         # Deterministic human-readable explanation
    info: Dict[str, Any]


def format_action_explanation(
    controller_name: str,
    solar_kw: float,
    load_kw: float,
    soc: float,
    tariff: float,
    battery_power_kw: float,
    future_tariff_max: float = 0.0,
    forecast_load_1h: float = 0.0,
) -> str:
    """
    Generates deterministic explanation of why an action was chosen.
    Does not use LLMs.
    """
    if abs(battery_power_kw) < 1e-4:
        if controller_name == "Forecast-Assisted PPO":
            if future_tariff_max > tariff:
                return (
                    f"PPO preserved battery (idle) at Rs. {tariff:.1f}/kWh because higher tariff "
                    f"(Rs. {future_tariff_max:.1f}/kWh) and load ({forecast_load_1h:.2f} kW) are expected."
                )
            return "Battery idle: balanced current state without active dispatch."
        else:
            return "Rule-based: Battery idle because solar matches load or battery is at limits."
    elif battery_power_kw < 0.0:
        charge_p = abs(battery_power_kw)
        if solar_kw > load_kw:
            return f"Charging {charge_p:.2f} kW using surplus solar energy ({solar_kw - load_kw:.2f} kW excess)."
        else:
            return f"Charging {charge_p:.2f} kW from grid during low tariff period (Rs. {tariff:.1f}/kWh) to prepare for future peak."
    else:
        # Discharging
        if tariff >= 8.0:
            return f"Discharging {battery_power_kw:.2f} kW to offset high grid tariff (Rs. {tariff:.1f}/kWh)."
        else:
            return f"Discharging {battery_power_kw:.2f} kW to satisfy net load ({load_kw - solar_kw:.2f} kW)."
