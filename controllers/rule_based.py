"""
Conventional Rule-Based Energy Management System (Baseline EMS)
Follows standard deterministic heuristic logic:
  - If Solar >= Load: satisfy load from solar, charge battery with surplus solar if SOC allows.
  - If Solar < Load: use solar for load, discharge battery to cover deficit if SOC allows,
    draw remaining energy from the grid.
"""

from typing import Dict, Any
from energy_system.battery import Battery
from controllers.controller_utils import ControllerDecision, format_action_explanation


class RuleBasedEMS:
    """
    Heuristic rule-based energy management controller.
    Operates reactively without forecasting or long-term price arbitrage.
    """

    def __init__(self, name: str = "Rule-Based EMS"):
        self.name = name

    def compute_action(
        self,
        solar_kw: float,
        load_kw: float,
        battery: Battery,
        tariff: float = 5.0,
    ) -> ControllerDecision:
        """
        Calculates battery power command (kW) based on conventional heuristic priority.
        
        Returns:
            ControllerDecision containing battery_power_kw:
              < 0 : charge (kW)
              > 0 : discharge (kW)
              = 0 : idle
        """
        solar_kw = max(0.0, float(solar_kw))
        load_kw = max(0.0, float(load_kw))

        if solar_kw >= load_kw:
            excess_solar = solar_kw - load_kw
            if excess_solar > 0.0 and battery.soc < battery.max_soc:
                max_charge = battery.get_max_feasible_charge_kw()
                # Charge with excess solar up to max feasible charge rate
                charge_power = min(excess_solar, max_charge)
                battery_power_kw = -charge_power
                action_type = "CHARGE" if charge_power > 0.001 else "IDLE"
            else:
                battery_power_kw = 0.0
                action_type = "IDLE"
        else:
            net_deficit = load_kw - solar_kw
            if battery.soc > battery.min_soc:
                max_discharge = battery.get_max_feasible_discharge_kw()
                # Discharge to cover deficit up to max discharge rate
                discharge_power = min(net_deficit, max_discharge)
                battery_power_kw = discharge_power
                action_type = "DISCHARGE" if discharge_power > 0.001 else "IDLE"
            else:
                battery_power_kw = 0.0
                action_type = "IDLE"

        explanation = format_action_explanation(
            controller_name=self.name,
            solar_kw=solar_kw,
            load_kw=load_kw,
            soc=battery.soc,
            tariff=tariff,
            battery_power_kw=battery_power_kw,
        )

        return ControllerDecision(
            controller_name=self.name,
            battery_power_kw=battery_power_kw,
            action_type=action_type,
            explanation=explanation,
            info={
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "battery_soc": battery.soc,
                "tariff": tariff,
            },
        )
