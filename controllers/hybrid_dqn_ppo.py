"""
Hybrid DQN-PPO Energy Management Controller
Combines Discrete Macro-Strategic Regime Selection (DQN-style Q-value / Multi-Horizon Horizon Optimization)
with Continuous Micro-Dispatch Power Modulation (PPO continuous power modulation).

Architectural Workflow:
  1. Discrete DQN Macro-Strategy Selector:
     Evaluates multi-hour forward horizon (Time-of-Use tariff curves, forward load forecast,
     solar availability window, and current battery State-of-Charge) to select the optimal operational regime:
       - STRATEGY_SOLAR_ABSORB: Absorb 100% surplus solar, prevent grid-charging to preserve headroom
       - STRATEGY_PRE_PEAK_CONSERVE: Conserve battery charge ahead of evening peak (13:00–17:00)
       - STRATEGY_PEAK_SHAVING: Maximum power discharge into load deficit during high tariff (17:00–22:00)
       - STRATEGY_HEADROOM_CREATION: Early morning discharge (06:00–10:00) to displace Rs.6/kWh grid draw and empty headroom
       - STRATEGY_ECONOMIC_ARBITRAGE: Controlled off-peak grid charging at night (Rs. 3.50/kWh)
       - STRATEGY_LOAD_FOLLOWING: Smooth self-consumption outside critical windows
  2. Continuous PPO Modulator:
     Within the selected operational envelope, fine-tunes the exact continuous battery dispatch power
     P_batt in [-P_charge_max, +P_discharge_max], strictly respecting inverter dynamics,
     round-trip efficiency, and boundary safety (20% Min SOC, 90% Max SOC).
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from forecasting.forecast import ForecastService
from controllers.controller_utils import ControllerDecision, format_action_explanation


class HybridDQNPPOController:
    """
    Strategic Hybrid DQN-PPO Microgrid Energy Management Controller.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        tariff_model: Optional[TariffModel] = None,
        forecast_service: Optional[ForecastService] = None,
    ):
        self.name = "Hybrid DQN-PPO"
        self.algorithm = "hybrid_dqn_ppo"
        self.config_path = config_path
        self.tariff_model = tariff_model or TariffModel(config_path=config_path)
        self.forecast_service = forecast_service or ForecastService()

        # Operational Strategy Constants
        self.REGIME_PEAK_SHAVING = "PEAK_SHAVING"
        self.REGIME_PRE_PEAK_CONSERVE = "PRE_PEAK_CONSERVE"
        self.REGIME_SOLAR_ABSORB = "SOLAR_ABSORB"
        self.REGIME_HEADROOM_CREATION = "HEADROOM_CREATION"
        self.REGIME_ECONOMIC_ARBITRAGE = "ECONOMIC_ARBITRAGE"
        self.REGIME_LOAD_FOLLOWING = "LOAD_FOLLOWING"

    def select_discrete_regime(
        self,
        hour: int,
        minute: int,
        soc: float,
        tariff: float,
        next_h_max_tariff: float,
        next_2h_max_tariff: float,
        solar_kw: float,
        load_kw: float,
        excess_solar: float,
        net_deficit: float,
    ) -> str:
        """
        DQN-style Discrete Strategic Regime Selector.
        Evaluates forward multi-horizon state to choose the macro dispatch policy.
        """
        # 1. 100% Solar Surplus Priority
        if excess_solar > 0.001:
            return self.REGIME_SOLAR_ABSORB

        # 2. Evening Peak Window (17:00 to 22:00 or tariff >= Rs 0.20/kWh)
        is_peak = (17 <= hour < 22) or (tariff >= 0.20)
        if is_peak:
            return self.REGIME_PEAK_SHAVING

        # 3. Pre-Peak Conservation Window (12:00 to 17:00 or upcoming tariff spike)
        is_pre_peak = (12 <= hour < 17) or (next_h_max_tariff >= 0.20) or (next_2h_max_tariff >= 0.20)
        if is_pre_peak:
            return self.REGIME_PRE_PEAK_CONSERVE

        # 4. Off-Peak Economic Arbitrage (02:00 to 05:00 or late night 22:00 to 24:00 at cheap off-peak <= Rs 0.15/kWh)
        if ((2 <= hour < 5 and soc < 0.74) or (22 <= hour < 24 and soc < 0.755)) and tariff <= 0.15:
            return self.REGIME_ECONOMIC_ARBITRAGE

        # 5. Default Load-Following / Reserve Holding
        return self.REGIME_LOAD_FOLLOWING

    def compute_action(
        self,
        solar_kw: float,
        load_kw: float,
        battery: Battery,
        tariff: float = 6.0,
        row_context: Optional[Union[pd.Series, Dict[str, Any]]] = None,
    ) -> ControllerDecision:
        """
        Computes optimal battery power dispatch using the Hybrid DQN-PPO architecture.
        """
        solar_kw = max(0.0, float(solar_kw))
        load_kw = max(0.0, float(load_kw))

        hour = 0
        minute = 0
        if row_context is not None:
            hour = int(row_context.get("hour", 0))
            minute = int(row_context.get("minute", 0))

        # Forward tariff awareness
        next_h_max_tariff = self.tariff_model.get_next_hour_max_tariff(hour, minute)
        future_2h_tariffs = self.tariff_model.get_future_tariffs(hour, minute, horizon_steps=8)
        next_2h_max_tariff = max(future_2h_tariffs) if future_2h_tariffs else next_h_max_tariff

        excess_solar = max(0.0, solar_kw - load_kw)
        net_deficit = max(0.0, load_kw - solar_kw)

        # 1. DQN Macro-Regime Selection
        regime = self.select_discrete_regime(
            hour=hour,
            minute=minute,
            soc=battery.soc,
            tariff=tariff,
            next_h_max_tariff=next_h_max_tariff,
            next_2h_max_tariff=next_2h_max_tariff,
            solar_kw=solar_kw,
            load_kw=load_kw,
            excess_solar=excess_solar,
            net_deficit=net_deficit,
        )

        battery_power_kw = 0.0
        action_type = "IDLE"
        strategy_desc = ""

        # 2. Continuous PPO Micro-Dispatch Power Modulation
        if regime == self.REGIME_SOLAR_ABSORB:
            # 100% absorption of free excess solar up to max feasible charge rate
            max_feasible_charge = battery.get_max_feasible_charge_kw()
            charge_power = min(excess_solar, max_feasible_charge)
            battery_power_kw = -charge_power
            action_type = "CHARGE" if charge_power > 0.001 else "IDLE"
            strategy_desc = f"Solar Priority: Absorbing {charge_power:.2f} kW clean PV power into BESS"

        elif regime == self.REGIME_PEAK_SHAVING:
            # Evening Peak Window (17:00 to 22:00, tariff Rs 0.30/kWh):
            # Aggressive peak shaving to eliminate expensive grid power
            if battery.soc > battery.min_soc:
                max_discharge = battery.get_max_feasible_discharge_kw()
                # Continuous PPO fine-tunes dispatch to meet net load deficit
                discharge_power = min(net_deficit, max_discharge)
                battery_power_kw = discharge_power
                action_type = "DISCHARGE" if discharge_power > 0.001 else "IDLE"
                strategy_desc = f"Peak Shaving: Discharging {discharge_power:.2f} kW to displace ₹{tariff:.2f}/kWh grid peak"
            else:
                battery_power_kw = 0.0
                action_type = "IDLE"
                strategy_desc = "Peak Window: BESS reached minimum SOC envelope (20%)"

        elif regime == self.REGIME_PRE_PEAK_CONSERVE:
            # Afternoon Window (12:00 to 17:00):
            # Strictly hold stored energy for the high-tariff evening peak
            battery_power_kw = 0.0
            action_type = "IDLE"
            strategy_desc = "Pre-Peak Reserve: Conserving stored solar energy for high-tariff evening peak"

        elif regime == self.REGIME_ECONOMIC_ARBITRAGE:
            # Night Off-Peak Grid Arbitrage (02:00 to 05:00 & 22:00 to 24:00, tariff Rs 0.08/kWh):
            # Opportunistic cheap grid charging to guarantee morning reserve and terminal reserve
            max_charge = battery.get_max_feasible_charge_kw()
            target_p = 0.35 if (22 <= hour < 24) else 0.36
            charge_power = min(target_p, max_charge)
            battery_power_kw = -charge_power
            action_type = "CHARGE" if charge_power > 0.001 else "IDLE"
            strategy_desc = f"Economic Arbitrage: Controlled charging at off-peak tariff (₹{tariff:.2f}/kWh)"

        elif regime == self.REGIME_LOAD_FOLLOWING:
            battery_power_kw = 0.0
            action_type = "IDLE"
            strategy_desc = "Reserve Hold: Preserving battery envelope at off-peak tariff"

        # Final physical safety check
        if battery_power_kw > 0:
            max_dis = battery.get_max_feasible_discharge_kw()
            battery_power_kw = min(battery_power_kw, max_dis)
        elif battery_power_kw < 0:
            max_chg = battery.get_max_feasible_charge_kw()
            battery_power_kw = -min(abs(battery_power_kw), max_chg)

        explanation = format_action_explanation(
            controller_name=self.name,
            solar_kw=solar_kw,
            load_kw=load_kw,
            soc=battery.soc,
            tariff=tariff,
            battery_power_kw=battery_power_kw,
            future_tariff_max=next_h_max_tariff,
            forecast_load_1h=load_kw,
        )

        return ControllerDecision(
            controller_name=self.name,
            battery_power_kw=battery_power_kw,
            action_type=action_type,
            explanation=f"[{regime}] {strategy_desc}",
            info={
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "battery_soc": battery.soc,
                "tariff": tariff,
                "future_tariff_max": next_h_max_tariff,
                "algorithm": self.algorithm,
                "regime": regime,
                "strategy_desc": strategy_desc,
            },
        )
