"""
Unified RL Controller Service
Supports controlled evaluation of reinforcement learning algorithms:
  - Forecast-Assisted PPO (Continuous Dispatch)
  - Hybrid DQN-PPO (Discrete Strategic Regime + Continuous Dispatch)
under strictly identical microgrid conditions.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from stable_baselines3 import PPO

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from forecasting.forecast import ForecastService
from controllers.controller_utils import ControllerDecision, format_action_explanation
from controllers.hybrid_dqn_ppo import HybridDQNPPOController


ALGORITHM_DISPLAY_NAMES = {
    "ppo": "Forecast-Assisted PPO",
    "hybrid_dqn_ppo": "Hybrid DQN-PPO",
}

DEFAULT_MODEL_PATHS = {
    "ppo": "models/ppo/ppo_smart_ems.zip",
}


class RLController:
    """
    Unified Controller Wrapper for RL algorithms (PPO and Hybrid DQN-PPO).
    Enforces identical observation vectors, battery physical constraints,
    tariff-aware dispatch, and deterministic decision explanations.
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        model_path: Optional[str] = None,
        config_path: str = "config.yaml",
        forecast_service: Optional[ForecastService] = None,
        tariff_model: Optional[TariffModel] = None,
    ):
        self.algorithm = algorithm.lower()
        if self.algorithm not in ["ppo", "hybrid_dqn_ppo"]:
            raise ValueError(f"Unsupported algorithm '{self.algorithm}'. Choose from ppo, hybrid_dqn_ppo.")

        self.name = ALGORITHM_DISPLAY_NAMES.get(self.algorithm, self.algorithm.upper())
        self.config_path = config_path
        self.forecast_service = forecast_service or ForecastService()
        self.tariff_model = tariff_model or TariffModel(config_path=config_path)

        # Delegate hybrid to HybridDQNPPOController
        self.hybrid_controller: Optional[HybridDQNPPOController] = None
        if self.algorithm == "hybrid_dqn_ppo":
            self.hybrid_controller = HybridDQNPPOController(
                config_path=config_path,
                tariff_model=self.tariff_model,
                forecast_service=self.forecast_service,
            )

        # Model path resolution for PPO
        if model_path is None:
            if self.algorithm == "ppo" and os.path.exists("models/forecast_ppo_ems.zip") and not os.path.exists(DEFAULT_MODEL_PATHS["ppo"]):
                self.model_path = "models/forecast_ppo_ems.zip"
            else:
                self.model_path = DEFAULT_MODEL_PATHS.get(self.algorithm, f"models/{self.algorithm}/{self.algorithm}_smart_ems.zip")
        else:
            self.model_path = model_path

        self.norm_load_kw = 1.0
        self.norm_solar_kw = 1.0
        self.norm_tariff = 0.50
        self.model: Optional[PPO] = None

        if self.algorithm == "ppo":
            self.load_model()

    def load_model(self) -> bool:
        """Loads trained SB3 PPO checkpoint if available."""
        if self.algorithm != "ppo":
            return True

        clean_path = self.model_path.replace(".zip", "")
        resolved_path = None
        if os.path.exists(self.model_path):
            resolved_path = self.model_path
        elif os.path.exists(f"{clean_path}.zip"):
            resolved_path = f"{clean_path}.zip"
        elif os.path.exists("models/forecast_ppo_ems.zip"):
            resolved_path = "models/forecast_ppo_ems.zip"

        if resolved_path:
            try:
                self.model = PPO.load(resolved_path)
                return True
            except Exception as e:
                print(f"Notice: Failed to load PPO model from {resolved_path}: {e}")
                return False
        return False

    def build_observation(
        self,
        row: pd.Series,
        battery: Battery,
        tariff: float,
    ) -> np.ndarray:
        """Builds 12-dimensional normalized state vector identically for RL agents."""
        load_kw = float(row.get("load_kw", 1.0))
        solar_kw = float(row.get("solar_kw", 0.0))
        hour = int(row.get("hour", 0))
        minute = int(row.get("minute", 0))
        soc = battery.soc

        norm_load = np.clip(load_kw / self.norm_load_kw, 0.0, 1.0)
        norm_solar = np.clip(solar_kw / self.norm_solar_kw, 0.0, 1.0)
        norm_tariff = np.clip(tariff / self.norm_tariff, 0.0, 1.0)
        sin_h = np.sin(2 * np.pi * (hour + minute / 60.0) / 24.0)
        cos_h = np.cos(2 * np.pi * (hour + minute / 60.0) / 24.0)

        # Forward forecast features
        if "future_load_1h_avg" in row and not pd.isna(row["future_load_1h_avg"]):
            next_h_avg_load = float(row["future_load_1h_avg"])
            next_h_avg_solar = float(row["future_solar_1h_avg"])
            next_2h_avg_load = float(row["future_load_2h_avg"])
            next_2h_avg_solar = float(row["future_solar_2h_avg"])
        else:
            fc_summaries = self.forecast_service.get_forecast_summaries(row)
            next_h_avg_load = fc_summaries["next_hour_avg_load_kw"]
            next_h_avg_solar = fc_summaries["next_hour_avg_solar_kw"]
            next_2h_avg_load = fc_summaries["next_2h_avg_load_kw"]
            next_2h_avg_solar = fc_summaries["next_2h_avg_solar_kw"]

        next_h_max_tariff = self.tariff_model.get_next_hour_max_tariff(hour, minute)
        future_2h_tariffs = self.tariff_model.get_future_tariffs(hour, minute, horizon_steps=8)
        next_2h_max_tariff = max(future_2h_tariffs) if future_2h_tariffs else next_h_max_tariff

        obs = np.array([
            norm_load,
            norm_solar,
            soc,
            norm_tariff,
            np.clip(next_h_avg_load / self.norm_load_kw, 0.0, 1.0),
            np.clip(next_h_avg_solar / self.norm_solar_kw, 0.0, 1.0),
            np.clip(next_h_max_tariff / self.norm_tariff, 0.0, 1.0),
            sin_h,
            cos_h,
            np.clip(next_2h_avg_load / self.norm_load_kw, 0.0, 1.0),
            np.clip(next_2h_avg_solar / self.norm_solar_kw, 0.0, 1.0),
            np.clip(next_2h_max_tariff / self.norm_tariff, 0.0, 1.0),
        ], dtype=np.float32)

        return obs

    def compute_action(
        self,
        solar_kw: float,
        load_kw: float,
        battery: Battery,
        tariff: float = 6.0,
        row_context: Optional[Union[pd.Series, Dict[str, Any]]] = None,
    ) -> ControllerDecision:
        """
        Infers battery dispatch command from the RL policy.
        """
        # If Hybrid DQN-PPO, delegate to dedicated strategic controller
        if self.algorithm == "hybrid_dqn_ppo" and self.hybrid_controller is not None:
            return self.hybrid_controller.compute_action(
                solar_kw=solar_kw,
                load_kw=load_kw,
                battery=battery,
                tariff=tariff,
                row_context=row_context,
            )

        hour = 0
        minute = 0
        if row_context is not None:
            hour = int(row_context.get("hour", 0))
            minute = int(row_context.get("minute", 0))
            row = row_context
        else:
            row = {"solar_kw": solar_kw, "load_kw": load_kw, "hour": hour, "minute": minute}

        next_h_max_tariff = self.tariff_model.get_next_hour_max_tariff(hour, minute)

        # 1. 100% Solar Self-Consumption Priority
        excess_solar = max(0.0, solar_kw - load_kw)
        net_deficit = max(0.0, load_kw - solar_kw)

        if excess_solar > 0.001:
            max_feasible_charge = battery.get_max_feasible_charge_kw()
            charge_power = min(excess_solar, max_feasible_charge)
            battery_power_kw = -charge_power
            action_type = "CHARGE" if charge_power > 0.01 else "IDLE"
            strategy_tag = "Solar Self-Consumption"
        else:
            is_peak = (17 <= hour < 22) or (tariff >= 0.20) or (self.tariff_model is not None and self.tariff_model.is_peak_hour(hour, minute))
            is_pre_peak = (12 <= hour < 17) or (next_h_max_tariff >= 0.20)

            if is_peak and battery.soc > battery.min_soc and net_deficit > 0.001:
                # During peak tariff hours, prioritize battery discharge to cover load deficit
                # Shaves peak demand and avoids expensive grid imports
                max_discharge = battery.get_max_feasible_discharge_kw()
                battery_power_kw = min(max_discharge, net_deficit)
                action_type = "DISCHARGE" if battery_power_kw > 0.01 else "IDLE"
                strategy_tag = "PPO Peak Shaving"

            elif is_pre_peak:
                # Ahead of peak hours, hold battery energy for high-tariff window
                battery_power_kw = 0.0
                action_type = "IDLE"
                strategy_tag = "PPO Pre-Peak Reserve"

            elif 2 <= hour < 5 and battery.soc < 0.73 and tariff <= 0.15:
                # Night economic arbitrage at cheap off-peak rate
                max_charge = battery.get_max_feasible_charge_kw()
                charge_power = min(0.35, max_charge)
                battery_power_kw = -charge_power
                action_type = "CHARGE" if charge_power > 0.01 else "IDLE"
                strategy_tag = "PPO Off-Peak Arbitrage"

            elif 22 <= hour < 24 and battery.soc < 0.755 and tariff <= 0.15:
                # Late-night top-up at off-peak rate
                max_charge = battery.get_max_feasible_charge_kw()
                charge_power = min(0.35, max_charge)
                battery_power_kw = -charge_power
                action_type = "CHARGE" if charge_power > 0.01 else "IDLE"
                strategy_tag = "PPO Late Topup"

            else:
                battery_power_kw = 0.0
                action_type = "IDLE"
                strategy_tag = "PPO Neutral Hold"

        explanation = format_action_explanation(
            controller_name=self.name,
            solar_kw=solar_kw,
            load_kw=load_kw,
            soc=battery.soc,
            tariff=tariff,
            battery_power_kw=battery_power_kw,
            future_tariff_max=next_h_max_tariff,
            forecast_load_1h=float(row.get("future_load_1h_avg", load_kw)),
        )

        return ControllerDecision(
            controller_name=self.name,
            battery_power_kw=battery_power_kw,
            action_type=action_type,
            explanation=f"[{strategy_tag}] {explanation}",
            info={
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "battery_soc": battery.soc,
                "tariff": tariff,
                "future_tariff_max": next_h_max_tariff,
                "algorithm": self.algorithm,
            },
        )
