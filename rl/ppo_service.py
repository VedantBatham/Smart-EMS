"""
PPO Controller Service
Wraps trained PPO model for real-time inference, compatibility with EnergySimulator,
safety boundary enforcement, and deterministic decision explanations.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from stable_baselines3 import PPO

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from forecasting.forecast import ForecastService
from controllers.controller_utils import ControllerDecision, format_action_explanation


class PPOEMSController:
    """
    Controller wrapper for Forecast-Assisted PPO Energy Management System.
    """

    def __init__(
        self,
        model_path: str = "models/forecast_ppo_ems.zip",
        config_path: str = "config.yaml",
        forecast_service: Optional[ForecastService] = None,
        tariff_model: Optional[TariffModel] = None,
    ):
        self.model_path = model_path
        self.config_path = config_path
        self.forecast_service = forecast_service or ForecastService()
        self.tariff_model = tariff_model or TariffModel(config_path=config_path)
        self.model: Optional[PPO] = None
        self.name = "Forecast-Assisted PPO"

        self.norm_load_kw = 5.0
        self.norm_solar_kw = 4.0
        self.norm_tariff = 12.0

        self.load_model()

    def load_model(self) -> bool:
        """Loads trained SB3 PPO model if present."""
        clean_path = self.model_path.replace(".zip", "")
        if os.path.exists(self.model_path) or os.path.exists(f"{clean_path}.zip"):
            try:
                self.model = PPO.load(self.model_path)
                return True
            except Exception as e:
                print(f"Error loading PPO model: {e}")
                return False
        return False

    def build_observation(
        self,
        row: pd.Series,
        battery: Battery,
        tariff: float,
    ) -> np.ndarray:
        """Builds 12-dimensional normalized state vector."""
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

        # Extract forward forecast features (fast cache lookup if available in dataset row)
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
        tariff: float = 5.0,
        row_context: Optional[pd.Series] = None,
    ) -> ControllerDecision:
        """
        Infers battery dispatch command from trained PPO policy.
        """
        hour = 0
        minute = 0
        if row_context is not None:
            hour = int(row_context.get("hour", 0))
            minute = int(row_context.get("minute", 0))
            row = row_context
        else:
            row = pd.Series({"solar_kw": solar_kw, "load_kw": load_kw, "hour": hour, "minute": minute})

        next_h_max_tariff = self.tariff_model.get_next_hour_max_tariff(hour, minute)

        # 1. Solar Priority (Section 12)
        excess_solar = max(0.0, solar_kw - load_kw)

        if excess_solar > 0.0:
            # Solar charging has absolute physical priority
            max_feasible_charge = battery.get_max_feasible_charge_kw()
            charge_power = min(excess_solar, max_feasible_charge)
            battery_power_kw = -charge_power
            action_type = "CHARGE" if charge_power > 0.01 else "IDLE"
        elif self.model is not None:
            obs = self.build_observation(row, battery, tariff)
            action, _ = self.model.predict(obs, deterministic=True)
            raw_action = float(action[0])

            if raw_action > 0.01:
                # Discharge to supply deficit
                max_discharge = battery.get_max_feasible_discharge_kw()
                net_deficit = max(0.0, load_kw - solar_kw)
                # When PPO decides to discharge during standard or peak tariff hours,
                # supply net deficit up to max discharge rating
                if tariff >= 5.0:
                    scale = min(1.0, max(0.6, raw_action * 2.0))
                    requested_discharge = scale * max_discharge
                else:
                    requested_discharge = raw_action * battery.max_discharge_kw
                battery_power_kw = min(requested_discharge, max_discharge, net_deficit)
                action_type = "DISCHARGE" if battery_power_kw > 0.01 else "IDLE"
            elif raw_action < -0.01 and tariff <= 4.0:
                # Economically sensible off-peak grid charging
                max_charge = battery.get_max_feasible_charge_kw()
                charge_power = min(abs(raw_action) * battery.max_charge_kw, max_charge)
                battery_power_kw = -charge_power
                action_type = "CHARGE" if charge_power > 0.01 else "IDLE"
            else:
                battery_power_kw = 0.0
                action_type = "IDLE"
        else:
            battery_power_kw = 0.0
            action_type = "IDLE"

        # Fast explanation format without re-querying forecast models
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
            explanation=explanation,
            info={
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "battery_soc": battery.soc,
                "tariff": tariff,
                "future_tariff_max": next_h_max_tariff,
            },
        )
