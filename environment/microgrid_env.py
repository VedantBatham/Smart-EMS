"""
Mathematical Microgrid Reinforcement Learning Simulation Environment.

Generates realistic microgrid data purely on the fly via closed-form mathematical equations:
  1. Solar Generation (_solar_power):
     - Gaussian bell curve centered at solar noon (12:00 PM):
       base = 700 * exp(-0.5 * ((hour - 12) / 3.0) ** 2)
     - Daylight window: 06:00 to 18:00 (6 AM to 6 PM); returns 0.0 W at night.
     - Randomness: 10% Gaussian noise (rng.normal) to simulate cloud cover.
  2. Campus Load Demand (_campus_load):
     - Base load: 280 W
     - Three distinct peak curves:
       * Morning peak at 09:00: +380 W
       * Lunch peak at 13:00: +180 W
       * Evening peak at 19:00: +340 W
     - Weekend scaling: 55% load on Saturdays & Sundays (day_of_week >= 5).
     - Noise: Gaussian noise (+/- 20 W) each step.
  3. Grid Electricity Price (_grid_price):
     - Peak hours (17:00 to 22:00): ₹0.30 / kWh
     - Off-peak hours (all other times): ₹0.08 / kWh
  4. Battery State-of-Charge (SOC):
     - Capacity: 5.0 kWh
     - Round-trip efficiency: 95%
     - Updated step-by-step based on previous SOC and agent dispatch.

Eliminates static CSV/XLSX dataset dependencies and enables infinite, non-repeating procedural training.
"""

import os
import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple, Union


class MicrogridEnv(gym.Env):
    """
    Gymnasium environment that simulates a campus microgrid using pure mathematical models.
    Supports continuous action dispatch for PPO / SAC as well as discrete dispatch.
    """
    metadata = {"render_modes": ["human"]}

    # Named discrete action indices
    ACTION_IDLE = 0
    ACTION_CHARGE = 1
    ACTION_DISCHARGE = 2
    ACTION_GRID_CHARGE = 3

    ACTION_NAMES = {
        0: "IDLE",
        1: "CHARGE",
        2: "DISCHARGE",
        3: "GRID_CHARGE",
    }

    def __init__(
        self,
        steps_per_day: int = 96,
        time_step_hours: float = 0.25,
        battery_capacity_kwh: float = 5.0,
        battery_max_charge_kw: float = 1.5,
        battery_max_discharge_kw: float = 1.5,
        min_soc: float = 0.20,
        max_soc: float = 0.90,
        initial_soc: float = 0.50,
        round_trip_efficiency: float = 0.95,
        forecast_assisted: bool = True,
        discrete_actions: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.steps_per_day = steps_per_day
        self.time_step_hours = time_step_hours
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_charge_kw = battery_max_charge_kw
        self.battery_max_discharge_kw = battery_max_discharge_kw
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.default_initial_soc = initial_soc
        self.round_trip_efficiency = round_trip_efficiency
        # Symmetrical square-root conversion for charge and discharge efficiency:
        self.charge_efficiency = math.sqrt(round_trip_efficiency)
        self.discharge_efficiency = math.sqrt(round_trip_efficiency)
        self.forecast_assisted = forecast_assisted
        self.discrete_actions = discrete_actions

        self.rng = np.random.default_rng(seed)

        # Action Space
        if self.discrete_actions:
            self.action_space = spaces.Discrete(4)  # 0: IDLE, 1: CHARGE, 2: DISCHARGE, 3: GRID_CHARGE
        else:
            # Continuous action in [-1.0, 1.0]
            # a < 0: charge; a > 0: discharge; a = 0: idle
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

        # Observation Space (12-dim forecast-assisted, matching SmartEnergyEnv)
        # [norm_load, norm_solar, soc, norm_tariff, fc_load_1h, fc_solar_1h, fc_tariff_1h,
        #  sin_h, cos_h, fc_load_2h, fc_solar_2h, fc_tariff_2h]
        obs_dim = 12 if self.forecast_assisted else 5
        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Normalization scale factors (Matching reference article)
        self.norm_load_kw = 1.0   # Campus load peaks ~0.7 kW (in kW)
        self.norm_solar_kw = 1.0  # Solar peak ~0.7-0.8 kW (in kW)
        self.norm_tariff = 0.50   # Peak tariff ₹0.30/kWh

        # Reward weights
        self.weight_battery_degradation = 0.005
        self.weight_soc_violation = 20.0
        self.weight_curtailment = 2.0
        self.weight_solar_crowding = 2.0
        self.target_terminal_soc = 0.50

        # State tracking
        self.soc = initial_soc
        self.current_step = 0
        self.day_of_week = 0
        self.episode_cost = 0.0
        self.episode_throughput = 0.0
        self.episode_violations = 0

        # Pre-generated daily episode buffers
        self.ep_solar_kw = np.zeros(self.steps_per_day, dtype=np.float32)
        self.ep_load_kw = np.zeros(self.steps_per_day, dtype=np.float32)
        self.ep_tariff = np.zeros(self.steps_per_day, dtype=np.float32)
        self.ep_hour = np.zeros(self.steps_per_day, dtype=np.float32)
        self.ep_minute = np.zeros(self.steps_per_day, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Core Mathematical Simulation Models (As described in Reference Article)
    # -------------------------------------------------------------------------

    def _solar_power(self, hour: float) -> float:
        """
        Solar Generation mathematical model:
          - Gaussian bell curve centered at solar noon (12:00 PM):
              base = 700 * exp(-0.5 * ((hour - 12) / 3.0) ** 2)
          - Daylight window: 06:00 to 18:00 (returns 0.0 W at night)
          - Randomness: 10% Gaussian noise (rng.normal)
        Returns power in Watts (W).
        """
        if hour < 6.0 or hour >= 18.0:
            return 0.0
        base = 700.0 * np.exp(-0.5 * ((hour - 12.0) / 3.0) ** 2)
        noise = self.rng.normal(0.0, 0.10)
        solar = max(0.0, float(base * (1.0 + noise)))
        return solar

    def _campus_load(self, hour: float, day_of_week: int = 0) -> float:
        """
        Campus Load Demand mathematical model:
          - Base load: 280 W
          - Three distinct peak curves:
              * Morning peak at 09:00 (classes starting): +380 W
              * Lunch peak at 13:00 (lunchtime activity): +180 W
              * Evening peak at 19:00 (library / lab / hostel): +340 W
          - Weekend scaling: 55% on Saturdays and Sundays (day_of_week >= 5)
          - Noise: Gaussian noise (+/- 20 W)
        Returns power in Watts (W).
        """
        base = 280.0
        p_morning = 380.0 * np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2)
        p_lunch = 180.0 * np.exp(-0.5 * ((hour - 13.0) / 1.0) ** 2)
        p_evening = 340.0 * np.exp(-0.5 * ((hour - 19.0) / 2.0) ** 2)
        total = base + p_morning + p_lunch + p_evening
        if day_of_week >= 5:
            total *= 0.55
        noise = self.rng.normal(0.0, 20.0)
        load = max(0.0, float(total + noise))
        return load

    def _grid_price(self, hour: float) -> float:
        """
        Grid Electricity Price (TOU Tariff):
          - Peak hours (17:00 to 22:00): ₹0.30 / kWh
          - Off-peak hours (all other times): ₹0.08 / kWh
        """
        if 17.0 <= hour < 22.0:
            return 0.30
        return 0.08

    # -------------------------------------------------------------------------
    # Procedural Mathematical Episode Generation
    # -------------------------------------------------------------------------

    def generate_day(
        self,
        day_of_week: Optional[int] = None,
        date_str: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates a full daily sequence of mathematical microgrid data.
        Returns a pandas DataFrame ready for training or evaluation.
        """
        if day_of_week is None:
            day_of_week = int(self.rng.integers(0, 7))

        records = []
        steps = self.steps_per_day
        dt = self.time_step_hours

        for step in range(steps):
            total_hours = step * dt
            hour = int(total_hours) % 24
            minute = int(round((total_hours - int(total_hours)) * 60))
            frac_hour = hour + minute / 60.0

            # Compute power in Watts from mathematical equations
            solar_w = self._solar_power(frac_hour)
            load_w = self._campus_load(frac_hour, day_of_week=day_of_week)
            tariff = self._grid_price(frac_hour)

            # Convert to kW for microgrid energy management
            solar_kw = solar_w / 1000.0
            load_kw = load_w / 1000.0

            records.append({
                "step": step,
                "hour": hour,
                "minute": minute,
                "day_of_week": day_of_week,
                "solar_w": solar_w,
                "load_w": load_w,
                "solar_kw": solar_kw,
                "load_kw": load_kw,
                "tariff": tariff,
                "current_tariff": tariff,
            })

        df = pd.DataFrame(records)

        # Build rolling forward forecast values (1-hour and 2-hour ahead)
        steps_1h = max(1, int(round(1.0 / dt)))
        steps_2h = max(2, int(round(2.0 / dt)))

        df["future_load_1h_avg"] = df["load_kw"].rolling(steps_1h, min_periods=1).mean().shift(-steps_1h).bfill().fillna(df["load_kw"])
        df["future_solar_1h_avg"] = df["solar_kw"].rolling(steps_1h, min_periods=1).mean().shift(-steps_1h).bfill().fillna(df["solar_kw"])
        df["future_load_2h_avg"] = df["load_kw"].rolling(steps_2h, min_periods=1).mean().shift(-steps_2h).bfill().fillna(df["load_kw"])
        df["future_solar_2h_avg"] = df["solar_kw"].rolling(steps_2h, min_periods=1).mean().shift(-steps_2h).bfill().fillna(df["solar_kw"])
        df["next_hour_max_tariff"] = df["tariff"].rolling(steps_1h, min_periods=1).max().shift(-steps_1h).bfill().fillna(df["tariff"])
        df["next_2h_max_tariff"] = df["tariff"].rolling(steps_2h, min_periods=1).max().shift(-steps_2h).bfill().fillna(df["tariff"])

        if date_str:
            timestamps = pd.date_range(start=date_str, periods=steps, freq=f"{int(dt*60)}min")
            df["timestamp"] = timestamps
        else:
            df["timestamp"] = pd.date_range(start="2026-01-01", periods=steps, freq=f"{int(dt*60)}min")

        return df

    # -------------------------------------------------------------------------
    # Gymnasium API: reset and step
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Procedurally generate a new non-repeating day
        self.day_of_week = int(self.rng.integers(0, 7)) if not (options and "day_of_week" in options) else int(options["day_of_week"])
        day_df = self.generate_day(day_of_week=self.day_of_week)

        self.ep_solar_kw = day_df["solar_kw"].values.astype(np.float32)
        self.ep_load_kw = day_df["load_kw"].values.astype(np.float32)
        self.ep_tariff = day_df["tariff"].values.astype(np.float32)
        self.ep_hour = day_df["hour"].values.astype(np.float32)
        self.ep_minute = day_df["minute"].values.astype(np.float32)
        self.ep_fc_load_1h = day_df["future_load_1h_avg"].values.astype(np.float32)
        self.ep_fc_solar_1h = day_df["future_solar_1h_avg"].values.astype(np.float32)
        self.ep_fc_load_2h = day_df["future_load_2h_avg"].values.astype(np.float32)
        self.ep_fc_solar_2h = day_df["future_solar_2h_avg"].values.astype(np.float32)
        self.ep_fc_tariff_1h = day_df["next_hour_max_tariff"].values.astype(np.float32)
        self.ep_fc_tariff_2h = day_df["next_2h_max_tariff"].values.astype(np.float32)

        # Initial SOC: random in [0.30, 0.80] for robust generalist policy training
        if options and "initial_soc" in options:
            self.soc = float(options["initial_soc"])
        else:
            self.soc = float(self.rng.uniform(0.30, 0.80))

        self.current_step = 0
        self.episode_cost = 0.0
        self.episode_throughput = 0.0
        self.episode_violations = 0

        obs = self._get_observation()
        info = {
            "initial_soc": self.soc,
            "day_of_week": self.day_of_week,
            "is_weekend": self.day_of_week >= 5,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_step, self.steps_per_day - 1)
        load_kw = self.ep_load_kw[idx]
        solar_kw = self.ep_solar_kw[idx]
        tariff = self.ep_tariff[idx]
        frac_hour = self.ep_hour[idx] + self.ep_minute[idx] / 60.0

        norm_load = np.clip(load_kw / self.norm_load_kw, 0.0, 1.0)
        norm_solar = np.clip(solar_kw / self.norm_solar_kw, 0.0, 1.0)
        norm_tariff = np.clip(tariff / self.norm_tariff, 0.0, 1.0)
        sin_h = np.sin(2.0 * np.pi * frac_hour / 24.0)
        cos_h = np.cos(2.0 * np.pi * frac_hour / 24.0)

        if not self.forecast_assisted:
            return np.array([norm_load, norm_solar, self.soc, norm_tariff, sin_h], dtype=np.float32)

        fc_load_1h = np.clip(self.ep_fc_load_1h[idx] / self.norm_load_kw, 0.0, 1.0)
        fc_solar_1h = np.clip(self.ep_fc_solar_1h[idx] / self.norm_solar_kw, 0.0, 1.0)
        fc_tariff_1h = np.clip(self.ep_fc_tariff_1h[idx] / self.norm_tariff, 0.0, 1.0)
        fc_load_2h = np.clip(self.ep_fc_load_2h[idx] / self.norm_load_kw, 0.0, 1.0)
        fc_solar_2h = np.clip(self.ep_fc_solar_2h[idx] / self.norm_solar_kw, 0.0, 1.0)
        fc_tariff_2h = np.clip(self.ep_fc_tariff_2h[idx] / self.norm_tariff, 0.0, 1.0)

        obs = np.array([
            norm_load,
            norm_solar,
            self.soc,
            norm_tariff,
            fc_load_1h,
            fc_solar_1h,
            fc_tariff_1h,
            sin_h,
            cos_h,
            fc_load_2h,
            fc_solar_2h,
            fc_tariff_2h,
        ], dtype=np.float32)
        return obs

    def step(self, action: Union[int, np.ndarray, float]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        idx = min(self.current_step, self.steps_per_day - 1)
        load_kw = float(self.ep_load_kw[idx])
        solar_kw = float(self.ep_solar_kw[idx])
        tariff = float(self.ep_tariff[idx])
        dt = self.time_step_hours

        # 1. Parse Action
        action_name = "IDLE"
        if self.discrete_actions or isinstance(action, (int, np.integer)):
            act_idx = int(action)
            action_name = self.ACTION_NAMES.get(act_idx, "IDLE")
            if act_idx == self.ACTION_CHARGE:
                raw_continuous = -1.0  # Max charge
            elif act_idx == self.ACTION_DISCHARGE:
                raw_continuous = 1.0   # Max discharge
            elif act_idx == self.ACTION_GRID_CHARGE:
                raw_continuous = -1.0  # Grid charge
            else:
                raw_continuous = 0.0   # Idle
        else:
            if isinstance(action, (list, np.ndarray)):
                raw_continuous = float(np.clip(action[0], -1.0, 1.0))
            else:
                raw_continuous = float(np.clip(action, -1.0, 1.0))

            if raw_continuous > 0.05:
                action_name = "DISCHARGE"
            elif raw_continuous < -0.05:
                action_name = "CHARGE"
            else:
                action_name = "IDLE"

        # 2. Solar Self-Consumption Priority
        # Solar supplies local load first
        solar_to_load = min(solar_kw, load_kw)
        net_load = load_kw - solar_to_load
        excess_solar = max(0.0, solar_kw - solar_to_load)

        # 3. Battery Constraints & Power Dispatch
        headroom_kwh = max(0.0, (self.max_soc - self.soc) * self.battery_capacity_kwh)
        stored_kwh = max(0.0, (self.soc - self.min_soc) * self.battery_capacity_kwh)

        max_charge_power = min(self.battery_max_charge_kw, headroom_kwh / (dt * self.charge_efficiency + 1e-9))
        max_discharge_power = min(self.battery_max_discharge_kw, (stored_kwh * self.discharge_efficiency) / (dt + 1e-9))

        battery_kw = 0.0
        grid_kw = 0.0
        curtailed_kw = 0.0
        violation = False

        if excess_solar > 0.0:
            # Solar charging takes priority
            solar_charge = min(excess_solar, max_charge_power)
            curtailed_kw = excess_solar - solar_charge
            battery_kw = -solar_charge
            grid_kw = net_load  # 0.0 since net_load was already satisfied
            action_name = "CHARGE"
        else:
            curtailed_kw = 0.0
            if raw_continuous > 0.0:
                # Discharge to cover net load
                desired_discharge = raw_continuous * self.battery_max_discharge_kw
                actual_discharge = min(desired_discharge, max_discharge_power, net_load)
                battery_kw = actual_discharge
                grid_kw = max(0.0, net_load - actual_discharge)
                if desired_discharge > max_discharge_power + 1e-4:
                    violation = True
            elif raw_continuous < 0.0:
                # Charge from utility grid
                desired_charge = abs(raw_continuous) * self.battery_max_charge_kw
                actual_charge = min(desired_charge, max_charge_power)
                battery_kw = -actual_charge
                grid_kw = net_load + actual_charge
                action_name = "GRID_CHARGE"
                if desired_charge > max_charge_power + 1e-4:
                    violation = True
            else:
                battery_kw = 0.0
                grid_kw = net_load

        # 4. Battery SOC Update (Hourly step, 95% round-trip efficiency)
        if battery_kw > 0.0:
            # Discharging
            delta_soc = -(battery_kw * dt) / (self.battery_capacity_kwh * self.discharge_efficiency)
        elif battery_kw < 0.0:
            # Charging
            delta_soc = (abs(battery_kw) * dt * self.charge_efficiency) / self.battery_capacity_kwh
        else:
            delta_soc = 0.0

        self.soc = float(np.clip(self.soc + delta_soc, self.min_soc, self.max_soc))

        # 5. Economics & Reward Calculation
        grid_cost = tariff * grid_kw * dt
        battery_deg = abs(battery_kw) * dt * self.weight_battery_degradation
        curtail_penalty = curtailed_kw * dt * (tariff * self.weight_curtailment)
        soc_penalty = self.weight_soc_violation * dt if violation else 0.0

        step_reward = -(grid_cost + battery_deg + curtail_penalty + soc_penalty)

        self.episode_cost += grid_cost
        self.episode_throughput += abs(battery_kw) * dt
        if violation:
            self.episode_violations += 1

        self.current_step += 1
        terminated = self.current_step >= self.steps_per_day
        truncated = False

        # Terminal SOC credit
        if terminated:
            soc_diff = self.soc - self.target_terminal_soc
            terminal_credit = (soc_diff * self.battery_capacity_kwh * 0.30)
            step_reward += terminal_credit

        next_obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "step": self.current_step,
            "hour": int(self.ep_hour[idx]),
            "minute": int(self.ep_minute[idx]),
            "solar_kw": solar_kw,
            "load_kw": load_kw,
            "battery_kw": battery_kw,
            "grid_kw": grid_kw,
            "soc": self.soc,
            "tariff": tariff,
            "action_name": action_name,
            "grid_cost": grid_cost,
            "curtailed_kw": curtailed_kw,
            "violation": violation,
            "episode_cost": self.episode_cost,
            "episode_throughput": self.episode_throughput,
            "episode_violations": self.episode_violations,
        }

        return next_obs, step_reward, terminated, truncated, info
