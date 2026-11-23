"""
Gymnasium Reinforcement Learning Environment for Smart Energy Management System.
Implements the microgrid MDP:
  - Action Space: Box(-1.0, 1.0, shape=(1,)) -> Continuous Battery dispatch
  - Observation Space: Forecast-assisted normalized energy state
  - Reward: Minimizes electricity cost + throughput penalty + safety penalties + terminal SOC alignment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import yaml
import os

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.power_balance import calculate_power_balance
from forecasting.forecast import ForecastService


class SmartEnergyEnv(gym.Env):
    """
    Custom Gymnasium Environment for sequential residential energy scheduling.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_df: Optional[pd.DataFrame] = None,
        config_path: str = "config.yaml",
        forecast_assisted: bool = True,
        forecast_service: Optional[ForecastService] = None,
        episode_steps: int = 96,  # 1 day = 96 intervals of 15 min
        random_start_day: bool = True,
        scenario_generator: Optional[Any] = None,
        use_scenario_generator: bool = False,
        use_mathematical_simulation: bool = False,
    ):
        super().__init__()

        self.use_mathematical_simulation = use_mathematical_simulation or (data_df is None)
        self.config_path = config_path
        self.forecast_assisted = forecast_assisted
        self.episode_steps = episode_steps
        self.random_start_day = random_start_day
        self.scenario_generator = scenario_generator
        self.use_scenario_generator = use_scenario_generator

        # Load parameters from config
        self.cfg = self._load_config(config_path)
        b_cfg = self.cfg.get("battery", {})
        rl_cfg = self.cfg.get("rl", {})
        grid_cfg = self.cfg.get("grid", {})
        sim_cfg = self.cfg.get("simulation", {})

        time_step_hours = sim_cfg.get("time_step_hours", 0.25)
        self.battery = Battery(
            capacity_kwh=b_cfg.get("capacity_kwh", 5.0),
            initial_soc=b_cfg.get("initial_soc", 0.50),
            min_soc=b_cfg.get("min_soc", 0.20),
            max_soc=b_cfg.get("max_soc", 0.90),
            max_charge_kw=b_cfg.get("max_charge_kw", 1.5),
            max_discharge_kw=b_cfg.get("max_discharge_kw", 1.5),
            charge_efficiency=b_cfg.get("charge_efficiency", 0.95),
            discharge_efficiency=b_cfg.get("discharge_efficiency", 0.95),
            time_step_hours=time_step_hours,
        )

        from environment.microgrid_env import MicrogridEnv
        self.microgrid_env = MicrogridEnv(
            steps_per_day=episode_steps,
            time_step_hours=time_step_hours,
            battery_capacity_kwh=self.battery.capacity_kwh,
            battery_max_charge_kw=self.battery.max_charge_kw,
            battery_max_discharge_kw=self.battery.max_discharge_kw,
            min_soc=self.battery.min_soc,
            max_soc=self.battery.max_soc,
            initial_soc=self.battery.initial_soc,
            forecast_assisted=forecast_assisted,
        )

        if data_df is None or self.use_mathematical_simulation:
            self.data_df = self.microgrid_env.generate_day().reset_index(drop=True)
            self.use_mathematical_simulation = True
        else:
            self.data_df = data_df.reset_index(drop=True)

        self.tariff_model = TariffModel(config_path=config_path)
        self.forecast_service = forecast_service or ForecastService()
        self.allow_grid_export = grid_cfg.get("allow_grid_export", False)
        self.allow_grid_charging = grid_cfg.get("allow_grid_charging", True)

        # Reward penalty weights
        self.weight_battery_degradation = rl_cfg.get("weight_battery_degradation", 0.02)
        self.weight_soc_violation = rl_cfg.get("weight_soc_violation", 15.0)
        self.weight_unserved_load = rl_cfg.get("weight_unserved_load", 60.0)
        self.weight_terminal_soc = rl_cfg.get("weight_terminal_soc", 10.0)
        self.weight_solar_crowding = rl_cfg.get("weight_solar_crowding", 2.5)
        self.weight_curtailment = rl_cfg.get("weight_curtailment", 2.0)
        self.target_terminal_soc = b_cfg.get("target_terminal_soc", 0.50)

        # Normalization scale factors
        self.norm_load_kw = 5.0
        self.norm_solar_kw = 4.0
        self.norm_tariff = 12.0

        # Action Space: a in [-1.0, 1.0]
        # a = -1 -> Max charge (-1.5 kW)
        # a =  0 -> Idle (0.0 kW)
        # a = +1 -> Max discharge (+1.5 kW)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation Space
        # Forecast-Assisted (12 dims):
        # [load, solar, soc, tariff, fc_load_1h, fc_solar_1h, fc_tariff_1h_max, sin_h, cos_h, fc_load_2h, fc_solar_2h, fc_tariff_2h_max]
        # Basic (5 dims):
        # [load, solar, soc, tariff, sin_h]
        if self.forecast_assisted:
            obs_dim = 12
        else:
            obs_dim = 5

        self.observation_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Pre-extract numpy arrays for ultra-fast step execution
        self.n_records = len(self.data_df)
        self.load_arr = self.data_df["load_kw"].values.astype(np.float32)
        self.solar_arr = self.data_df["solar_kw"].values.astype(np.float32)
        self.hour_arr = self.data_df["hour"].values.astype(np.int32)
        self.minute_arr = self.data_df["minute"].values.astype(np.int32)

        if "future_load_1h_avg" in self.data_df.columns:
            self.future_load_1h_arr = self.data_df["future_load_1h_avg"].values.astype(np.float32)
            self.future_solar_1h_arr = self.data_df["future_solar_1h_avg"].values.astype(np.float32)
            self.future_load_2h_arr = self.data_df["future_load_2h_avg"].values.astype(np.float32)
            self.future_solar_2h_arr = self.data_df["future_solar_2h_avg"].values.astype(np.float32)
            self.tariff_arr = (self.data_df["current_tariff"].values.astype(np.float32) if "current_tariff" in self.data_df.columns
                               else np.array([self.tariff_model.get_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32))
            self.next_h_max_tariff_arr = (self.data_df["next_hour_max_tariff"].values.astype(np.float32) if "next_hour_max_tariff" in self.data_df.columns
                                         else np.array([self.tariff_model.get_next_hour_max_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32))
            self.next_2h_max_tariff_arr = (self.data_df["next_2h_max_tariff"].values.astype(np.float32) if "next_2h_max_tariff" in self.data_df.columns
                                          else self.next_h_max_tariff_arr)
        else:
            self.future_load_1h_arr = self.load_arr.copy()
            self.future_solar_1h_arr = self.solar_arr.copy()
            self.future_load_2h_arr = self.load_arr.copy()
            self.future_solar_2h_arr = self.solar_arr.copy()
            self.tariff_arr = np.array([self.tariff_model.get_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32)
            self.next_h_max_tariff_arr = np.array([self.tariff_model.get_next_hour_max_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32)
            self.next_2h_max_tariff_arr = self.next_h_max_tariff_arr

        # Precompute sin and cos of hour
        self.sin_h_arr = np.sin(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)
        self.cos_h_arr = np.cos(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)

        # Episode tracking variables
        self.current_step = 0
        self.start_index = 0
        self.current_index = 0
        self.episode_cost = 0.0
        self.episode_throughput = 0.0
        self.episode_violations = 0
        self.episode_steps = sim_cfg.get("steps_per_day", 96)

        # Precalculate remaining solar surplus for solar-forecast-aware reward shaping
        self.remaining_solar_surplus_arr = self._compute_remaining_solar_surplus(self.solar_arr, self.load_arr)

        # Retain references to base historical arrays for clean reset after scenario mode
        self._base_n_records = self.n_records
        self._base_load_arr = self.load_arr
        self._base_solar_arr = self.solar_arr
        self._base_hour_arr = self.hour_arr
        self._base_minute_arr = self.minute_arr
        self._base_future_load_1h_arr = self.future_load_1h_arr
        self._base_future_solar_1h_arr = self.future_solar_1h_arr
        self._base_future_load_2h_arr = self.future_load_2h_arr
        self._base_future_solar_2h_arr = self.future_solar_2h_arr
        self._base_tariff_arr = self.tariff_arr
        self._base_next_h_max_tariff_arr = self.next_h_max_tariff_arr
        self._base_next_2h_max_tariff_arr = self.next_2h_max_tariff_arr
        self._base_sin_h_arr = self.sin_h_arr
        self._base_cos_h_arr = self.cos_h_arr
        self._base_remaining_solar_surplus_arr = self.remaining_solar_surplus_arr

    def _compute_remaining_solar_surplus(self, solar_arr: np.ndarray, load_arr: np.ndarray) -> np.ndarray:
        """Calculates upcoming excess solar energy (kWh) remaining in the current daily episode."""
        dt = self.battery.time_step_hours
        excess = np.maximum(0.0, solar_arr - load_arr) * dt
        n = len(excess)
        rem = np.zeros(n, dtype=np.float32)
        chunk_size = getattr(self, "episode_steps", 96)
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            chunk = excess[start:end]
            s_sum = np.cumsum(chunk[::-1])[::-1]
            rem[start:end] = (s_sum - chunk).astype(np.float32)
        return rem

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Check if pure mathematical simulation is active
        if self.use_mathematical_simulation:
            day_of_week = options.get("day_of_week", None) if options else None
            ep_df = self.microgrid_env.generate_day(day_of_week=day_of_week)
            self.solar_arr = ep_df["solar_kw"].values.astype(np.float32)
            self.load_arr = ep_df["load_kw"].values.astype(np.float32)
            self.hour_arr = ep_df["hour"].values.astype(np.int32)
            self.minute_arr = ep_df["minute"].values.astype(np.int32)
            self.future_solar_1h_arr = ep_df["future_solar_1h_avg"].values.astype(np.float32)
            self.future_solar_2h_arr = ep_df["future_solar_2h_avg"].values.astype(np.float32)
            self.future_load_1h_arr = ep_df["future_load_1h_avg"].values.astype(np.float32)
            self.future_load_2h_arr = ep_df["future_load_2h_avg"].values.astype(np.float32)
            self.tariff_arr = ep_df["tariff"].values.astype(np.float32)
            self.next_h_max_tariff_arr = ep_df["next_hour_max_tariff"].values.astype(np.float32)
            self.next_2h_max_tariff_arr = ep_df["next_2h_max_tariff"].values.astype(np.float32)
            self.sin_h_arr = np.sin(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)
            self.cos_h_arr = np.cos(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)
            self.n_records = len(self.solar_arr)
            self.remaining_solar_surplus_arr = self._compute_remaining_solar_surplus(self.solar_arr, self.load_arr)
            self.start_index = 0
            init_soc = options.get("initial_soc", float(self.microgrid_env.rng.uniform(0.30, 0.80))) if options else float(self.microgrid_env.rng.uniform(0.30, 0.80))
        # Check if scenario randomization should be applied for this episode
        elif self.use_scenario_generator and self.scenario_generator is not None and not (options and "start_index" in options):
            ep = self.scenario_generator.generate_training_episode(self.data_df)
            self.solar_arr = ep["solar_kw"].astype(np.float32)
            self.load_arr = ep["load_kw"].astype(np.float32)
            self.hour_arr = ep["hour"].astype(np.int32)
            self.minute_arr = ep["minute"].astype(np.int32)
            self.future_solar_1h_arr = ep["future_solar_1h"].astype(np.float32)
            self.future_solar_2h_arr = ep["future_solar_2h"].astype(np.float32)
            self.future_load_1h_arr = ep["future_load_1h"].astype(np.float32)
            self.future_load_2h_arr = ep["future_load_2h"].astype(np.float32)

            self.tariff_arr = np.array([self.tariff_model.get_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32)
            self.next_h_max_tariff_arr = np.array([self.tariff_model.get_next_hour_max_tariff(h, m) for h, m in zip(self.hour_arr, self.minute_arr)], dtype=np.float32)
            self.next_2h_max_tariff_arr = self.next_h_max_tariff_arr

            self.sin_h_arr = np.sin(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)
            self.cos_h_arr = np.cos(2 * np.pi * (self.hour_arr + self.minute_arr / 60.0) / 24.0).astype(np.float32)
            self.n_records = len(self.solar_arr)
            self.remaining_solar_surplus_arr = self._compute_remaining_solar_surplus(self.solar_arr, self.load_arr)

            self.start_index = 0
            init_soc = options.get("initial_soc", ep["initial_soc"]) if options else ep["initial_soc"]
        else:
            # Revert to base historical data
            self.load_arr = self._base_load_arr
            self.solar_arr = self._base_solar_arr
            self.hour_arr = self._base_hour_arr
            self.minute_arr = self._base_minute_arr
            self.future_load_1h_arr = self._base_future_load_1h_arr
            self.future_solar_1h_arr = self._base_future_solar_1h_arr
            self.future_load_2h_arr = self._base_future_load_2h_arr
            self.future_solar_2h_arr = self._base_future_solar_2h_arr
            self.tariff_arr = self._base_tariff_arr
            self.next_h_max_tariff_arr = self._base_next_h_max_tariff_arr
            self.next_2h_max_tariff_arr = self._base_next_2h_max_tariff_arr
            self.sin_h_arr = self._base_sin_h_arr
            self.cos_h_arr = self._base_cos_h_arr
            self.n_records = self._base_n_records
            self.remaining_solar_surplus_arr = self._base_remaining_solar_surplus_arr

            if options and "start_index" in options:
                self.start_index = int(options["start_index"])
            elif self.random_start_day:
                max_start = self.n_records - self.episode_steps - 1
                if max_start > 0:
                    num_days = max_start // 96
                    day_idx = np.random.randint(0, max(1, num_days))
                    self.start_index = day_idx * 96
                else:
                    self.start_index = 0
            else:
                self.start_index = 0

            init_soc = options.get("initial_soc", self.battery.initial_soc) if options else self.battery.initial_soc

        self.current_step = 0
        self.current_index = self.start_index

        self.battery.reset(initial_soc=init_soc)

        self.episode_cost = 0.0
        self.episode_throughput = 0.0
        self.episode_violations = 0

        obs = self._get_observation()
        info = {
            "start_index": self.start_index,
            "initial_soc": self.battery.soc,
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        idx = min(self.current_index, self.n_records - 1)

        load_kw = self.load_arr[idx]
        solar_kw = self.solar_arr[idx]
        tariff = self.tariff_arr[idx]
        soc = float(self.battery.soc)

        norm_load = np.clip(load_kw / self.norm_load_kw, 0.0, 1.0)
        norm_solar = np.clip(solar_kw / self.norm_solar_kw, 0.0, 1.0)
        norm_tariff = np.clip(tariff / self.norm_tariff, 0.0, 1.0)
        sin_h = self.sin_h_arr[idx]
        cos_h = self.cos_h_arr[idx]

        if not self.forecast_assisted:
            return np.array([norm_load, norm_solar, soc, norm_tariff, sin_h], dtype=np.float32)

        next_h_avg_load = self.future_load_1h_arr[idx]
        next_h_avg_solar = self.future_solar_1h_arr[idx]
        next_2h_avg_load = self.future_load_2h_arr[idx]
        next_2h_avg_solar = self.future_solar_2h_arr[idx]
        next_h_max_tariff = self.next_h_max_tariff_arr[idx]
        next_2h_max_tariff = self.next_2h_max_tariff_arr[idx]

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

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map continuous action in [-1, 1] to battery kW
        raw_action = float(np.clip(action[0], -1.0, 1.0))

        if raw_action > 0.0:
            # Discharge request
            requested_battery_kw = raw_action * self.battery.max_discharge_kw
        elif raw_action < 0.0:
            # Charge request (negative power)
            requested_battery_kw = raw_action * self.battery.max_charge_kw
        else:
            requested_battery_kw = 0.0

        # Physical battery step
        actual_battery_kw, next_soc, b_info = self.battery.step(requested_battery_kw)

        # Environmental state from fast pre-extracted arrays
        idx = min(self.current_index, self.n_records - 1)
        load_kw = float(self.load_arr[idx])
        solar_kw = float(self.solar_arr[idx])
        tariff = float(self.tariff_arr[idx])
        dt = self.battery.time_step_hours  # 0.25 h

        # 1. Physical Solar Priority (Section 12)
        # Solar satisfies load first; excess solar is automatically available for battery storage
        solar_to_load = min(solar_kw, load_kw)
        net_load = load_kw - solar_to_load
        excess_solar = max(0.0, solar_kw - solar_to_load)

        # 2. Battery Action Interpretation:
        # a > 0: Discharge to meet net load (PPO decides how much to discharge vs preserve)
        # a < 0: Charge from grid (allowed during cheap tariff hours)
        # If excess solar exists, excess solar charges battery up to available charge headroom
        max_feasible_charge = self.battery.get_max_feasible_charge_kw()
        max_feasible_discharge = self.battery.get_max_feasible_discharge_kw()

        solar_crowding_penalty = 0.0

        if excess_solar > 0.0:
            # Solar charging takes absolute priority
            requested_battery_kw = -excess_solar
            solar_charge = min(excess_solar, max_feasible_charge)
            solar_curtailed = excess_solar - solar_charge
            actual_battery_kw, next_soc, b_info = self.battery.step(-solar_charge)
            grid_import = net_load
            grid_cost = tariff * grid_import * dt
            # Battery degradation penalty is NOT charged for absorbing free solar that would otherwise be wasted
            battery_deg_cost = 0.0
            curtailment_penalty = solar_curtailed * dt * (tariff * self.weight_curtailment)
            solar_used_kw = solar_to_load + solar_charge
        else:
            # No excess solar: PPO controls battery dispatch
            solar_curtailed = 0.0
            curtailment_penalty = 0.0
            solar_used_kw = solar_to_load
            if raw_action > 0.0:
                # Discharging
                requested_battery_kw = raw_action * self.battery.max_discharge_kw
                actual_battery_kw, next_soc, b_info = self.battery.step(requested_battery_kw)
                # Battery supplies net load
                battery_to_load = min(actual_battery_kw, net_load)
                grid_import = max(0.0, net_load - battery_to_load)
                battery_deg_cost = actual_battery_kw * dt * self.weight_battery_degradation
            elif raw_action < 0.0 and self.allow_grid_charging:
                # Charging from utility grid
                requested_battery_kw = raw_action * self.battery.max_charge_kw
                actual_battery_kw, next_soc, b_info = self.battery.step(requested_battery_kw)
                grid_charge = abs(actual_battery_kw)
                grid_import = net_load + grid_charge
                battery_deg_cost = grid_charge * dt * self.weight_battery_degradation

                # Solar-forecast-aware solar crowding penalty:
                # Penalize pre-charging from grid if upcoming daytime solar surplus exceeds available headroom
                idx_surplus = min(self.current_index, len(self.remaining_solar_surplus_arr) - 1)
                upcoming_surplus_kwh = float(self.remaining_solar_surplus_arr[idx_surplus])
                headroom_kwh = max(0.0, (self.battery.max_soc - self.battery.soc) * self.battery.capacity_kwh)
                if upcoming_surplus_kwh > 0.2 and headroom_kwh < upcoming_surplus_kwh:
                    crowding_ratio = min(1.0, (upcoming_surplus_kwh - headroom_kwh) / max(0.1, self.battery.capacity_kwh))
                    solar_crowding_penalty = grid_charge * dt * (tariff * self.weight_solar_crowding) * (0.5 + 0.5 * crowding_ratio)
            else:
                # Idle
                requested_battery_kw = 0.0
                actual_battery_kw, next_soc, b_info = self.battery.step(0.0)
                grid_import = net_load
                battery_deg_cost = 0.0

            grid_cost = tariff * grid_import * dt

        # 3. Constraint violation penalties
        soc_penalty = 0.0
        if b_info.get("violation", False):
            self.episode_violations += 1
            severity = abs(requested_battery_kw - actual_battery_kw)
            soc_penalty = self.weight_soc_violation * severity * dt

        # Step Reward: negative total cost + economic storage incentive
        step_reward = -(grid_cost + battery_deg_cost + soc_penalty + curtailment_penalty + solar_crowding_penalty)

        self.episode_cost += grid_cost
        self.episode_throughput += abs(actual_battery_kw) * dt
        self.current_step += 1
        self.current_index += 1

        terminated = self.current_step >= self.episode_steps
        truncated = self.current_index >= self.n_records

        # 4. Fair Terminal Stored Energy Valuation (Section 34)
        if terminated or truncated:
            # Reward or credit terminal stored energy difference at standard tariff rate
            soc_diff = self.battery.soc - self.target_terminal_soc
            terminal_credit = (soc_diff * self.battery.capacity_kwh * 5.0)
            step_reward += terminal_credit

        next_obs = self._get_observation() if not (terminated or truncated) else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "step": self.current_step,
            "grid_cost": grid_cost,
            "grid_kw": grid_import,
            "battery_kw": actual_battery_kw,
            "soc": next_soc,
            "solar_used_kw": solar_used_kw,
            "solar_curtailed_kw": solar_curtailed,
            "violation": b_info.get("violation", False),
            "episode_cost": self.episode_cost,
            "episode_throughput": self.episode_throughput,
            "episode_violations": self.episode_violations,
        }

        return next_obs, step_reward, terminated, truncated, info
