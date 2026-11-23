"""
Scenario Generator Module for Hybrid Data + PPO Training Strategy.

Anchor RL training to real historical Open Power System Data while injecting controlled,
physically realistic stochastic variations in:
  - Solar generation scaling, cloud events, and diurnal weather patterns
  - Load scaling, morning/evening peak variations, timing shifts, and appliance spikes
  - Initial battery state-of-charge (SOC_0 ~ U(0.3, 0.8))
  - Realistic forward forecast inputs with uncertainty residuals

This eliminates fixed CSV trajectory memorization while preserving real microgrid dynamics.
Final testing remains strictly on unmodified, untouched historical data.
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List


class ScenarioGenerator:
    """
    Generates diverse, physically consistent training scenarios from historical base days.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        seed: Optional[int] = None,
    ):
        self.config_path = config_path
        self.cfg = self._load_config(config_path)
        rand_cfg = self.cfg.get("scenario_randomization", {})

        self.enabled = rand_cfg.get("enabled", True)
        self.solar_scaling_min = rand_cfg.get("solar_scaling_min", 0.70)
        self.solar_scaling_max = rand_cfg.get("solar_scaling_max", 1.10)
        self.load_scaling_min = rand_cfg.get("load_scaling_min", 0.80)
        self.load_scaling_max = rand_cfg.get("load_scaling_max", 1.20)
        self.initial_soc_min = rand_cfg.get("initial_soc_min", 0.30)
        self.initial_soc_max = rand_cfg.get("initial_soc_max", 0.80)
        self.cloud_event_prob = rand_cfg.get("cloud_event_probability", 0.35)
        self.peak_shift_max_steps = rand_cfg.get("peak_shift_max_steps", 2)
        self.demand_spike_prob = rand_cfg.get("demand_spike_probability", 0.25)
        self.fc_noise_solar = rand_cfg.get("forecast_noise_std_solar", 0.05)
        self.fc_noise_load = rand_cfg.get("forecast_noise_std_load", 0.08)

        # Convenient aliases
        self.solar_scale_min = self.solar_scaling_min
        self.solar_scale_max = self.solar_scaling_max
        self.load_scale_min = self.load_scaling_min
        self.load_scale_max = self.load_scaling_max

        self.rng = np.random.default_rng(seed)

    def _load_config(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def sample_base_day(
        self,
        data_df: pd.DataFrame,
        day_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Samples a contiguous 96-step (24-hour) day from the dataset.
        """
        n_records = len(data_df)
        total_days = n_records // 96
        if total_days < 1:
            raise ValueError(f"Dataset has only {n_records} rows, need at least 96.")

        if day_idx is None:
            day_idx = int(self.rng.integers(0, total_days))
        else:
            day_idx = int(day_idx % total_days)

        start_idx = day_idx * 96
        day_df = data_df.iloc[start_idx : start_idx + 96].copy().reset_index(drop=True)
        return day_df

    def inject_cloud_event(
        self,
        solar_arr: np.ndarray,
        hour_arr: np.ndarray,
        event_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        """
        Injects realistic weather attenuation events:
          - sunny: baseline clear day
          - temporary_cloud_event: 1-2 hour dip (30-70% drop) during midday peak
          - morning_cloud_cover: morning suppression (hours 6-11)
          - afternoon_solar_dip: afternoon convective clouds (hours 13-18)
          - passing_clouds: multiple short transient dips
          - cloudy: heavy overcast attenuation
        """
        solar_mod = solar_arr.copy()
        events = [
            "sunny",
            "temporary_cloud_event",
            "morning_cloud_cover",
            "afternoon_solar_dip",
            "passing_clouds",
            "cloudy",
        ]

        if event_type is None:
            # Sunny is common, cloud events sampled probabilistically
            if self.rng.random() > self.cloud_event_prob:
                event_type = "sunny"
            else:
                event_type = str(self.rng.choice(events[1:]))

        if event_type == "temporary_cloud_event":
            # 1 to 2 hour reduction between 11:00 and 15:00
            start_hour = self.rng.uniform(11.0, 13.5)
            duration_hours = self.rng.uniform(1.0, 2.0)
            end_hour = start_hour + duration_hours
            reduction_factor = self.rng.uniform(0.30, 0.70)  # Drop by 30-70%

            mask = (hour_arr >= start_hour) & (hour_arr <= end_hour)
            # Smooth bell-shaped reduction curve
            if np.any(mask):
                mid = (start_hour + end_hour) / 2.0
                rad = (end_hour - start_hour) / 2.0
                factor = 1.0 - (reduction_factor * np.cos(np.pi * (hour_arr[mask] - mid) / (2 * rad)) ** 2)
                solar_mod[mask] = solar_mod[mask] * np.clip(factor, 0.15, 1.0)

        elif event_type == "morning_cloud_cover":
            # Hours 6 to 11 suppressed
            mask = (hour_arr >= 6) & (hour_arr <= 11)
            suppression = self.rng.uniform(0.40, 0.65)
            solar_mod[mask] = solar_mod[mask] * suppression

        elif event_type == "afternoon_solar_dip":
            # Hours 13 to 18 suppressed
            mask = (hour_arr >= 13) & (hour_arr <= 18)
            suppression = self.rng.uniform(0.35, 0.60)
            solar_mod[mask] = solar_mod[mask] * suppression

        elif event_type == "passing_clouds":
            # 3 to 5 transient passing cloud dips of 15-30 mins
            midday_indices = np.where((hour_arr >= 9) & (hour_arr <= 16))[0]
            if len(midday_indices) > 5:
                num_dips = int(self.rng.integers(2, 5))
                dip_centers = self.rng.choice(midday_indices, size=num_dips, replace=False)
                for c in dip_centers:
                    drop = self.rng.uniform(0.35, 0.65)
                    for offset in [-1, 0, 1]:
                        idx = c + offset
                        if 0 <= idx < len(solar_mod):
                            solar_mod[idx] *= drop if offset == 0 else (drop + 0.2)

        elif event_type == "cloudy":
            # Overcast day
            overcast_factor = self.rng.uniform(0.25, 0.50)
            solar_mod = solar_mod * overcast_factor

        # Physically reasonable constraints:
        # 1. Zero solar at night (hours < 6 or hours >= 19)
        night_mask = (hour_arr < 6) | (hour_arr >= 19)
        solar_mod[night_mask] = 0.0

        # 2. Non-negative
        solar_mod = np.clip(solar_mod, 0.0, None)

        return solar_mod, event_type

    def randomize_solar(
        self,
        base_solar: np.ndarray,
        hour_arr: np.ndarray,
        weather_type: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies scaling: P_train_solar(t) = P_base_solar(t) * k_s + epsilon_s(t)
        combined with diurnal weather events.
        """
        k_s = float(self.rng.uniform(self.solar_scaling_min, self.solar_scaling_max))
        scaled_solar = base_solar * k_s

        # Inject correlated weather event
        weather_solar, event_name = self.inject_cloud_event(
            scaled_solar, hour_arr, event_type=weather_type
        )

        # Micro-variations on daytime steps
        day_mask = (hour_arr >= 6) & (hour_arr < 19) & (weather_solar > 0.02)
        noise = self.rng.normal(0.0, 0.03, size=len(weather_solar))
        final_solar = weather_solar.copy()
        final_solar[day_mask] = np.clip(final_solar[day_mask] + noise[day_mask], 0.0, None)

        # Absolute guarantee: zero nighttime solar
        final_solar[~day_mask] = 0.0

        meta = {
            "solar_scaling_k_s": round(k_s, 3),
            "weather_event": event_name,
            "base_total_solar_kwh": round(float(np.sum(base_solar) * 0.25), 2),
            "randomized_total_solar_kwh": round(float(np.sum(final_solar) * 0.25), 2),
        }

        return final_solar.astype(np.float32), meta

    def shift_load_peak(
        self,
        load_arr: np.ndarray,
        hour_arr: np.ndarray,
        shift_steps: int = 0,
    ) -> np.ndarray:
        """
        Shifts the evening peak (17:00-22:00) by +/- shift_steps (up to +/- 30 min).
        """
        if shift_steps == 0:
            return load_arr

        shifted = load_arr.copy()
        peak_mask = (hour_arr >= 17) & (hour_arr <= 22)
        peak_indices = np.where(peak_mask)[0]

        if len(peak_indices) > 4:
            sub_load = load_arr[peak_indices]
            rolled = np.roll(sub_load, shift_steps)
            # Smooth edge boundaries
            if shift_steps > 0:
                rolled[:shift_steps] = sub_load[:shift_steps]
            elif shift_steps < 0:
                rolled[shift_steps:] = sub_load[shift_steps:]
            shifted[peak_indices] = rolled

        return shifted

    def randomize_load(
        self,
        base_load: np.ndarray,
        hour_arr: np.ndarray,
        day_of_week: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Randomizes base load with:
          - Overall scaling k_l ~ U(load_scaling_min, load_scaling_max)
          - Morning peak variation (7:00 - 9:00)
          - Evening peak variation (18:00 - 22:00)
          - Peak timing shift (+/- 1-2 intervals)
          - Short-duration high-power appliance demand spikes
          - Small white noise
        """
        k_l = float(self.rng.uniform(self.load_scaling_min, self.load_scaling_max))
        load_mod = base_load.copy() * k_l

        # 1. Morning peak variation (7:00 - 9:00)
        morn_mask = (hour_arr >= 7) & (hour_arr <= 9)
        morn_factor = self.rng.uniform(0.85, 1.20)
        load_mod[morn_mask] *= morn_factor

        # 2. Evening peak variation (18:00 - 22:00)
        eve_mask = (hour_arr >= 18) & (hour_arr <= 22)
        eve_factor = self.rng.uniform(0.85, 1.25)
        load_mod[eve_mask] *= eve_factor

        # 3. Peak timing shift (+/- up to peak_shift_max_steps)
        shift_steps = int(self.rng.integers(-self.peak_shift_max_steps, self.peak_shift_max_steps + 1))
        load_mod = self.shift_load_peak(load_mod, hour_arr, shift_steps=shift_steps)

        # 4. Short-duration demand spikes (e.g. water heater, microwave, EV top-up)
        num_spikes = 0
        if self.rng.random() < self.demand_spike_prob:
            num_spikes = int(self.rng.integers(1, 3))
            awake_indices = np.where((hour_arr >= 6) & (hour_arr <= 23))[0]
            if len(awake_indices) > 0:
                spike_indices = self.rng.choice(awake_indices, size=num_spikes, replace=False)
                for s_idx in spike_indices:
                    spike_kw = self.rng.uniform(0.60, 1.40)
                    load_mod[s_idx] += spike_kw

        # 5. Base white noise
        noise = self.rng.normal(0.0, 0.025, size=len(load_mod))
        load_mod = np.clip(load_mod + noise, 0.05, None)

        meta = {
            "load_scaling_k_l": round(k_l, 3),
            "morning_factor": round(float(morn_factor), 3),
            "evening_factor": round(float(eve_factor), 3),
            "peak_shift_steps": shift_steps,
            "demand_spikes_injected": num_spikes,
            "base_total_load_kwh": round(float(np.sum(base_load) * 0.25), 2),
            "randomized_total_load_kwh": round(float(np.sum(load_mod) * 0.25), 2),
        }

        return load_mod.astype(np.float32), meta

    def randomize_initial_soc(self) -> float:
        """
        Samples starting battery SOC uniformly within safe bounds: SOC_0 ~ U(0.3, 0.8).
        """
        return float(round(self.rng.uniform(self.initial_soc_min, self.initial_soc_max), 3))

    def generate_training_episode(
        self,
        data_df: pd.DataFrame,
        base_day_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates a complete, non-identical randomized 96-step training episode:
          1. Samples a historical base day
          2. Applies solar randomization + weather events
          3. Applies load randomization + peak shifts + appliance spikes
          4. Samples random initial SOC
          5. Generates forecast features with realistic prediction error
        """
        base_df = self.sample_base_day(data_df, day_idx=base_day_idx)

        hour_arr = (base_df["hour"] + base_df["minute"] / 60.0).to_numpy(dtype=np.float32)
        base_solar = base_df["solar_kw"].to_numpy(dtype=np.float32)
        base_load = base_df["load_kw"].to_numpy(dtype=np.float32)
        day_of_week = int(base_df["day_of_week"].iloc[0]) if "day_of_week" in base_df else 0

        # Randomize solar & load
        rand_solar, solar_meta = self.randomize_solar(base_solar, hour_arr)
        rand_load, load_meta = self.randomize_load(base_load, hour_arr, day_of_week=day_of_week)

        # Randomize initial SOC
        init_soc = self.randomize_initial_soc()

        # Build realistic forward forecast inputs from randomized profiles with noise
        # 1-hour ahead (4 steps) and 2-hour ahead (8 steps) rolling estimates
        n = len(rand_solar)
        fc_solar_1h = np.zeros(n, dtype=np.float32)
        fc_load_1h = np.zeros(n, dtype=np.float32)
        fc_solar_2h = np.zeros(n, dtype=np.float32)
        fc_load_2h = np.zeros(n, dtype=np.float32)

        solar_err = self.rng.normal(0.0, self.fc_noise_solar, size=n)
        load_err = self.rng.normal(0.0, self.fc_noise_load, size=n)

        for t in range(n):
            # 1-hour window (t+1 to t+4)
            w1_end = min(n, t + 5)
            w1_solar = rand_solar[t + 1 : w1_end]
            w1_load = rand_load[t + 1 : w1_end]
            fc_solar_1h[t] = max(0.0, float(np.mean(w1_solar) if len(w1_solar) > 0 else rand_solar[t]) + solar_err[t])
            fc_load_1h[t] = max(0.05, float(np.mean(w1_load) if len(w1_load) > 0 else rand_load[t]) + load_err[t])

            # 2-hour window (t+1 to t+8)
            w2_end = min(n, t + 9)
            w2_solar = rand_solar[t + 1 : w2_end]
            w2_load = rand_load[t + 1 : w2_end]
            fc_solar_2h[t] = max(0.0, float(np.mean(w2_solar) if len(w2_solar) > 0 else rand_solar[t]) + solar_err[t] * 1.2)
            fc_load_2h[t] = max(0.05, float(np.mean(w2_load) if len(w2_load) > 0 else rand_load[t]) + load_err[t] * 1.2)

        return {
            "base_date": str(base_df["timestamp"].iloc[0])[:10] if "timestamp" in base_df else "historical_day",
            "initial_soc": init_soc,
            "solar_kw": rand_solar,
            "load_kw": rand_load,
            "base_solar_kw": base_solar,
            "base_load_kw": base_load,
            "hour": base_df["hour"].to_numpy(dtype=np.int32),
            "minute": base_df["minute"].to_numpy(dtype=np.int32),
            "future_solar_1h_avg": fc_solar_1h,
            "future_load_1h_avg": fc_load_1h,
            "future_solar_2h_avg": fc_solar_2h,
            "future_load_2h_avg": fc_load_2h,
            "future_solar_1h": fc_solar_1h,
            "future_load_1h": fc_load_1h,
            "future_solar_2h": fc_solar_2h,
            "future_load_2h": fc_load_2h,
            "metadata": {
                **solar_meta,
                **load_meta,
                "initial_soc": init_soc,
            },
        }

    def generate_comparison_sample(
        self,
        data_df: pd.DataFrame,
        day_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Returns side-by-side historical vs randomized profiles for dashboard visualization.
        """
        episode = self.generate_training_episode(data_df, base_day_idx=day_idx)
        time_labels = [f"{h:02d}:{m:02d}" for h, m in zip(episode["hour"], episode["minute"])]

        return {
            "time_labels": time_labels,
            "base_date": episode["base_date"],
            "base_solar": episode["base_solar_kw"].tolist(),
            "base_solar_kw": episode["base_solar_kw"].tolist(),
            "randomized_solar": episode["solar_kw"].tolist(),
            "randomized_solar_kw": episode["solar_kw"].tolist(),
            "base_load": episode["base_load_kw"].tolist(),
            "base_load_kw": episode["base_load_kw"].tolist(),
            "randomized_load": episode["load_kw"].tolist(),
            "randomized_load_kw": episode["load_kw"].tolist(),
            "future_solar_1h_avg": episode["future_solar_1h_avg"].tolist(),
            "future_load_1h_avg": episode["future_load_1h_avg"].tolist(),
            "initial_soc": episode["initial_soc"],
            "metadata": episode["metadata"],
        }
