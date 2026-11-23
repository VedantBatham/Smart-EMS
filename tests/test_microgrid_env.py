"""
Unit tests for the Mathematical Microgrid Simulation Environment (MicrogridEnv)
and pure mathematical RL training environment.
"""

import pytest
import numpy as np
import pandas as pd
from environment.microgrid_env import MicrogridEnv
from environment.smart_energy_env import SmartEnergyEnv


def test_solar_power_mathematical_model():
    env = MicrogridEnv(seed=42)

    # 1. Daylight window: strictly 0.0 at night (before 06:00 and after 18:00)
    assert env._solar_power(0.0) == 0.0
    assert env._solar_power(5.9) == 0.0
    assert env._solar_power(18.0) == 0.0
    assert env._solar_power(23.5) == 0.0

    # 2. Solar noon (12:00 PM): base 700 W with ~10% Gaussian noise
    noon_vals = [env._solar_power(12.0) for _ in range(50)]
    mean_noon = np.mean(noon_vals)
    assert 600.0 <= mean_noon <= 800.0

    # 3. Gaussian shape: 12:00 > 09:00 > 06:30
    p_noon = 700.0 * np.exp(-0.5 * ((12.0 - 12.0) / 3.0) ** 2)
    p_9am = 700.0 * np.exp(-0.5 * ((9.0 - 12.0) / 3.0) ** 2)
    assert p_noon > p_9am > 0.0


def test_campus_load_mathematical_model():
    env = MicrogridEnv(seed=42)

    # 1. Base load is ~280 W late at night (03:00)
    night_loads = [env._campus_load(3.0, day_of_week=1) for _ in range(30)]
    assert 240.0 <= np.mean(night_loads) <= 320.0

    # 2. Peaks at 09:00 (+380 W), 13:00 (+180 W), 19:00 (+340 W)
    load_9am = np.mean([env._campus_load(9.0, day_of_week=1) for _ in range(30)])
    load_13pm = np.mean([env._campus_load(13.0, day_of_week=1) for _ in range(30)])
    load_19pm = np.mean([env._campus_load(19.0, day_of_week=1) for _ in range(30)])

    assert load_9am > 550.0   # 280 + 380 = 660 W nominal
    assert load_13pm > 400.0  # 280 + 180 = 460 W nominal
    assert load_19pm > 550.0  # 280 + 340 = 620 W nominal

    # 3. Weekend scaling: 55% reduction on Saturday (day 5) and Sunday (day 6)
    weekday_loads = [env._campus_load(9.0, day_of_week=2) for _ in range(50)]
    weekend_loads = [env._campus_load(9.0, day_of_week=6) for _ in range(50)]
    ratio = np.mean(weekend_loads) / np.mean(weekday_loads)
    assert 0.45 <= ratio <= 0.65  # approx 0.55


def test_grid_tou_tariff_mathematical_model():
    env = MicrogridEnv()

    # Peak hours: 17:00 to 22:00 -> ₹0.30/kWh
    assert env._grid_price(17.0) == 0.30
    assert env._grid_price(19.5) == 0.30
    assert env._grid_price(21.9) == 0.30

    # Off-peak hours: all other times -> ₹0.08/kWh
    assert env._grid_price(0.0) == 0.08
    assert env._grid_price(6.0) == 0.08
    assert env._grid_price(12.0) == 0.08
    assert env._grid_price(16.9) == 0.08
    assert env._grid_price(22.0) == 0.08


def test_microgrid_env_procedural_generation():
    env = MicrogridEnv(steps_per_day=96, time_step_hours=0.25, seed=42)
    df = env.generate_day(day_of_week=2)

    assert len(df) == 96
    assert "solar_kw" in df.columns
    assert "load_kw" in df.columns
    assert "tariff" in df.columns
    assert "future_solar_1h_avg" in df.columns
    assert "future_load_1h_avg" in df.columns

    # Peak solar in kW should be around 0.6 - 0.8 kW
    assert 0.5 <= df["solar_kw"].max() <= 1.0
    # Peak load in kW should be around 0.5 - 0.8 kW
    assert 0.4 <= df["load_kw"].max() <= 1.0


def test_microgrid_env_gym_interface():
    env = MicrogridEnv(steps_per_day=96, time_step_hours=0.25, seed=42)
    obs, info = env.reset(options={"initial_soc": 0.50})

    assert obs.shape == (12,)
    assert info["initial_soc"] == 0.50
    assert 0.20 <= env.soc <= 0.90

    # Discharge action: a = 1.0
    action = np.array([1.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert next_obs.shape == (12,)
    assert isinstance(reward, float)
    assert not terminated
    assert step_info["step"] == 1

    # Step through entire day
    done = False
    step_count = 1
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.0], dtype=np.float32))
        done = term or trunc
        step_count += 1

    assert step_count == 96
    assert 0.20 <= env.soc <= 0.90


def test_smart_energy_env_pure_mathematical_mode():
    # Test initialization with data_df=None
    env = SmartEnergyEnv(data_df=None, use_mathematical_simulation=True, episode_steps=96)
    obs, info = env.reset(options={"initial_soc": 0.60})

    assert env.use_mathematical_simulation is True
    assert obs.shape == (12,)
    assert info["initial_soc"] == 0.60

    # Step in environment
    next_obs, reward, term, trunc, s_info = env.step(np.array([0.5], dtype=np.float32))
    assert next_obs.shape == (12,)
    assert isinstance(reward, float)
    assert s_info["step"] == 1
