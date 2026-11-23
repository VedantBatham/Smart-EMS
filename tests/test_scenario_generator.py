import os
import pytest
import numpy as np
import pandas as pd
from environment.scenario_generator import ScenarioGenerator
from environment.smart_energy_env import SmartEnergyEnv

CONFIG_PATH = "config.yaml"
TRAIN_DATA_PATH = "data/processed/train_data.csv"

@pytest.fixture
def train_df():
    if os.path.exists(TRAIN_DATA_PATH):
        return pd.read_csv(TRAIN_DATA_PATH)
    # Fallback dummy dataframe with 192 rows (2 days)
    records = []
    for d in range(2):
        for step in range(96):
            h = step // 4
            m = (step % 4) * 15
            s = 2.5 * max(0.0, np.sin(np.pi * (h - 6) / 12)) if 6 <= h < 18 else 0.0
            l = 1.0 + 0.5 * np.sin(np.pi * h / 12)
            records.append({
                "timestamp": f"2020-01-0{d+1} {h:02d}:{m:02d}:00",
                "load_kw": l,
                "solar_kw": s,
                "hour": h,
                "minute": m,
                "day_of_week": d,
                "future_load_1h_avg": l,
                "future_solar_1h_avg": s,
                "future_load_2h_avg": l,
                "future_solar_2h_avg": s,
            })
    return pd.DataFrame(records)

def test_scenario_generator_initialization():
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=42)
    assert sg.solar_scale_min == 0.7
    assert sg.solar_scale_max == 1.1
    assert sg.load_scale_min == 0.8
    assert sg.load_scale_max == 1.2
    assert sg.initial_soc_min == 0.3
    assert sg.initial_soc_max == 0.8

def test_randomize_solar_physical_validity(train_df):
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=123)
    base_df = sg.sample_base_day(train_df)
    hour_arr = (base_df["hour"] + base_df["minute"] / 60.0).to_numpy(dtype=np.float32)
    base_solar = base_df["solar_kw"].to_numpy(dtype=np.float32)
    
    rand_solar, meta = sg.randomize_solar(base_solar, hour_arr)
    
    # 1. Non-negative everywhere
    assert np.all(rand_solar >= 0.0)
    
    # 2. Nighttime solar strictly zero (hour < 6 or hour >= 19)
    night_mask = (hour_arr < 6.0) | (hour_arr >= 19.0)
    assert np.all(rand_solar[night_mask] == 0.0)
    
    # 3. Scaling factor in bounds
    scale = meta["solar_scaling_k_s"]
    assert 0.65 <= scale <= 1.15

def test_randomize_load_validity(train_df):
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=456)
    base_df = sg.sample_base_day(train_df)
    hour_arr = (base_df["hour"] + base_df["minute"] / 60.0).to_numpy(dtype=np.float32)
    base_load = base_df["load_kw"].to_numpy(dtype=np.float32)
    
    rand_load, meta = sg.randomize_load(base_load, hour_arr)
    
    # 1. Load is strictly positive
    assert np.all(rand_load >= 0.05)
    
    # 2. Scaling factor in bounds
    scale = meta["load_scaling_k_l"]
    assert 0.75 <= scale <= 1.25

def test_randomize_initial_soc():
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=789)
    for _ in range(50):
        soc = sg.randomize_initial_soc()
        assert 0.3 <= soc <= 0.8

def test_generate_training_episode(train_df):
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=999)
    ep = sg.generate_training_episode(train_df)
    
    assert len(ep["solar_kw"]) == 96
    assert len(ep["load_kw"]) == 96
    assert len(ep["future_solar_1h"]) == 96
    assert len(ep["future_load_1h"]) == 96
    assert 0.3 <= ep["initial_soc"] <= 0.8

def test_env_integration_with_scenario_generator(train_df):
    sg = ScenarioGenerator(config_path=CONFIG_PATH, seed=42)
    env = SmartEnergyEnv(
        data_df=train_df,
        config_path=CONFIG_PATH,
        scenario_generator=sg,
        use_scenario_generator=True
    )
    
    obs, info = env.reset()
    assert obs.shape == (12,)
    assert 0.3 <= info["initial_soc"] <= 0.8
    
    # Run 5 steps
    for _ in range(5):
        action = np.array([0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs.shape == (12,)
        assert isinstance(reward, float)
