"""
Unit tests for the SmartEnergyEnv Gymnasium environment.
"""

import numpy as np
import pandas as pd
import pytest
from environment.smart_energy_env import SmartEnergyEnv


@pytest.fixture
def sample_dataset():
    dates = pd.date_range("2020-01-01", periods=192, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "solar_kw": [1.5 if 24 <= i % 96 <= 72 else 0.0 for i in range(192)],
        "load_kw": [1.0 + 0.5 * np.sin(i / 10.0) for i in range(192)],
        "hour": [d.hour for d in dates],
        "minute": [d.minute for d in dates],
        "day_of_week": [d.dayofweek for d in dates],
        "month": [d.month for d in dates],
    })


def test_environment_initialization_forecast_assisted(sample_dataset):
    env = SmartEnergyEnv(data_df=sample_dataset, forecast_assisted=True)
    assert env.observation_space.shape == (12,)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1.0
    assert env.action_space.high[0] == 1.0


def test_environment_initialization_basic(sample_dataset):
    env = SmartEnergyEnv(data_df=sample_dataset, forecast_assisted=False)
    assert env.observation_space.shape == (5,)


def test_environment_reset_and_step(sample_dataset):
    env = SmartEnergyEnv(data_df=sample_dataset, forecast_assisted=True, random_start_day=False)
    obs, info = env.reset(options={"initial_soc": 0.60})

    assert obs.shape == (12,)
    assert info["initial_soc"] == 0.60
    assert np.all(obs >= -1.5) and np.all(obs <= 1.5)

    # Step: discharge at 50% power (a = +0.5)
    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert isinstance(reward, float)
    assert step_info["step"] == 1
    assert step_info["battery_kw"] > 0
    assert terminated is False


def test_environment_episode_completion(sample_dataset):
    env = SmartEnergyEnv(data_df=sample_dataset, episode_steps=96, random_start_day=False)
    env.reset()

    done = False
    step_count = 0
    while not done:
        action = np.array([0.0], dtype=np.float32)  # Idle action
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    assert step_count == 96
    assert info["episode_cost"] > 0
