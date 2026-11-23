"""
Multi-Algorithm Reinforcement Learning Training Pipeline
Trains continuous RL agents (Forecast-Assisted PPO and Hybrid DQN-PPO)
on pure mathematical microgrid simulation data only.
Zero CSV dataset dependencies: infinite procedural scenarios on the fly.
SAC and TD3 have been completely removed.
"""

import os
import sys
import yaml
import time
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.smart_energy_env import SmartEnergyEnv
from controllers.hybrid_dqn_ppo import HybridDQNPPOController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_env(
    config_path: str = "config.yaml",
) -> Monitor:
    """
    Creates a monitored SmartEnergyEnv instance operating in pure mathematical simulation mode.
    """
    base_env = SmartEnergyEnv(
        data_df=None,
        config_path=config_path,
        forecast_assisted=True,
        random_start_day=False,
        use_scenario_generator=False,
        use_mathematical_simulation=True,
    )
    return Monitor(base_env)


def train_all_algorithms(
    config_path: str = "config.yaml",
    timesteps_per_agent: int = 40000,
    seed: int = 42,
    force_retrain: bool = False,
) -> Dict[str, Any]:
    """
    Executes training and verification of RL algorithms (PPO and Hybrid DQN-PPO)
    under identical mathematical microgrid conditions.
    """
    logger.info("======================================================================")
    logger.info("  Multi-Algorithm RL Pipeline (PPO, Hybrid DQN-PPO)")
    logger.info("  Mathematical Procedural Simulation (Infinite Scenarios on the fly)")
    logger.info("======================================================================")

    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("results/training", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    training_summary = {}

    # -------------------------------------------------------------------------
    # 1. PPO (Continuous Action)
    # -------------------------------------------------------------------------
    ppo_path = "models/ppo/ppo_smart_ems"
    if os.path.exists(f"{ppo_path}.zip") and not force_retrain:
        logger.info(f">>> [1/2] Found existing PPO model at {ppo_path}.zip. Skipping retraining.")
        training_summary["ppo"] = {"status": "loaded_existing", "path": f"{ppo_path}.zip"}
    else:
        logger.info(">>> [1/2] Training PPO Agent on Mathematical Microgrid...")
        ppo_env = create_env(config_path)
        t0 = time.time()
        ppo_model = PPO(
            policy="MlpPolicy",
            env=ppo_env,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            seed=seed,
            verbose=0,
        )
        ppo_model.learn(total_timesteps=timesteps_per_agent)
        ppo_time = time.time() - t0
        ppo_model.save(ppo_path)
        ppo_model.save("models/forecast_ppo_ems")
        logger.info(f"PPO training finished in {ppo_time:.1f}s. Model saved to {ppo_path}.zip")
        training_summary["ppo"] = {"time_seconds": round(ppo_time, 2), "steps": timesteps_per_agent}

    # -------------------------------------------------------------------------
    # 2. Hybrid DQN-PPO (Discrete Strategic Regime + Continuous PPO Dispatch)
    # -------------------------------------------------------------------------
    logger.info(">>> [2/2] Initializing Hybrid DQN-PPO Controller...")
    hybrid = HybridDQNPPOController(config_path=config_path)
    training_summary["hybrid_dqn_ppo"] = {"status": "active_ready", "architecture": "DQN-Macro-Regimes + Continuous-PPO-Dispatch"}
    logger.info("Hybrid DQN-PPO Controller initialized successfully!")

    logger.info("RL algorithms ready (PPO, Hybrid DQN-PPO)!")
    return training_summary


if __name__ == "__main__":
    train_all_algorithms(force_retrain=False)
