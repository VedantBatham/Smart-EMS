"""
Training Module for Forecast-Assisted Proximal Policy Optimization (PPO) EMS.
Trains Stable-Baselines3 PPO agent on pure mathematical microgrid simulation data only.
No CSV or XLSX dataset is used for RL policy training.
Procedural equations generate non-repeating solar, campus load, and TOU price signals on the fly.
Saves model checkpoint and training curves.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.smart_energy_env import SmartEnergyEnv
from environment.microgrid_env import MicrogridEnv
from rl.callbacks import PPOTrainingLoggerCallback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_forecast_ppo(
    config_path: str = "config.yaml",
    model_output_path: str = "models/forecast_ppo_ems.zip",
    training_log_path: str = "results/training/ppo_training_log.csv",
    plots_dir: str = "results/plots",
    total_timesteps: int = 50000,
    seed: int = 42,
) -> PPO:
    """
    Trains PPO agent for energy management scheduling purely on mathematical procedural data.
    """
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("models/ppo", exist_ok=True)

    # Load configuration
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    rl_cfg = cfg.get("rl", {})
    sim_cfg = cfg.get("simulation", {})

    logger.info("==========================================================================")
    logger.info("  Training RL Agent on Pure Mathematical Simulation Data Only")
    logger.info("  Zero CSV/XLSX dataset dependencies: infinite procedural scenarios")
    logger.info("  - Solar: Gaussian bell curve centered at noon (700 W base, 10% noise)")
    logger.info("  - Load: Campus 280 W base + 3 peaks (09:00, 13:00, 19:00) + weekend scaling")
    logger.info("  - Grid Price: Time-of-Use ₹0.30/kWh peak (17-22h) / ₹0.08/kWh off-peak")
    logger.info("==========================================================================")

    # Create and monitor pure mathematical environment
    base_env = SmartEnergyEnv(
        data_df=None,
        config_path=config_path,
        forecast_assisted=True,
        random_start_day=False,
        use_scenario_generator=False,
        use_mathematical_simulation=True,
    )
    env = Monitor(base_env)

    logger.info(f"Initializing PPO agent with {total_timesteps} target timesteps...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=rl_cfg.get("learning_rate", 0.0003),
        n_steps=min(rl_cfg.get("n_steps", 2048), 2048),
        batch_size=rl_cfg.get("batch_size", 64),
        n_epochs=10,
        gamma=rl_cfg.get("gamma", 0.99),
        gae_lambda=rl_cfg.get("gae_lambda", 0.95),
        clip_range=rl_cfg.get("clip_range", 0.2),
        ent_coef=rl_cfg.get("ent_coef", 0.005),
        verbose=1,
        seed=seed,
    )

    callback = PPOTrainingLoggerCallback(log_csv_path=training_log_path, verbose=1)

    logger.info("Commencing PPO policy optimization on mathematical data...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model checkpoints
    model.save(model_output_path.replace(".zip", ""))
    model.save("models/ppo/ppo_smart_ems")
    logger.info(f"Saved trained PPO model to {model_output_path} and models/ppo/ppo_smart_ems.zip")

    # Generate Training Reward Curve Plot
    try:
        if os.path.exists(training_log_path):
            log_df = pd.read_csv(training_log_path)
            if len(log_df) > 1:
                plt.figure(figsize=(10, 4.5), dpi=150)
                plt.plot(log_df["total_timesteps"], log_df["episode_reward"], alpha=0.3, color="#38bdf8", label="Episode Reward")
                plt.plot(log_df["total_timesteps"], log_df["mean_recent_reward"], color="#0284c7", lw=2, label="Mean Reward (20-ep MA)")
                plt.title("PPO Energy Management Training on Mathematical Data", fontsize=13, fontweight="bold")
                plt.xlabel("Total Timesteps")
                plt.ylabel("Reward")
                plt.grid(True, alpha=0.3)
                plt.legend(frameon=True)
                plt.tight_layout()
                reward_plot = os.path.join(plots_dir, "training_reward_curve.png")
                plt.savefig(reward_plot)
                plt.close()
                logger.info(f"Saved training reward curve to {reward_plot}")
    except Exception as e:
        logger.warning(f"Could not generate training reward plot: {e}")

    return model


if __name__ == "__main__":
    train_forecast_ppo(total_timesteps=50000)
