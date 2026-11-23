"""
Script to train the Forecast-Assisted PPO Energy Management System agent.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.train_ppo import train_forecast_ppo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STARTING PPO REINFORCEMENT LEARNING TRAINING ===")
    total_timesteps = 100000  # Default 100k timesteps
    train_forecast_ppo(total_timesteps=total_timesteps)
    logger.info("=== PPO TRAINING COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
