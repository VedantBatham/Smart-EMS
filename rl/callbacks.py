"""
Training callbacks for PPO Energy Management System.
Logs episode rewards, electricity cost, battery throughput, and SOC violations to CSV.
"""

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOTrainingLoggerCallback(BaseCallback):
    """
    Custom callback to record training progress metrics at the end of each episode.
    """

    def __init__(self, log_csv_path: str = "results/training/ppo_training_log.csv", verbose: int = 0):
        super().__init__(verbose)
        self.log_csv_path = log_csv_path
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_throughputs = []
        self.episode_violations = []

        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
        # Initialize CSV with header
        with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "total_timesteps",
                "episode_reward",
                "episode_grid_cost",
                "battery_throughput_kwh",
                "soc_violations",
                "mean_recent_reward",
            ])

    def _on_step(self) -> bool:
        # Check if an episode ended
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if dones[0]:
            self.episode_count += 1
            info = infos[0]
            # Sum of step rewards in episode can be read or approximated from info
            ep_cost = info.get("episode_cost", 0.0)
            ep_th = info.get("episode_throughput", 0.0)
            ep_viol = info.get("episode_violations", 0)

            # Access episode reward from info if Monitor wrapper is used
            ep_reward = info.get("episode", {}).get("r", -ep_cost)

            self.episode_rewards.append(ep_reward)
            self.episode_costs.append(ep_cost)
            self.episode_throughputs.append(ep_th)
            self.episode_violations.append(ep_viol)

            recent_mean_reward = float(np.mean(self.episode_rewards[-20:]))

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_count,
                    self.num_timesteps,
                    round(float(ep_reward), 2),
                    round(float(ep_cost), 2),
                    round(float(ep_th), 2),
                    ep_viol,
                    round(recent_mean_reward, 2),
                ])

            if self.verbose > 0 and self.episode_count % 50 == 0:
                print(
                    f"Episode {self.episode_count:4d} | Timestep: {self.num_timesteps:6d} | "
                    f"Reward: {ep_reward:7.1f} | Cost: INR {ep_cost:5.1f} | Mean(20): {recent_mean_reward:7.1f}"
                )

        return True
