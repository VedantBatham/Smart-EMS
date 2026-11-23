"""
Evaluation module for Forecast-Assisted PPO vs Rule-Based EMS.
Runs identical test days through both controllers and logs comprehensive comparisons.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS
from rl.ppo_service import PPOEMSController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_ppo_on_test_data(
    test_data_path: str = "data/processed/test_data.csv",
    days_to_eval: int = 7,  # Evaluate on 7 distinct test days (672 steps)
    initial_soc: float = 0.50,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Simulates Rule-Based EMS and Forecast-Assisted PPO over identical test conditions.
    """
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)

    steps_total = min(len(test_df), days_to_eval * 96)
    eval_slice = test_df.iloc[:steps_total].copy()

    sim_rb = EnergySimulator()
    rb_controller = RuleBasedEMS()

    logger.info("Running Baseline Rule-Based EMS simulation...")
    rb_res = sim_rb.run_simulation(
        data_df=eval_slice,
        controller_fn_or_obj=rb_controller,
        controller_name="Rule-Based EMS",
        initial_soc=initial_soc,
    )

    sim_ppo = EnergySimulator()
    ppo_controller = PPOEMSController()

    logger.info("Running Forecast-Assisted PPO EMS simulation...")

    def ppo_step_fn(row, battery, tariff):
        dec = ppo_controller.compute_action(
            solar_kw=row["solar_kw"],
            load_kw=row["load_kw"],
            battery=battery,
            tariff=tariff,
            row_context=row,
        )
        return dec.battery_power_kw, dec.action_type, dec.explanation

    ppo_res = sim_ppo.run_simulation(
        data_df=eval_slice,
        controller_fn_or_obj=ppo_step_fn,
        controller_name="Forecast-Assisted PPO",
        initial_soc=initial_soc,
    )

    logger.info("Simulation completed.")
    return rb_res, ppo_res
