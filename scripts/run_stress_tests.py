"""
Robustness and Stress-Test Evaluation Module.
Evaluates Rule-Based EMS vs. Forecast-Assisted PPO on extreme/adversarial scenarios:
  1. High-Load Day (k_l = 1.30)
  2. Low-Solar Day (k_s = 0.40)
  3. Severe Midday Cloud Drop (70% reduction 12:00-14:00)
  4. Heavy Evening Peak Surge (1.40x multiplier 18:00-22:00)
  5. Low Initial Battery SOC (SOC_0 = 0.20)
All stress-test results are segregated from the primary untouched test-set benchmark.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS
from rl.ppo_service import PPOEMSController
from environment.scenario_generator import ScenarioGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_stress_scenario_df(
    base_df: pd.DataFrame,
    scenario_type: str,
    sg: ScenarioGenerator,
) -> pd.DataFrame:
    """Creates a modified DataFrame corresponding to a specific stress-test condition."""
    df = base_df.copy().reset_index(drop=True)
    hour_arr = (df["hour"] + df["minute"] / 60.0).to_numpy()

    if scenario_type == "high_load":
        # Scale load uniformly by +30%
        df["load_kw"] = df["load_kw"] * 1.30
    elif scenario_type == "low_solar":
        # Heavily overcast day: 60% solar reduction
        df["solar_kw"] = df["solar_kw"] * 0.40
    elif scenario_type == "cloud_event":
        # Midday drop of 70% between 12:00 and 14:00
        mask = (hour_arr >= 12.0) & (hour_arr <= 14.0)
        df.loc[mask, "solar_kw"] = df.loc[mask, "solar_kw"] * 0.30
    elif scenario_type == "evening_surge":
        # Evening demand surge: +40% between 18:00 and 22:00
        mask = (hour_arr >= 18.0) & (hour_arr <= 22.0)
        df.loc[mask, "load_kw"] = df.loc[mask, "load_kw"] * 1.40
    elif scenario_type == "low_initial_soc":
        # Handled in initial_soc parameter
        pass

    # Update forward rolling forecast approximations
    n = len(df)
    s = df["solar_kw"].to_numpy()
    l = df["load_kw"].to_numpy()
    fc_s_1h = np.zeros(n)
    fc_l_1h = np.zeros(n)
    fc_s_2h = np.zeros(n)
    fc_l_2h = np.zeros(n)

    for t in range(n):
        w1_s = s[t + 1 : min(n, t + 5)]
        w1_l = l[t + 1 : min(n, t + 5)]
        fc_s_1h[t] = float(np.mean(w1_s) if len(w1_s) > 0 else s[t])
        fc_l_1h[t] = float(np.mean(w1_l) if len(w1_l) > 0 else l[t])

        w2_s = s[t + 1 : min(n, t + 9)]
        w2_l = l[t + 1 : min(n, t + 9)]
        fc_s_2h[t] = float(np.mean(w2_s) if len(w2_s) > 0 else s[t])
        fc_l_2h[t] = float(np.mean(w2_l) if len(w2_l) > 0 else l[t])

    df["future_solar_1h_avg"] = fc_s_1h
    df["future_load_1h_avg"] = fc_l_1h
    df["future_solar_2h_avg"] = fc_s_2h
    df["future_load_2h_avg"] = fc_l_2h

    return df


def run_stress_test_suite(
    test_data_path: str = "data/processed/test_data.csv",
    output_dir: str = "results/robustness",
) -> List[Dict[str, Any]]:
    """Runs full adversarial stress testing across both controllers."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Loading test dataset from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)

    sg = ScenarioGenerator(config_path="config.yaml", seed=777)
    base_day = sg.sample_base_day(test_df, day_idx=0)

    stress_scenarios = [
        ("High-Load Day (+30% Demand)", "high_load", 0.50),
        ("Low-Solar Overcast Day (-60% Solar)", "low_solar", 0.50),
        ("Midday Cloud Drop (70% Dip 12-14h)", "cloud_event", 0.50),
        ("Evening Demand Surge (+40% Peak)", "evening_surge", 0.50),
        ("Low Battery Reserve (SOC_0 = 20%)", "low_initial_soc", 0.20),
    ]

    results_table = []

    for name, s_type, init_soc in stress_scenarios:
        logger.info(f"Executing stress test: {name} (Initial SOC={init_soc})...")
        scen_df = create_stress_scenario_df(base_day, s_type, sg)

        # 1. Run Rule-Based EMS
        rb_battery = Battery(initial_soc=init_soc)
        rb_tariff = TariffModel(config_path="config.yaml")
        rb_controller = RuleBasedEMS()
        rb_sim = EnergySimulator(battery=rb_battery, tariff_model=rb_tariff)
        rb_res = rb_sim.run_simulation(scen_df, rb_controller)

        # 2. Run Forecast-Assisted PPO EMS
        ppo_battery = Battery(initial_soc=init_soc)
        ppo_tariff = TariffModel(config_path="config.yaml")
        ppo_controller = PPOEMSController(model_path="models/forecast_ppo_ems.zip", tariff_model=ppo_tariff)
        ppo_sim = EnergySimulator(battery=ppo_battery, tariff_model=ppo_tariff)
        ppo_res = ppo_sim.run_simulation(scen_df, ppo_controller)

        rb_m = rb_res["metrics"]
        ppo_m = ppo_res["metrics"]

        cost_diff = ppo_m["total_electricity_cost"] - rb_m["total_electricity_cost"]
        cost_savings_pct = ((rb_m["total_electricity_cost"] - ppo_m["total_electricity_cost"]) / max(0.01, rb_m["total_electricity_cost"])) * 100.0

        item = {
            "scenario": name,
            "scenario_type": s_type,
            "initial_soc": init_soc,
            "rule_based_cost": round(rb_m["total_electricity_cost"], 2),
            "ppo_cost": round(ppo_m["total_electricity_cost"], 2),
            "cost_difference": round(cost_diff, 2),
            "ppo_savings_pct": round(cost_savings_pct, 2),
            "rule_based_grid_kwh": round(rb_m.get("grid_energy_consumed_kwh", 0.0), 2),
            "ppo_grid_kwh": round(ppo_m.get("grid_energy_consumed_kwh", 0.0), 2),
            "rule_based_solar_pct": round(rb_m.get("solar_utilization_pct", 0.0), 1),
            "ppo_solar_pct": round(ppo_m.get("solar_utilization_pct", 0.0), 1),
            "rule_based_final_soc": round(rb_m.get("final_battery_soc", 0.0) * 100, 1),
            "ppo_final_soc": round(ppo_m.get("final_battery_soc", 0.0) * 100, 1),
        }
        results_table.append(item)

    # Save outputs
    res_df = pd.DataFrame(results_table)
    csv_path = os.path.join(output_dir, "stress_test_summary.csv")
    json_path = os.path.join(output_dir, "stress_test_results.json")
    res_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(results_table, f, indent=2)

    logger.info(f"Saved stress test summary to {csv_path} and {json_path}")

    print("\n============================== STRESS TEST EVALUATION SUMMARY ==============================")
    print(f"{'Scenario Name':36s} | {'RB Cost':>9s} | {'PPO Cost':>9s} | {'Savings':>9s} | {'RB Grid':>8s} | {'PPO Grid':>8s}")
    print("-" * 92)
    for r in results_table:
        print(f"{r['scenario']:36s} | Rs.{r['rule_based_cost']:7.2f} | Rs.{r['ppo_cost']:7.2f} | {r['ppo_savings_pct']:+7.1f}% | {r['rule_based_grid_kwh']:6.2f}kW | {r['ppo_grid_kwh']:6.2f}kW")
    print("============================================================================================\n")

    return results_table


if __name__ == "__main__":
    run_stress_test_suite()
