"""
Controller Comparison Runner
Runs Rule-Based EMS and Forecast-Assisted PPO under strictly identical initial conditions,
historical profiles, battery constraints, and tariff schedules.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS
from rl.ppo_service import PPOEMSController
from evaluation.metrics import compute_comparison_metrics


def run_head_to_head_comparison(
    data_df: pd.DataFrame,
    battery: Optional[Battery] = None,
    tariff_model: Optional[TariffModel] = None,
    initial_soc: float = 0.50,
    config_path: str = "config.yaml",
    save_results: bool = True,
) -> Dict[str, Any]:
    """
    Executes fair evaluation of both controllers on the exact same dataset slice.
    """
    if battery is None:
        battery = Battery.from_config(config_path)
    if tariff_model is None:
        tariff_model = TariffModel(config_path)

    # 1. Rule-Based EMS
    sim_rb = EnergySimulator(battery=battery, tariff_model=tariff_model)
    rb_controller = RuleBasedEMS()
    rb_output = sim_rb.run_simulation(
        data_df=data_df,
        controller_fn_or_obj=rb_controller,
        controller_name="Rule-Based EMS",
        initial_soc=initial_soc,
    )

    # 2. Forecast-Assisted PPO EMS
    # Note: re-instantiate battery to ensure identical starting physical specs
    sim_ppo = EnergySimulator(
        battery=Battery(
            capacity_kwh=battery.capacity_kwh,
            initial_soc=initial_soc,
            min_soc=battery.min_soc,
            max_soc=battery.max_soc,
            max_charge_kw=battery.max_charge_kw,
            max_discharge_kw=battery.max_discharge_kw,
            charge_efficiency=battery.charge_efficiency,
            discharge_efficiency=battery.discharge_efficiency,
            time_step_hours=battery.time_step_hours,
        ),
        tariff_model=tariff_model,
    )
    ppo_controller = PPOEMSController(config_path=config_path, tariff_model=tariff_model)

    def ppo_eval_fn(row, b_inst, t_val):
        dec = ppo_controller.compute_action(
            solar_kw=float(row["solar_kw"]),
            load_kw=float(row["load_kw"]),
            battery=b_inst,
            tariff=t_val,
            row_context=row,
        )
        return dec.battery_power_kw, dec.action_type, dec.explanation

    ppo_output = sim_ppo.run_simulation(
        data_df=data_df,
        controller_fn_or_obj=ppo_eval_fn,
        controller_name="Forecast-Assisted PPO",
        initial_soc=initial_soc,
    )

    # 3. Comparative Analytics
    comp_metrics = compute_comparison_metrics(
        rb_metrics=rb_output["metrics"],
        ppo_metrics=ppo_output["metrics"],
    )

    # Combine timeseries for convenient plotting and download
    rb_ts = rb_output["timeseries"].copy()
    ppo_ts = ppo_output["timeseries"].copy()

    merged_ts = pd.DataFrame({
        "timestamp": rb_ts["timestamp"],
        "solar_kw": rb_ts["solar_kw"],
        "load_kw": rb_ts["load_kw"],
        "tariff": rb_ts["tariff"],
        "rb_battery_power_kw": rb_ts["battery_power_kw"],
        "ppo_battery_power_kw": ppo_ts["battery_power_kw"],
        "rb_battery_soc": rb_ts["battery_soc"],
        "ppo_battery_soc": ppo_ts["battery_soc"],
        "rb_grid_power_kw": rb_ts["grid_power_kw"],
        "ppo_grid_power_kw": ppo_ts["grid_power_kw"],
        "rb_interval_cost": rb_ts["interval_grid_cost"],
        "ppo_interval_cost": ppo_ts["interval_grid_cost"],
        "rb_cumulative_cost": rb_ts["cumulative_grid_cost"],
        "ppo_cumulative_cost": ppo_ts["cumulative_grid_cost"],
        "rb_action": rb_ts["action"],
        "ppo_action": ppo_ts["action"],
        "rb_explanation": rb_ts["explanation"],
        "ppo_explanation": ppo_ts["explanation"],
    })

    if save_results:
        os.makedirs("data/results", exist_ok=True)
        os.makedirs("results/metrics", exist_ok=True)
        rb_ts.to_csv("data/results/rule_based_results.csv", index=False)
        ppo_ts.to_csv("data/results/ppo_results.csv", index=False)
        merged_ts.to_csv("data/results/comparison_results.csv", index=False)

        metrics_payload = {
            "rule_based": rb_output["metrics"],
            "ppo": ppo_output["metrics"],
            "comparison": comp_metrics,
        }
        with open("results/metrics/controller_metrics.json", "w") as f:
            json.dump(metrics_payload, f, indent=2)

    return {
        "rule_based_timeseries": rb_ts,
        "ppo_timeseries": ppo_ts,
        "merged_timeseries": merged_ts,
        "rule_based_metrics": rb_output["metrics"],
        "ppo_metrics": ppo_output["metrics"],
        "comparison_metrics": comp_metrics,
    }
