"""
Backend Comparison Service
Coordinates head-to-head execution of Rule-Based vs Forecast-Assisted PPO,
caches active results in-memory and disk, and powers the Decision Explorer.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from evaluation.compare_all import run_full_multi_comparison
from backend.services.simulation_service import BackendSimulationService


class BackendComparisonService:
    def __init__(self):
        self.sim_service = BackendSimulationService()
        self.latest_results: Optional[Dict[str, Any]] = None
        self._load_cached_results()

    def _load_cached_results(self):
        comp_csv = "data/results/rl_comparison_results.csv"
        if not os.path.exists(comp_csv):
            comp_csv = "data/results/comparison_results.csv"

        metrics_json = "results/metrics/rl_comparison_metrics.json"
        if not os.path.exists(metrics_json):
            metrics_json = "results/metrics/controller_metrics.json"

        if os.path.exists(comp_csv) and os.path.exists(metrics_json):
            try:
                with open(metrics_json, "r") as f:
                    m_data = json.load(f)
                merged_df = pd.read_csv(comp_csv)
                self.latest_results = {
                    "merged_timeseries": merged_df,
                    "rule_based_metrics": m_data.get("rule_based", {}),
                    "ppo_metrics": m_data.get("ppo", {}),
                    "hybrid_metrics": m_data.get("hybrid_dqn_ppo", {}),
                    "selected_best_rl": m_data.get("selected_best_rl", "hybrid_dqn_ppo"),
                    "selected_best_rl_name": m_data.get("selected_best_rl_name", "Hybrid DQN-PPO"),
                    "comparison_table": m_data.get("comparison_table", {}),
                    "comparison_metrics": m_data.get("comparison", {}),
                }
            except Exception as e:
                print(f"Notice: Cache not loaded: {e}")

    def run_comparison(
        self,
        date_str: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: Optional[str] = "day",
        initial_soc: float = 0.50,
        battery_capacity: float = 5.0,
        allow_grid_charging: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs controlled multi-algorithm comparison on selected date/range with identical parameters.
        """
        data_slice = self.sim_service.get_data_slice(date_str, start_date, end_date, mode=mode)

        battery = Battery(
            capacity_kwh=battery_capacity,
            initial_soc=initial_soc,
            time_step_hours=0.25,
        )
        tariff_model = self.sim_service.tariff_model

        res = run_full_multi_comparison(
            data_df=data_slice,
            battery=battery,
            tariff_model=tariff_model,
            initial_soc=initial_soc,
            save_results=True,
        )

        all_res = res["results"]
        self.latest_results = {
            "merged_timeseries": res["merged_timeseries"],
            "rule_based_metrics": all_res.get("rule_based", {}),
            "ppo_metrics": all_res.get("ppo", {}),
            "hybrid_metrics": all_res.get("hybrid_dqn_ppo", {}),
            "selected_best_rl": res.get("selected_best_rl", "ppo"),
            "selected_best_rl_name": res.get("selected_best_rl_name", "Forecast-Assisted PPO"),
            "comparison_table": res.get("comparison_table", {}),
            "comparison_metrics": res["comparison_table"].get("metrics_map", {}),
        }

        return {
            "comparison_metrics": self.latest_results["comparison_metrics"],
            "comparison_table": self.latest_results["comparison_table"],
            "selected_best_rl": self.latest_results["selected_best_rl"],
            "selected_best_rl_name": self.latest_results["selected_best_rl_name"],
            "rule_based_metrics": self.latest_results["rule_based_metrics"],
            "ppo_metrics": self.latest_results["ppo_metrics"],
            "hybrid_metrics": self.latest_results["hybrid_metrics"],
            "timeseries": res["merged_timeseries"].to_dict(orient="records"),
        }

    def get_latest_comparison(self) -> Dict[str, Any]:
        """Returns the most recent comparison result."""
        if self.latest_results is None:
            return self.run_comparison()

        merged_df = self.latest_results["merged_timeseries"]
        return {
            "comparison_metrics": self.latest_results.get("comparison_metrics", {}),
            "comparison_table": self.latest_results.get("comparison_table", {}),
            "selected_best_rl": self.latest_results.get("selected_best_rl", "hybrid_dqn_ppo"),
            "selected_best_rl_name": self.latest_results.get("selected_best_rl_name", "Hybrid DQN-PPO"),
            "rule_based_metrics": self.latest_results.get("rule_based_metrics", {}),
            "ppo_metrics": self.latest_results.get("ppo_metrics", {}),
            "hybrid_metrics": self.latest_results.get("hybrid_metrics", {}),
            "timeseries": merged_df.to_dict(orient="records"),
        }

    def get_decision_at_timestamp(self, timestamp: str) -> Dict[str, Any]:
        """
        Finds row matching timestamp and returns detailed state, forecasts,
        Rule-Based action, PPO / Hybrid DQN-PPO actions, and explanations.
        """
        if self.latest_results is None:
            self.run_comparison()

        merged_df = self.latest_results["merged_timeseries"]
        row_match = merged_df[merged_df["timestamp"] == timestamp]

        if row_match.empty:
            row = merged_df.iloc[0]
        else:
            row = row_match.iloc[0]

        best_algo = self.latest_results.get("selected_best_rl", "ppo")

        return {
            "timestamp": str(row["timestamp"]),
            "solar_kw": float(row["solar_kw"]),
            "load_kw": float(row["load_kw"]),
            "tariff": float(row["tariff"]),
            "selected_best_rl": best_algo,
            "rule_based": {
                "action": str(row["rb_action"]),
                "battery_power_kw": float(row["rb_battery_power_kw"]),
                "battery_soc": float(row["rb_battery_soc"]),
                "grid_power_kw": float(row["rb_grid_power_kw"]),
                "interval_cost": float(row.get("rb_cumulative_cost", 0.0)),
                "explanation": str(row.get("rb_explanation", "")),
            },
            "ppo": {
                "action": str(row.get("ppo_action", "IDLE")),
                "battery_power_kw": float(row.get("ppo_battery_power_kw", 0.0)),
                "battery_soc": float(row.get("ppo_battery_soc", 0.5)),
                "grid_power_kw": float(row.get("ppo_grid_power_kw", 0.0)),
                "interval_cost": float(row.get("ppo_cumulative_cost", 0.0)),
                "explanation": str(row.get("ppo_explanation", "")),
            },
            "best_rl": {
                "algorithm": best_algo,
                "action": str(row.get(f"{best_algo}_action", row.get("ppo_action", "IDLE"))),
                "battery_power_kw": float(row.get(f"{best_algo}_battery_power_kw", row.get("ppo_battery_power_kw", 0.0))),
                "battery_soc": float(row.get(f"{best_algo}_battery_soc", row.get("ppo_battery_soc", 0.5))),
                "grid_power_kw": float(row.get(f"{best_algo}_grid_power_kw", row.get("ppo_grid_power_kw", 0.0))),
                "interval_cost": float(row.get(f"{best_algo}_cumulative_cost", row.get("ppo_cumulative_cost", 0.0))),
                "explanation": str(row.get(f"{best_algo}_explanation", row.get("ppo_explanation", ""))),
            }
        }

