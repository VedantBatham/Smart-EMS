"""
Backend Simulation Service
Handles procedural mathematical simulation data generation, dates discovery,
single-controller simulations, and interactive decision explanations.
All data is generated purely on the fly via the mathematical equations in MicrogridEnv
with ZERO CSV dataset files.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS
from rl.rl_service import RLController
from controllers.hybrid_dqn_ppo import HybridDQNPPOController
from environment.microgrid_env import MicrogridEnv


class BackendSimulationService:
    def __init__(self, data_path: Optional[str] = None):
        self.tariff_model = TariffModel()
        self.ppo_controller = RLController(algorithm="ppo", tariff_model=self.tariff_model)
        self.hybrid_controller = HybridDQNPPOController(tariff_model=self.tariff_model)
        self.rb_controller = RuleBasedEMS()
        self.env = MicrogridEnv(seed=100)
        self._cached_dates_metadata: Optional[Dict[str, Any]] = None

    def get_available_dates(self) -> Dict[str, Any]:
        """
        Returns list of calendar dates with partition, profile, and energy metadata.
        Generated purely mathematically from MicrogridEnv equations.
        """
        if self._cached_dates_metadata is not None:
            return self._cached_dates_metadata

        start_date = datetime(2026, 10, 1)
        unique_dates = []
        test_dates_list = []
        metadata = {}

        for d in range(14):
            cur_dt = start_date + timedelta(days=d)
            d_str = cur_dt.strftime("%Y-%m-%d")
            day_name = cur_dt.strftime("%A")
            is_test = (d < 7)

            unique_dates.append(d_str)
            if is_test:
                test_dates_list.append(d_str)

            # Generate day to compute accurate solar & load energy metadata
            day_env = MicrogridEnv(seed=100 + d)
            day_df = day_env.generate_day(day_of_week=d % 7, date_str=d_str)
            solar_kwh = float(day_df["solar_kw"].sum() * 0.25)
            load_kwh = float(day_df["load_kw"].sum() * 0.25)

            metadata[d_str] = {
                "date": d_str,
                "day_name": day_name,
                "label": f"{d_str} ({day_name}) — {'Test Benchmark' if is_test else 'Operational'}",
                "partition": "Held-Out Test Benchmark" if is_test else "Operational Profile",
                "is_test": is_test,
                "solar_kwh": round(solar_kwh, 2),
                "load_kwh": round(load_kwh, 2),
                "steps": len(day_df),
            }

        default_d = test_dates_list[0] if test_dates_list else unique_dates[0]

        res = {
            "dates": unique_dates,
            "test_dates": test_dates_list,
            "default_date": default_d,
            "metadata": metadata,
        }
        self._cached_dates_metadata = res
        return res

    def get_data_slice(
        self,
        date_str: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generates mathematical microgrid data on the fly according to specified criteria.
        Pure procedural equations: Gaussian solar bell curve, 3 campus peaks, TOU tariff.
        Zero CSV dataset files.
        """
        # 1. 14-Day Continuous Test Benchmark (1,344 steps)
        if mode == "14day" or date_str == "14day_benchmark":
            env = MicrogridEnv(seed=100)
            dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
            return pd.concat(dfs, ignore_index=True)

        # 2. 7-Day Continuous Test Benchmark (672 steps)
        if mode == "7day" or date_str == "7day_benchmark":
            env = MicrogridEnv(seed=100)
            dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(7)]
            return pd.concat(dfs, ignore_index=True)

        # 2. Date Range
        if start_date and end_date:
            try:
                dt_start = datetime.strptime(start_date, "%Y-%m-%d")
                dt_end = datetime.strptime(end_date, "%Y-%m-%d")
                days_count = max(1, min(14, (dt_end - dt_start).days + 1))
            except Exception:
                days_count = 7
            env = MicrogridEnv(seed=100)
            dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(days_count)]
            return pd.concat(dfs, ignore_index=True)

        # 3. Single Day Mode (96 steps)
        target_date = date_str or "2026-10-01"
        try:
            day_num = int(target_date.split("-")[-1])
            day_idx = (day_num - 1) % 14
        except Exception:
            day_idx = 0

        day_of_week = day_idx % 7
        env = MicrogridEnv(seed=100 + day_idx)
        return env.generate_day(day_of_week=day_of_week, date_str=target_date)

    def run_single_simulation(
        self,
        controller_name: str,
        date_str: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: Optional[str] = "day",
        initial_soc: float = 0.50,
        battery_capacity: float = 5.0,
        allow_grid_charging: bool = True,
    ) -> Dict[str, Any]:
        """Runs simulation for one chosen controller on procedural mathematical data."""
        slice_df = self.get_data_slice(date_str=date_str, start_date=start_date, end_date=end_date, mode=mode)
        battery = Battery(
            capacity_kwh=battery_capacity,
            initial_soc=initial_soc,
            time_step_hours=0.25,
        )
        sim = EnergySimulator(
            battery=battery,
            tariff_model=self.tariff_model,
            allow_grid_charging=allow_grid_charging,
        )

        c_name_lower = controller_name.lower()
        if "hybrid" in c_name_lower:
            res = sim.run_simulation(slice_df, self.hybrid_controller, controller_name="Hybrid DQN-PPO", initial_soc=initial_soc)
        elif "ppo" in c_name_lower:
            def step_fn(row, b_inst, t_val):
                dec = self.ppo_controller.compute_action(
                    solar_kw=float(row["solar_kw"]),
                    load_kw=float(row["load_kw"]),
                    battery=b_inst,
                    tariff=t_val,
                    row_context=row,
                )
                return dec.battery_power_kw, dec.action_type, dec.explanation

            res = sim.run_simulation(slice_df, step_fn, controller_name="Forecast-Assisted PPO", initial_soc=initial_soc)
        else:
            res = sim.run_simulation(slice_df, self.rb_controller, controller_name="Rule-Based EMS", initial_soc=initial_soc)

        # Convert costs from USD to INR (1 USD = ₹83.00)
        usd_to_inr = 83.0
        metrics = res["metrics"].copy()
        metrics["total_electricity_cost_usd"] = metrics.get("total_electricity_cost", 0.0)
        metrics["adjusted_electricity_cost_usd"] = metrics.get("adjusted_electricity_cost", 0.0)
        metrics["total_electricity_cost"] = round(metrics.get("total_electricity_cost", 0.0) * usd_to_inr, 2)
        metrics["adjusted_electricity_cost"] = round(metrics.get("adjusted_electricity_cost", 0.0) * usd_to_inr, 2)

        ts_df = res["timeseries"].copy()
        if "cumulative_grid_cost" in ts_df.columns:
            ts_df["cumulative_grid_cost"] = (ts_df["cumulative_grid_cost"] * usd_to_inr).round(2)
        if "interval_cost" in ts_df.columns:
            ts_df["interval_cost"] = (ts_df["interval_cost"] * usd_to_inr).round(4)
        if "tariff" in ts_df.columns:
            ts_df["tariff"] = (ts_df["tariff"] * usd_to_inr).round(2)

        return {
            "timeseries": ts_df.to_dict(orient="records"),
            "metrics": metrics,
        }
