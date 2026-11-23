"""
Energy System Simulator
Executes sequential simulation of a microgrid (Solar + Battery + Grid + Load)
under identical historical profiles for any controller (Rule-Based, PPO, etc.).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.power_balance import calculate_power_balance
from controllers.rule_based import RuleBasedEMS


class EnergySimulator:
    """
    Simulates complete residential microgrid operation over a defined historical period.
    """

    def __init__(
        self,
        battery: Optional[Battery] = None,
        tariff_model: Optional[TariffModel] = None,
        allow_grid_export: bool = False,
        allow_grid_charging: bool = True,
    ):
        self.battery = battery if battery is not None else Battery.from_config()
        self.tariff_model = tariff_model if tariff_model is not None else TariffModel()
        self.allow_grid_export = allow_grid_export
        self.allow_grid_charging = allow_grid_charging

    def run_simulation(
        self,
        data_df: pd.DataFrame,
        controller_fn_or_obj: Any,
        controller_name: str = "Rule-Based EMS",
        initial_soc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs complete step-by-step simulation over data_df.
        
        Args:
            data_df: DataFrame containing at least timestamp, solar_kw, load_kw
            controller_fn_or_obj: Object with compute_action() or callable(row, battery, tariff)
            controller_name: Label for logging
            initial_soc: Optional starting SOC override
            
        Returns:
            Dictionary containing:
              - 'timeseries': DataFrame of every step
              - 'metrics': Summary metrics dictionary
        """
        self.battery.reset(initial_soc=initial_soc)

        records = []
        cumulative_cost = 0.0
        cumulative_grid_kwh = 0.0
        cumulative_solar_avail_kwh = 0.0
        cumulative_solar_used_kwh = 0.0
        peak_grid_kw = 0.0
        soc_violations = 0
        unserved_kwh = 0.0

        rows_iter = data_df.to_dict(orient="records") if isinstance(data_df, pd.DataFrame) else data_df
        for idx, row in enumerate(rows_iter):
            timestamp = str(row.get("timestamp", idx))
            solar_kw = float(row.get("solar_kw", 0.0))
            load_kw = float(row.get("load_kw", 0.0))
            if "hour" in row and not pd.isna(row["hour"]):
                hour = int(row["hour"])
                minute = int(row.get("minute", 0)) if not pd.isna(row.get("minute", 0)) else 0
            else:
                try:
                    dt_val = pd.to_datetime(timestamp)
                    hour = int(dt_val.hour)
                    minute = int(dt_val.minute)
                except Exception:
                    hour = int(row.get("hour", 0))
                    minute = int(row.get("minute", 0))

            tariff = float(row.get("tariff", row.get("current_tariff", self.tariff_model.get_tariff(hour, minute))))

            # Optional forecasts if available in row
            solar_fc = float(row.get("solar_forecast_kw", row.get("target_solar_next_15m", solar_kw)))
            load_fc = float(row.get("load_forecast_kw", row.get("target_load_next_15m", load_kw)))

            # Obtain controller action
            if hasattr(controller_fn_or_obj, "compute_action"):
                try:
                    decision = controller_fn_or_obj.compute_action(
                        solar_kw=solar_kw,
                        load_kw=load_kw,
                        battery=self.battery,
                        tariff=tariff,
                        row_context=row,
                    )
                except TypeError:
                    decision = controller_fn_or_obj.compute_action(
                        solar_kw=solar_kw,
                        load_kw=load_kw,
                        battery=self.battery,
                        tariff=tariff,
                    )
                requested_battery_kw = decision.battery_power_kw
                action_str = decision.action_type
                explanation = decision.explanation
            elif callable(controller_fn_or_obj):
                res = controller_fn_or_obj(row, self.battery, tariff)
                if hasattr(res, "battery_power_kw"):
                    requested_battery_kw = float(res.battery_power_kw)
                    action_str = getattr(res, "action_type", "IDLE")
                    explanation = getattr(res, "explanation", "")
                elif isinstance(res, tuple):
                    requested_battery_kw, action_str, explanation = res
                else:
                    requested_battery_kw = float(res)
                    action_str = "DISCHARGE" if requested_battery_kw > 0 else ("CHARGE" if requested_battery_kw < 0 else "IDLE")
                    explanation = ""
            else:
                requested_battery_kw = 0.0
                action_str = "IDLE"
                explanation = "Default Idle"

            # Execute battery step
            actual_battery_kw, next_soc, b_info = self.battery.step(requested_battery_kw)
            if b_info.get("violation", False):
                soc_violations += 1

            # Compute power balance
            pb = calculate_power_balance(
                load_kw=load_kw,
                solar_kw=solar_kw,
                battery_power_kw=actual_battery_kw,
                allow_grid_export=self.allow_grid_export,
                allow_grid_charging=self.allow_grid_charging,
            )

            grid_kw = pb["grid_power_kw"]
            solar_used_kw = pb["solar_used_kw"]
            solar_curtailed_kw = pb["solar_curtailed_kw"]

            dt = self.battery.time_step_hours  # 0.25 h
            interval_cost = tariff * grid_kw * dt
            cumulative_cost += interval_cost

            cumulative_grid_kwh += grid_kw * dt
            cumulative_solar_avail_kwh += solar_kw * dt
            cumulative_solar_used_kwh += solar_used_kw * dt
            peak_grid_kw = max(peak_grid_kw, grid_kw)
            unserved_kwh += pb["unserved_load_kw"] * dt

            records.append({
                "timestamp": timestamp,
                "solar_kw": round(solar_kw, 4),
                "load_kw": round(load_kw, 4),
                "solar_forecast_kw": round(solar_fc, 4),
                "load_forecast_kw": round(load_fc, 4),
                "tariff": round(tariff, 2),
                "battery_soc": round(next_soc, 4),
                "battery_power_kw": round(actual_battery_kw, 4),
                "grid_power_kw": round(grid_kw, 4),
                "solar_used_kw": round(solar_used_kw, 4),
                "solar_curtailed_kw": round(solar_curtailed_kw, 4),
                "interval_grid_cost": round(interval_cost, 4),
                "cumulative_grid_cost": round(cumulative_cost, 4),
                "controller": controller_name,
                "action": action_str,
                "explanation": explanation,
            })

        timeseries_df = pd.DataFrame(records)

        # Metrics calculation
        solar_utilization = (
            (cumulative_solar_used_kwh / cumulative_solar_avail_kwh * 100.0)
            if cumulative_solar_avail_kwh > 0.001
            else 100.0
        )

        # Economic terminal SOC adjustment:
        # If terminal SOC != initial SOC, value of remaining stored energy
        # Energy difference in kWh = (final_soc - initial_soc) * capacity
        # Valued at average tariff (or ~₹5.0/kWh)
        initial_soc_val = initial_soc if initial_soc is not None else self.battery.initial_soc
        final_soc_val = self.battery.soc
        stored_energy_diff_kwh = (final_soc_val - initial_soc_val) * self.battery.capacity_kwh
        avg_tariff = float(self.tariff_model.get_average_tariff()) if hasattr(self.tariff_model, "get_average_tariff") else 0.15
        if avg_tariff <= 0.001:
            avg_tariff = 0.15
        terminal_energy_value = stored_energy_diff_kwh * avg_tariff
        adjusted_cost = cumulative_cost - terminal_energy_value

        metrics = {
            "controller": controller_name,
            "total_electricity_cost": round(cumulative_cost, 2),
            "adjusted_electricity_cost": round(adjusted_cost, 2),
            "terminal_energy_credit": round(terminal_energy_value, 2),
            "grid_energy_consumed_kwh": round(cumulative_grid_kwh, 2),
            "solar_energy_available_kwh": round(cumulative_solar_avail_kwh, 2),
            "solar_energy_used_kwh": round(cumulative_solar_used_kwh, 2),
            "solar_utilization_pct": round(solar_utilization, 2),
            "battery_throughput_kwh": round(self.battery.total_throughput_kwh, 2),
            "peak_grid_demand_kw": round(peak_grid_kw, 2),
            "final_battery_soc": round(self.battery.soc, 4),
            "initial_battery_soc": round(initial_soc_val, 4),
            "soc_violations": soc_violations,
            "unserved_energy_kwh": round(unserved_kwh, 4),
            "total_timesteps": len(timeseries_df),
        }

        return {
            "timeseries": timeseries_df,
            "metrics": metrics,
        }
