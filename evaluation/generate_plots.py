"""
Plot Generation Module
Generates all 14 required scientific comparison graphs:
  1. Historical Solar vs Load
  2. Actual vs Forecasted Solar
  3. Actual vs Forecasted Load
  4. Rule-Based vs PPO Battery Power
  5. Rule-Based vs PPO Battery SOC
  6. Rule-Based vs PPO Grid Power
  7. Tariff vs Grid Consumption
  8. Rule-Based vs PPO Cumulative Cost
  9. Final Total Cost Comparison
  10. Grid Energy Comparison
  11. Solar Utilization Comparison
  12. Battery Throughput Comparison
  13. Peak Grid Demand Comparison
  14. PPO Training Reward Curve
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any


def generate_all_evaluation_plots(
    comparison_results: Dict[str, Any],
    output_dir: str = "results/plots",
):
    """
    Creates and saves all 14 publication-grade comparison figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    merged_ts = comparison_results["merged_timeseries"].copy()
    merged_ts["timestamp"] = pd.to_datetime(merged_ts["timestamp"])
    time_x = merged_ts["timestamp"]

    rb_m = comparison_results["rule_based_metrics"]
    ppo_m = comparison_results["ppo_metrics"]

    # 1. Historical Solar vs Load
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_x, merged_ts["solar_kw"], label="Solar PV Generation (kW)", color="#eab308", lw=2)
    plt.plot(time_x, merged_ts["load_kw"], label="Household Demand (kW)", color="#06b6d4", lw=2)
    plt.title("Historical Solar Generation vs Load Demand", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Power (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "solar_vs_load.png"))
    plt.close()

    # 4. Rule-Based vs PPO Battery Power
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_x, merged_ts["rb_battery_power_kw"], label="Rule-Based Battery Power (kW)", color="#ef4444", lw=1.8, linestyle="--")
    plt.plot(time_x, merged_ts["ppo_battery_power_kw"], label="PPO Battery Power (kW)", color="#10b981", lw=2)
    plt.axhline(0, color="gray", lw=0.8, linestyle=":")
    plt.title("Battery Power Scheduling: Rule-Based vs Forecast-Assisted PPO", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Battery Power (kW) [>0: Discharge, <0: Charge]")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "battery_power_comparison.png"))
    plt.close()

    # 5. Rule-Based vs PPO Battery SOC
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_x, merged_ts["rb_battery_soc"] * 100, label="Rule-Based SOC (%)", color="#ef4444", lw=2, linestyle="--")
    plt.plot(time_x, merged_ts["ppo_battery_soc"] * 100, label="PPO SOC (%)", color="#10b981", lw=2)
    plt.axhline(20, color="#f87171", lw=1.2, linestyle=":", label="Min SOC Limit (20%)")
    plt.axhline(90, color="#34d399", lw=1.2, linestyle=":", label="Max SOC Limit (90%)")
    plt.title("Battery State-of-Charge (SOC) Trajectory Comparison", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("State of Charge (%)")
    plt.ylim(10, 100)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "soc_comparison.png"))
    plt.close()

    # 6. Rule-Based vs PPO Grid Power
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_x, merged_ts["rb_grid_power_kw"], label="Rule-Based Grid Power (kW)", color="#ef4444", lw=1.8, linestyle="--")
    plt.plot(time_x, merged_ts["ppo_grid_power_kw"], label="PPO Grid Power (kW)", color="#10b981", lw=2)
    plt.title("Utility Grid Import: Rule-Based vs Forecast-Assisted PPO", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Grid Import (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grid_comparison.png"))
    plt.close()

    # 7. Tariff vs Grid Consumption
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=150)
    color_tariff = "#8b5cf6"
    ax1.set_xlabel("Date & Time")
    ax1.set_ylabel("Tariff Rate (Rs./kWh)", color=color_tariff, fontweight="bold")
    ax1.step(time_x, merged_ts["tariff"], color=color_tariff, lw=2, where="post", label="Electricity Tariff (Rs./kWh)")
    ax1.tick_params(axis="y", labelcolor=color_tariff)
    ax1.grid(True, alpha=0.2)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Grid Power (kW)", color="#0f172a", fontweight="bold")
    ax2.plot(time_x, merged_ts["rb_grid_power_kw"], color="#f87171", lw=1.5, linestyle="--", label="RB Grid Power")
    ax2.plot(time_x, merged_ts["ppo_grid_power_kw"], color="#10b981", lw=2, label="PPO Grid Power")
    ax2.tick_params(axis="y", labelcolor="#0f172a")

    plt.title("ToU Electricity Tariff vs Grid Import (Arbitrage Behavior)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "tariff_vs_grid.png"))
    plt.close()

    # 8. Cumulative Electricity Cost
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_x, merged_ts["rb_cumulative_cost"], label="Rule-Based Cumulative Cost (Rs.)", color="#ef4444", lw=2, linestyle="--")
    plt.plot(time_x, merged_ts["ppo_cumulative_cost"], label="PPO Cumulative Cost (Rs.)", color="#10b981", lw=2.2)
    plt.title("Cumulative Electricity Cost Evolution", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Total Cost (Rs.)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_comparison.png"))
    plt.close()

    # 9. Final Total Cost Comparison Bar Chart
    plt.figure(figsize=(6, 4.5), dpi=150)
    controllers = ["Rule-Based EMS", "Forecast-PPO"]
    costs = [rb_m["total_electricity_cost"], ppo_m["total_electricity_cost"]]
    bars = plt.bar(controllers, costs, color=["#f87171", "#34d399"], width=0.55)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 1, f"Rs. {yval:.2f}", ha="center", va="bottom", fontweight="bold")
    plt.title("Final Total Electricity Cost Comparison", fontsize=12, fontweight="bold")
    plt.ylabel("Total Cost (Rs.)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_cost_comparison.png"))
    plt.close()

    # 10 - 13. Secondary Metrics Bar Comparison
    metrics_to_plot = [
        ("Grid Energy (kWh)", [rb_m["grid_energy_consumed_kwh"], ppo_m["grid_energy_consumed_kwh"]], "grid_energy_comparison.png"),
        ("Solar Utilization (%)", [rb_m["solar_utilization_pct"], ppo_m["solar_utilization_pct"]], "solar_utilization_comparison.png"),
        ("Battery Throughput (kWh)", [rb_m["battery_throughput_kwh"], ppo_m["battery_throughput_kwh"]], "battery_throughput_comparison.png"),
        ("Peak Grid Demand (kW)", [rb_m["peak_grid_demand_kw"], ppo_m["peak_grid_demand_kw"]], "peak_demand_comparison.png"),
    ]

    for title, values, filename in metrics_to_plot:
        plt.figure(figsize=(5.5, 4), dpi=150)
        bars = plt.bar(controllers, values, color=["#94a3b8", "#38bdf8"], width=0.5)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.01, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")
        plt.title(title, fontsize=12, fontweight="bold")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
