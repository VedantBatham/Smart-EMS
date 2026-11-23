"""
Performance Evaluation and Metrics Module
Calculates primary and secondary performance indicators for energy controllers,
including fair economic terminal SOC adjustments and dynamic narrative summaries.
"""

from typing import Dict, Any
import numpy as np


def compute_comparison_metrics(
    rb_metrics: Dict[str, Any],
    ppo_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes comparative metrics, percentage savings, and honest tradeoff indicators.
    """
    rb_cost = float(rb_metrics["total_electricity_cost"])
    ppo_cost = float(ppo_metrics["total_electricity_cost"])

    cost_diff = rb_cost - ppo_cost
    cost_saving_pct = (cost_diff / rb_cost * 100.0) if rb_cost > 0 else 0.0

    rb_adj_cost = float(rb_metrics.get("adjusted_electricity_cost", rb_cost))
    ppo_adj_cost = float(ppo_metrics.get("adjusted_electricity_cost", ppo_cost))
    adj_cost_diff = rb_adj_cost - ppo_adj_cost
    adj_cost_saving_pct = (adj_cost_diff / rb_adj_cost * 100.0) if rb_adj_cost > 0 else 0.0

    rb_grid = float(rb_metrics["grid_energy_consumed_kwh"])
    ppo_grid = float(ppo_metrics["grid_energy_consumed_kwh"])
    grid_diff = rb_grid - ppo_grid
    grid_diff_pct = (grid_diff / rb_grid * 100.0) if rb_grid > 0 else 0.0

    rb_solar = float(rb_metrics["solar_utilization_pct"])
    ppo_solar = float(ppo_metrics["solar_utilization_pct"])
    solar_diff = ppo_solar - rb_solar

    rb_th = float(rb_metrics["battery_throughput_kwh"])
    ppo_th = float(ppo_metrics["battery_throughput_kwh"])
    th_diff = ppo_th - rb_th
    th_diff_pct = (th_diff / rb_th * 100.0) if rb_th > 0 else 0.0

    rb_peak = float(rb_metrics["peak_grid_demand_kw"])
    ppo_peak = float(ppo_metrics["peak_grid_demand_kw"])
    peak_diff = rb_peak - ppo_peak
    peak_diff_pct = (peak_diff / rb_peak * 100.0) if rb_peak > 0 else 0.0

    comparison = {
        "electricity_cost": {
            "metric": "Electricity Cost",
            "unit": "Rs.",
            "rule_based": round(rb_cost, 2),
            "ppo": round(ppo_cost, 2),
            "difference": round(cost_diff, 2),
            "percent_change": round(cost_saving_pct, 2),
            "better": "ppo" if cost_diff > 0 else "rule_based",
        },
        "adjusted_cost": {
            "metric": "Terminal-Adjusted Cost",
            "unit": "Rs.",
            "rule_based": round(rb_adj_cost, 2),
            "ppo": round(ppo_adj_cost, 2),
            "difference": round(adj_cost_diff, 2),
            "percent_change": round(adj_cost_saving_pct, 2),
            "better": "ppo" if adj_cost_diff > 0 else "rule_based",
        },
        "grid_energy": {
            "metric": "Grid Energy Consumed",
            "unit": "kWh",
            "rule_based": round(rb_grid, 2),
            "ppo": round(ppo_grid, 2),
            "difference": round(grid_diff, 2),
            "percent_change": round(grid_diff_pct, 2),
            "better": "ppo" if grid_diff > 0 else "rule_based",
        },
        "solar_utilization": {
            "metric": "Solar Utilization",
            "unit": "%",
            "rule_based": round(rb_solar, 2),
            "ppo": round(ppo_solar, 2),
            "difference": round(solar_diff, 2),
            "percent_change": round(solar_diff, 2),
            "better": "ppo" if solar_diff > 0 else "rule_based",
        },
        "battery_throughput": {
            "metric": "Battery Throughput",
            "unit": "kWh",
            "rule_based": round(rb_th, 2),
            "ppo": round(ppo_th, 2),
            "difference": round(th_diff, 2),
            "percent_change": round(th_diff_pct, 2),
            "better": "ppo" if th_diff < 0 else "rule_based",
        },
        "peak_grid_demand": {
            "metric": "Peak Grid Demand",
            "unit": "kW",
            "rule_based": round(rb_peak, 2),
            "ppo": round(ppo_peak, 2),
            "difference": round(peak_diff, 2),
            "percent_change": round(peak_diff_pct, 2),
            "better": "ppo" if peak_diff > 0 else "rule_based",
        },
        "final_soc": {
            "metric": "Final Battery SOC",
            "unit": "%",
            "rule_based": round(float(rb_metrics["final_battery_soc"]) * 100, 1),
            "ppo": round(float(ppo_metrics["final_battery_soc"]) * 100, 1),
            "difference": round((float(ppo_metrics["final_battery_soc"]) - float(rb_metrics["final_battery_soc"])) * 100, 1),
            "percent_change": 0.0,
            "better": "neutral",
        },
        "soc_violations": {
            "metric": "SOC Violations",
            "unit": "count",
            "rule_based": int(rb_metrics.get("soc_violations", 0)),
            "ppo": int(ppo_metrics.get("soc_violations", 0)),
            "difference": int(rb_metrics.get("soc_violations", 0)) - int(ppo_metrics.get("soc_violations", 0)),
            "percent_change": 0.0,
            "better": "ppo" if ppo_metrics.get("soc_violations", 0) <= rb_metrics.get("soc_violations", 0) else "rule_based",
        },
        "unserved_energy": {
            "metric": "Unserved Energy",
            "unit": "kWh",
            "rule_based": round(float(rb_metrics.get("unserved_energy_kwh", 0.0)), 3),
            "ppo": round(float(ppo_metrics.get("unserved_energy_kwh", 0.0)), 3),
            "difference": 0.0,
            "percent_change": 0.0,
            "better": "neutral",
        },
    }

    # Generate dynamic report narrative without hard-coded numbers
    summary_sentences = []
    if cost_diff > 0:
        summary_sentences.append(f"Forecast-Assisted PPO reduced total electricity expenditure by Rs. {cost_diff:.2f} ({cost_saving_pct:.1f}% savings).")
    elif cost_diff < 0:
        summary_sentences.append(f"Rule-Based EMS achieved Rs. {abs(cost_diff):.2f} lower raw cost, but evaluation of terminal stored energy should be reviewed.")
    else:
        summary_sentences.append("Both controllers achieved identical direct electricity costs.")

    if peak_diff > 0:
        summary_sentences.append(f"Peak grid demand was curtailed by {peak_diff:.2f} kW ({peak_diff_pct:.1f}% reduction).")

    if solar_diff > 0:
        summary_sentences.append(f"Solar self-consumption efficiency improved by +{solar_diff:.1f}% points.")

    if th_diff > 0:
        summary_sentences.append(f"PPO utilized {th_diff:.2f} kWh additional battery cycling to capitalize on tariff arbitrage.")
    elif th_diff < 0:
        summary_sentences.append(f"PPO preserved battery life with {abs(th_diff):.2f} kWh less cycling throughput.")

    comparison["narrative_summary"] = " ".join(summary_sentences)
    return comparison
