"""
Script to execute head-to-head comparison between Rule-Based EMS and Forecast-Assisted PPO,
save output CSV results, calculate metrics, and generate comparison graphs.
"""

import os
import sys
import json
import pandas as pd
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.compare_controllers import run_head_to_head_comparison
from evaluation.generate_plots import generate_all_evaluation_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Loading Test Dataset for Head-to-Head Evaluation ===")
    test_path = "data/processed/test_data.csv"
    if not os.path.exists(test_path):
        logger.error(f"Test data not found at {test_path}. Run preprocessing first.")
        return

    test_df = pd.read_csv(test_path)
    # Evaluate across a representative 7-day test window (672 steps = 7 days)
    eval_slice = test_df.iloc[: 7 * 96].copy()
    logger.info(f"Evaluating both controllers across {len(eval_slice)} steps ({len(eval_slice)/96:.1f} days)...")

    logger.info("=== STEP 2: Running Comparative Simulation ===")
    results = run_head_to_head_comparison(
        data_df=eval_slice,
        initial_soc=0.50,
        save_results=True,
    )

    logger.info("=== STEP 3: Generating All 14 Comparison Plots ===")
    generate_all_evaluation_plots(results, output_dir="results/plots")

    logger.info("=== COMPARISON RESULTS SUMMARY ===")
    comp = results["comparison_metrics"]
    for key, item in comp.items():
        if isinstance(item, dict) and "metric" in item:
            print(f"{item['metric']:25s} | Rule-Based: {item['rule_based']:8.2f} {item['unit']} | PPO: {item['ppo']:8.2f} {item['unit']} | Diff: {item['difference']:8.2f} ({item['percent_change']:+5.1f}%)")

    print("\nNarrative Summary:")
    print(comp.get("narrative_summary", ""))


if __name__ == "__main__":
    main()
