import sys
sys.path.insert(0, ".")
import pandas as pd
from evaluation.compare_all import run_full_multi_comparison

from environment.microgrid_env import MicrogridEnv

print("Generating Mathematical Simulation Evaluation Dataset on the fly...")
print("(Pure mathematical procedural equations: Gaussian solar bell curve, 3 campus peaks, TOU tariff)")
env = MicrogridEnv(seed=100)
# Generate 14 continuous procedural evaluation days (2 full weeks)
dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
test_df = pd.concat(dfs, ignore_index=True)
print(f"Generated test timesteps: {len(test_df)} (14 days @ 15-min intervals)")

print("Running evaluation (Rule-Based + PPO + Hybrid DQN-PPO)...")
results = run_full_multi_comparison(
    data_df=test_df,
    initial_soc=0.50,
    config_path="config.yaml",
    save_results=True,
)
ct = results.get("comparison_table", {})
mm = results.get("results", {})
print()
print("==== EVALUATION RESULTS ====")
metrics_keys = [
    "total_electricity_cost",
    "adjusted_electricity_cost",
    "grid_energy_consumed_kwh",
    "solar_utilization_pct",
    "battery_throughput_kwh",
    "peak_grid_demand_kw",
    "final_battery_soc",
    "soc_violations",
    "unserved_energy_kwh",
]
algos = ["rule_based", "ppo", "hybrid_dqn_ppo"]
print(f"{'Metric':<38} {'RuleBase':>10} {'PPO':>10} {'Hybrid':>10}")
print("-" * 72)
for mk in metrics_keys:
    vals = []
    for a in algos:
        v = mm.get(a, {}).get(mk, "N/A")
        if isinstance(v, (int, float)):
            vals.append(f"{v:10.3f}")
        else:
            vals.append(f"{'N/A':>10}")
    print(f"{mk:<38} {' '.join(vals)}")
print()
best = results.get("selected_best_rl", "?")
best_name = results.get("selected_best_rl_name", "?")
print(f"Best RL Algorithm: {best_name} ({best})")
