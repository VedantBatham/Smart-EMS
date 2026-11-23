import sys
sys.path.insert(0, ".")
import pandas as pd
from environment.microgrid_env import MicrogridEnv
from evaluation.compare_all import run_full_multi_comparison

env = MicrogridEnv(seed=100)
dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
test_df = pd.concat(dfs, ignore_index=True)

res = run_full_multi_comparison(data_df=test_df, save_results=False)
rate = 83.0

rb_m = res["results"]["rule_based"]
ppo_m = res["results"]["ppo"]
hyb_m = res["results"]["hybrid_dqn_ppo"]

print("=== RAW USD ($) ===")
print(f"Rule-Based: ${rb_m['total_electricity_cost']:.2f}")
print(f"PPO       : ${ppo_m['total_electricity_cost']:.2f} (-{(rb_m['total_electricity_cost']-ppo_m['total_electricity_cost'])/rb_m['total_electricity_cost']*100:.1f}%)")
print(f"Hybrid    : ${hyb_m['total_electricity_cost']:.2f} (-{(rb_m['total_electricity_cost']-hyb_m['total_electricity_cost'])/rb_m['total_electricity_cost']*100:.1f}%)")

print("\n=== CONVERTED TO INR (₹ at 83.0/$) ===")
print(f"Rule-Based: ₹{rb_m['total_electricity_cost']*rate:.2f}")
print(f"PPO       : ₹{ppo_m['total_electricity_cost']*rate:.2f} (-{(rb_m['total_electricity_cost']-ppo_m['total_electricity_cost'])/rb_m['total_electricity_cost']*100:.1f}%)")
print(f"Hybrid    : ₹{hyb_m['total_electricity_cost']*rate:.2f} (-{(rb_m['total_electricity_cost']-hyb_m['total_electricity_cost'])/rb_m['total_electricity_cost']*100:.1f}%)")

print(f"Terminal-Adjusted RB: ₹{rb_m['adjusted_electricity_cost']*rate:.2f}")
print(f"Terminal-Adjusted RL: ₹{hyb_m['adjusted_electricity_cost']*rate:.2f} (-{(rb_m['adjusted_electricity_cost']-hyb_m['adjusted_electricity_cost'])/rb_m['adjusted_electricity_cost']*100:.1f}%)")
