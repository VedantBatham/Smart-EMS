import requests
import json

base_url = "http://127.0.0.1:8000"

# 1. Test Dates
r = requests.get(f"{base_url}/api/data/dates")
print("1. /api/data/dates:", r.status_code)
dates_data = r.json()
print("Dates count:", len(dates_data.get("dates", [])))
print("Default date:", dates_data.get("default_date"))
print("Test dates:", dates_data.get("test_dates")[:3])

# 2. Test Initial Comparison Results
r = requests.get(f"{base_url}/api/results/comparison")
print("\n2. /api/results/comparison:", r.status_code)
comp_data = r.json()
print("Selected Best RL:", comp_data.get("selected_best_rl"))
rb_cost = comp_data.get("rule_based_metrics", {}).get("total_electricity_cost")
best_cost = comp_data.get("hybrid_metrics", {}).get("total_electricity_cost") or comp_data.get("ppo_metrics", {}).get("total_electricity_cost")
print(f"RB Cost: Rs. {rb_cost}, RL Cost: Rs. {best_cost}")
if rb_cost and best_cost:
    saving = (rb_cost - best_cost) / rb_cost * 100
    print(f"Cost Saving: {saving:.1f}%")

# 3. Test Single Day Run
r = requests.post(f"{base_url}/api/run/comparison", json={"mode": "day", "date": "2026-10-01", "initial_soc": 0.5, "battery_capacity": 5.0})
print("\n3. POST /api/run/comparison (Single Day):", r.status_code)
day_res = r.json()
day_rb = day_res.get("rule_based_metrics", {}).get("total_electricity_cost")
day_rl = day_res.get("hybrid_metrics", {}).get("total_electricity_cost")
print(f"Single Day: RB=Rs. {day_rb}, RL=Rs. {day_rl} -> Saving = {((day_rb - day_rl)/day_rb*100):.1f}%")

# 4. Test 7-Day Benchmark Run
r = requests.post(f"{base_url}/api/run/comparison", json={"mode": "7day", "initial_soc": 0.5, "battery_capacity": 5.0})
print("\n4. POST /api/run/comparison (7-Day Benchmark):", r.status_code)
week_res = r.json()
week_rb = week_res.get("rule_based_metrics", {}).get("total_electricity_cost")
week_rl = week_res.get("hybrid_metrics", {}).get("total_electricity_cost")
print(f"7-Day Benchmark: RB=Rs. {week_rb}, RL=Rs. {week_rl} -> Saving = {((week_rb - week_rl)/week_rb*100):.1f}%")
