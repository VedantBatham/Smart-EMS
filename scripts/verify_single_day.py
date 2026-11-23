import urllib.request
import json

payload = json.dumps({
    "mode": "day",
    "date": "2020-06-28",
    "initial_soc": 0.50,
    "battery_capacity": 5.0,
    "allow_grid_charging": True
}).encode("utf-8")

req = urllib.request.Request("http://127.0.0.1:8000/api/run/comparison", data=payload, headers={"Content-Type": "application/json"})
res = urllib.request.urlopen(req)
data = json.loads(res.read().decode("utf-8"))
comp = data["comparison_metrics"]

print("=== SINGLE-DAY 2020-06-28 EVALUATION RESULTS ===")
for k in ["electricity_cost", "adjusted_cost", "peak_grid_demand", "solar_utilization", "battery_throughput", "final_soc", "grid_energy"]:
    item = comp[k]
    print(f"{item['metric']:25s} | Rule-Based: {item['rule_based']:8.2f} {item['unit']} | PPO: {item['ppo']:8.2f} {item['unit']} | Diff: {item['difference']:8.2f} ({item['percent_change']:+.1f}%)")

print("\nNarrative:")
print(comp.get("narrative_summary", ""))
