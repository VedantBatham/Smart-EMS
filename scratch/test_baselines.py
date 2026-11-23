import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from environment.microgrid_env import MicrogridEnv
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS

env = MicrogridEnv(seed=100)
dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
test_df = pd.concat(dfs, ignore_index=True)

custom_slots = [
    {"start_hour": 0, "end_hour": 17, "rate": 0.08, "label": "Off-Peak"},
    {"start_hour": 17, "end_hour": 22, "rate": 0.30, "label": "Peak"},
    {"start_hour": 22, "end_hour": 24, "rate": 0.08, "label": "Off-Peak"},
]
tariff_model = TariffModel(custom_slots=custom_slots)

# Baseline Rule-Based
b_rb = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
sim_rb = EnergySimulator(battery=b_rb, tariff_model=tariff_model, allow_grid_charging=False)
rb_ctrl = RuleBasedEMS()
res_rb = sim_rb.run_simulation(test_df, rb_ctrl, initial_soc=0.50)
cost_rb = res_rb["metrics"]["total_electricity_cost"]
adj_rb = res_rb["metrics"]["adjusted_electricity_cost"]
soc_rb = res_rb["metrics"]["final_battery_soc"]
print(f"Rule-Based 14-day: Cost=Rs.{cost_rb:.2f}, AdjCost=Rs.{adj_rb:.2f}, Final SOC={soc_rb*100:.1f}%")

# Let's test single day (Day 0 - Wednesday)
df_1day = env.generate_day(day_of_week=0, date_str="2026-10-01")
b_rb_1 = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
sim_rb_1 = EnergySimulator(battery=b_rb_1, tariff_model=tariff_model, allow_grid_charging=False)
res_rb_1 = sim_rb_1.run_simulation(df_1day, rb_ctrl, initial_soc=0.50)
cost_rb_1 = res_rb_1["metrics"]["total_electricity_cost"]
adj_rb_1 = res_rb_1["metrics"]["adjusted_electricity_cost"]
print(f"Rule-Based 1-day: Cost=Rs.{cost_rb_1:.2f}, AdjCost=Rs.{adj_rb_1:.2f}")

# Let's test 7-day benchmark
dfs_7 = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(7)]
df_7day = pd.concat(dfs_7, ignore_index=True)
b_rb_7 = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
sim_rb_7 = EnergySimulator(battery=b_rb_7, tariff_model=tariff_model, allow_grid_charging=False)
res_rb_7 = sim_rb_7.run_simulation(df_7day, rb_ctrl, initial_soc=0.50)
cost_rb_7 = res_rb_7["metrics"]["total_electricity_cost"]
adj_rb_7 = res_rb_7["metrics"]["adjusted_electricity_cost"]
print(f"Rule-Based 7-day: Cost=Rs.{cost_rb_7:.2f}, AdjCost=Rs.{adj_rb_7:.2f}")
