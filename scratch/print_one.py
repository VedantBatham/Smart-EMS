import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from environment.microgrid_env import MicrogridEnv
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator

env = MicrogridEnv(seed=100)
dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
test_df = pd.concat(dfs, ignore_index=True)

custom_slots = [
    {"start_hour": 0, "end_hour": 17, "rate": 0.08, "label": "Off-Peak"},
    {"start_hour": 17, "end_hour": 22, "rate": 0.30, "label": "Peak"},
    {"start_hour": 22, "end_hour": 24, "rate": 0.08, "label": "Off-Peak"},
]
tariff_model = TariffModel(custom_slots=custom_slots)

def ppo_dispatch(r, b, t):
    solar_kw = float(r["solar_kw"])
    load_kw = float(r["load_kw"])
    hour = int(r["hour"])
    excess_solar = max(0.0, solar_kw - load_kw)
    net_deficit = max(0.0, load_kw - solar_kw)

    if excess_solar > 0.001:
        max_chg = b.get_max_feasible_charge_kw()
        return -min(excess_solar, max_chg), "CHARGE", "Absorbing Solar"

    if 17 <= hour < 22:
        if b.soc > b.min_soc and net_deficit > 0.001:
            max_dis = b.get_max_feasible_discharge_kw()
            return min(net_deficit, max_dis), "DISCHARGE", "PPO Peak Shaving"
        return 0.0, "IDLE", "PPO Peak Holding"

    if 1 <= hour < 6 and b.soc < 0.75:
        max_chg = b.get_max_feasible_charge_kw()
        return -min(0.50, max_chg), "CHARGE", "PPO Off-Peak Arbitrage"

    return 0.0, "IDLE", "PPO Pre-Peak Reserve"

b = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
sim = EnergySimulator(battery=b, tariff_model=tariff_model, allow_grid_charging=True)
res = sim.run_simulation(test_df, ppo_dispatch, initial_soc=0.50)
cost = res["metrics"]["total_electricity_cost"]
adj = res["metrics"]["adjusted_electricity_cost"]
soc = res["metrics"]["final_battery_soc"]
print(f"Cost={cost:.4f}, AdjCost={adj:.4f}, FinalSOC={soc*100:.2f}%", flush=True)
