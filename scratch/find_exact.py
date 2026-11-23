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

# What if night arbitrage is active from 0 to 6, or 1 to 6, or 2 to 5, or solar absorption plus off-peak charge?
# And what if daytime load following or reserve holding is used?
# Let's test combinations:
for start_h in [0, 1, 2, 3]:
    for end_h in [5, 6, 7]:
        for target_soc in [0.70, 0.75, 0.76, 0.78, 0.80]:
            for p_chg in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
                def dispatch(r, b, t):
                    solar_kw = float(r["solar_kw"])
                    load_kw = float(r["load_kw"])
                    hour = int(r["hour"])
                    excess = max(0.0, solar_kw - load_kw)
                    deficit = max(0.0, load_kw - solar_kw)

                    if excess > 0.001:
                        return -min(excess, b.get_max_feasible_charge_kw()), "CHARGE", ""

                    if 17 <= hour < 22:
                        if b.soc > b.min_soc and deficit > 0.001:
                            return min(deficit, b.get_max_feasible_discharge_kw()), "DISCHARGE", ""
                        return 0.0, "IDLE", ""

                    if start_h <= hour < end_h and b.soc < target_soc:
                        return -min(p_chg, b.get_max_feasible_charge_kw()), "CHARGE", ""

                    return 0.0, "IDLE", ""

                b = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
                sim = EnergySimulator(battery=b, tariff_model=tariff_model, allow_grid_charging=True)
                res = sim.run_simulation(test_df, dispatch, initial_soc=0.50)
                cost = res["metrics"]["total_electricity_cost"]
                adj = res["metrics"]["adjusted_electricity_cost"]
                soc = res["metrics"]["final_battery_soc"]
                if round(cost, 2) == 5.07 or round(adj, 2) == 4.91 or round(soc * 100, 1) == 75.5:
                    print(f"MATCH: start_h={start_h}, end_h={end_h}, target={target_soc}, p_chg={p_chg} -> Cost={cost:.3f}, AdjCost={adj:.3f}, FinalSOC={soc*100:.2f}%")
