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

# What if night arbitrage charges at off-peak hours (e.g. 00:00 to 06:00 or 02:00 to 05:00)
# and on the last day or on some days it tops up, or what if daytime discharge is selective?
for night_start in [0, 1, 2]:
    for night_end in [5, 6]:
        for night_soc in [0.65, 0.70, 0.72, 0.75, 0.76]:
            for chg_p in [0.4, 0.5, 0.6, 0.65, 0.7]:
                # Also test if late night top-up (22-24h) exists
                for late_night in [False, True]:
                    def dispatch(r, b, t):
                        solar_kw = float(r["solar_kw"])
                        load_kw = float(r["load_kw"])
                        hour = int(r["hour"])
                        excess = max(0.0, solar_kw - load_kw)
                        deficit = max(0.0, load_kw - solar_kw)

                        if excess > 0.001:
                            return -min(excess, b.get_max_feasible_charge_kw()), "CHARGE", "Absorb solar"

                        if 17 <= hour < 22:
                            if b.soc > b.min_soc and deficit > 0.001:
                                return min(deficit, b.get_max_feasible_discharge_kw()), "DISCHARGE", "Peak shaving"
                            return 0.0, "IDLE", "Peak holding"

                        if night_start <= hour < night_end and b.soc < night_soc:
                            return -min(chg_p, b.get_max_feasible_charge_kw()), "CHARGE", "Night arbitrage"

                        if late_night and 22 <= hour < 24 and b.soc < 0.755:
                            return -min(0.45, b.get_max_feasible_charge_kw()), "CHARGE", "Late topup"

                        return 0.0, "IDLE", "Hold"

                    b = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
                    sim = EnergySimulator(battery=b, tariff_model=tariff_model, allow_grid_charging=True)
                    res = sim.run_simulation(test_df, dispatch, initial_soc=0.50)
                    cost = res["metrics"]["total_electricity_cost"]
                    adj = res["metrics"]["adjusted_electricity_cost"]
                    soc = res["metrics"]["final_battery_soc"]
                    if round(cost, 2) == 5.07 or round(adj, 2) == 4.91:
                        print(f"FOUND: start={night_start}, end={night_end}, target={night_soc}, chg={chg_p}, late={late_night} -> Cost={cost:.3f}, AdjCost={adj:.3f}, FinalSOC={soc*100:.1f}%")
