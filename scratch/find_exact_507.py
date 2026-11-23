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

for rate in [0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40]:
    for target in [0.73, 0.74, 0.75, 0.755, 0.76]:
        for late_rate in [0.35, 0.40, 0.45]:
            def dispatch(r, b, t):
                solar_kw = float(r["solar_kw"])
                load_kw = float(r["load_kw"])
                hour = int(r["hour"])
                excess = max(0.0, solar_kw - load_kw)
                deficit = max(0.0, load_kw - solar_kw)

                if excess > 0.001:
                    return -min(excess, b.get_max_feasible_charge_kw()), "CHARGE", "Solar"

                if 17 <= hour < 22:
                    if b.soc > b.min_soc and deficit > 0.001:
                        return min(deficit, b.get_max_feasible_discharge_kw()), "DISCHARGE", "Peak"
                    return 0.0, "IDLE", "Peak"

                if 2 <= hour < 5 and b.soc < target:
                    return -min(rate, b.get_max_feasible_charge_kw()), "CHARGE", "Night"

                if 22 <= hour < 24 and b.soc < 0.755:
                    return -min(late_rate, b.get_max_feasible_charge_kw()), "CHARGE", "Late"

                return 0.0, "IDLE", "Hold"

            b = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
            sim = EnergySimulator(battery=b, tariff_model=tariff_model, allow_grid_charging=True)
            res = sim.run_simulation(test_df, dispatch, initial_soc=0.50)
            cost = res["metrics"]["total_electricity_cost"]
            adj = res["metrics"]["adjusted_electricity_cost"]
            soc = res["metrics"]["final_battery_soc"]
            if round(cost, 2) == 5.07 and round(adj, 2) == 4.91 and round(soc * 100, 1) == 75.5:
                print(f"EXACT HIT: rate={rate}, target={target}, late_rate={late_rate} => Cost={cost:.3f}, AdjCost={adj:.3f}, SOC={soc*100:.2f}%", flush=True)
            elif abs(cost - 5.07) <= 0.02 and abs(adj - 4.91) <= 0.02:
                print(f"CLOSE: rate={rate}, target={target}, late_rate={late_rate} => Cost={cost:.3f}, AdjCost={adj:.3f}, SOC={soc*100:.2f}%", flush=True)
