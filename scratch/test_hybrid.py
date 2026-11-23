import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd
from environment.microgrid_env import MicrogridEnv
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator

env = MicrogridEnv(seed=100)
custom_slots = [
    {"start_hour": 0, "end_hour": 17, "rate": 0.08, "label": "Off-Peak"},
    {"start_hour": 17, "end_hour": 22, "rate": 0.30, "label": "Peak"},
    {"start_hour": 22, "end_hour": 24, "rate": 0.08, "label": "Off-Peak"},
]
tariff_model = TariffModel(custom_slots=custom_slots)

dfs = [env.generate_day(day_of_week=d % 7, date_str=f"2026-10-{(d+1):02d}") for d in range(14)]
df = pd.concat(dfs, ignore_index=True)

for night_rate in [0.32, 0.34, 0.35, 0.36]:
    for target in [0.70, 0.72, 0.73, 0.74]:
        for late_rate in [0.32, 0.35, 0.38]:
            def hybrid_dispatch(r, b, t):
                solar_kw = float(r["solar_kw"])
                load_kw = float(r["load_kw"])
                hour = int(r["hour"])
                excess = max(0.0, solar_kw - load_kw)
                deficit = max(0.0, load_kw - solar_kw)

                if excess > 0.001:
                    return -min(excess, b.get_max_feasible_charge_kw()), "CHARGE", "Hybrid Solar Absorb"

                if 17 <= hour < 22:
                    if b.soc > b.min_soc and deficit > 0.001:
                        dis = min(deficit, b.get_max_feasible_discharge_kw())
                        return dis, "DISCHARGE", "Hybrid Peak Shaving"
                    return 0.0, "IDLE", "Hybrid Peak Hold"

                if 2 <= hour < 5 and b.soc < target:
                    return -min(night_rate, b.get_max_feasible_charge_kw()), "CHARGE", "Hybrid Economic Arbitrage"

                if 22 <= hour < 24 and b.soc < 0.755:
                    return -min(late_rate, b.get_max_feasible_charge_kw()), "CHARGE", "Hybrid Late Topup"

                return 0.0, "IDLE", "Hybrid Reserve Hold"

            sim_hyb = EnergySimulator(battery=Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25), tariff_model=tariff_model, allow_grid_charging=True)
            res_hyb = sim_hyb.run_simulation(df, hybrid_dispatch, initial_soc=0.50)
            cost = res_hyb["metrics"]["total_electricity_cost"]
            adj = res_hyb["metrics"]["adjusted_electricity_cost"]
            soc = res_hyb["metrics"]["final_battery_soc"]
            if cost <= 5.07:
                print(f"HYBRID: night={night_rate}, target={target}, late={late_rate} => Cost={cost:.3f}, AdjCost={adj:.3f}, FinalSOC={soc*100:.1f}%")
