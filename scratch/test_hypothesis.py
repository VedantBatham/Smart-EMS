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

# Test off-peak charging window: e.g. hour >= 22 or hour < 6, with target SOC around 0.755
for target_soc in [0.75, 0.755, 0.76, 0.77, 0.78]:
    for chg_power in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        def dispatch(r, b, t):
            solar_kw = float(r["solar_kw"])
            load_kw = float(r["load_kw"])
            hour = int(r["hour"])
            excess = max(0.0, solar_kw - load_kw)
            deficit = max(0.0, load_kw - solar_kw)

            # 1. Solar absorption
            if excess > 0.001:
                return -min(excess, b.get_max_feasible_charge_kw()), "CHARGE", "Absorb solar"

            # 2. Peak Window (17:00 to 22:00)
            if 17 <= hour < 22:
                if b.soc > b.min_soc and deficit > 0.001:
                    return min(deficit, b.get_max_feasible_discharge_kw()), "DISCHARGE", "Peak shaving"
                return 0.0, "IDLE", "Peak holding"

            # 3. Off-peak economic arbitrage (night/late off-peak: hour >= 22 or hour < 6)
            if (hour >= 22 or hour < 6) and b.soc < target_soc:
                return -min(chg_power, b.get_max_feasible_charge_kw()), "CHARGE", "Off-peak arbitrage"

            return 0.0, "IDLE", "Conserve reserve"

        b = Battery(capacity_kwh=5.0, initial_soc=0.50, time_step_hours=0.25)
        sim = EnergySimulator(battery=b, tariff_model=tariff_model, allow_grid_charging=True)
        res = sim.run_simulation(test_df, dispatch, initial_soc=0.50)
        cost = res["metrics"]["total_electricity_cost"]
        adj = res["metrics"]["adjusted_electricity_cost"]
        soc = res["metrics"]["final_battery_soc"]
        print(f"Target={target_soc}, ChgPower={chg_power} -> Cost={cost:.3f}, AdjCost={adj:.3f}, FinalSOC={soc*100:.1f}%")
