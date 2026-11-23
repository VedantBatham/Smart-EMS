import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS
from rl.ppo_service import PPOEMSController

df = pd.read_csv("data/processed/test_data.csv").iloc[:96]
ppo = PPOEMSController()
sim1 = EnergySimulator()
sim2 = EnergySimulator()

r1 = sim1.run_simulation(df, RuleBasedEMS())["timeseries"]
r2 = sim2.run_simulation(df, lambda row, b, t: ppo.compute_action(row.solar_kw, row.load_kw, b, t, row).battery_power_kw)["timeseries"]

midday = df[(df.hour >= 11) & (df.hour <= 14)]
print("Time  | Solar | Load | RB_Batt | PPO_Batt | RB_Curtailed | PPO_Curtailed | RB_SOC | PPO_SOC")
for i, r in midday.iterrows():
    print(f"{r.hour:02d}:{r.minute:02d} | {r.solar_kw:5.2f} | {r.load_kw:4.2f} | "
          f"{r1.loc[i, 'battery_power_kw']:7.2f} | {r2.loc[i, 'battery_power_kw']:8.2f} | "
          f"{r1.loc[i, 'solar_curtailed_kw']:12.2f} | {r2.loc[i, 'solar_curtailed_kw']:13.2f} | "
          f"{r1.loc[i, 'battery_soc']:6.2f} | {r2.loc[i, 'battery_soc']:7.2f}")
