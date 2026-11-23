"""
Unit tests for Rule-Based EMS controller and simulator execution.
"""

import pandas as pd
import pytest
from energy_system.battery import Battery
from energy_system.tariff import TariffModel
from energy_system.simulator import EnergySimulator
from controllers.rule_based import RuleBasedEMS


def test_rule_based_excess_solar_charges_battery():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50)
    ems = RuleBasedEMS()

    # Load = 1.0 kW, Solar = 2.0 kW -> excess 1.0 kW solar
    decision = ems.compute_action(solar_kw=2.0, load_kw=1.0, battery=battery, tariff=5.0)

    assert decision.action_type == "CHARGE"
    assert decision.battery_power_kw < 0
    # Requested charge should be -1.0 kW (limited by excess solar and battery max charge)
    assert pytest.approx(decision.battery_power_kw, abs=1e-3) == -1.0


def test_rule_based_solar_deficit_discharges_battery():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50)
    ems = RuleBasedEMS()

    # Load = 2.0 kW, Solar = 0.5 kW -> deficit 1.5 kW
    decision = ems.compute_action(solar_kw=0.5, load_kw=2.0, battery=battery, tariff=5.0)

    assert decision.action_type == "DISCHARGE"
    assert decision.battery_power_kw > 0
    assert pytest.approx(decision.battery_power_kw, abs=1e-3) == 1.5


def test_rule_based_min_soc_stops_discharge():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.20, min_soc=0.20)
    ems = RuleBasedEMS()

    # Even with deficit, battery cannot discharge below min_soc
    decision = ems.compute_action(solar_kw=0.0, load_kw=1.5, battery=battery, tariff=10.0)

    assert decision.battery_power_kw == 0.0
    assert decision.action_type == "IDLE"


def test_simulator_rule_based_run():
    # Create 96 steps test data (1 full day)
    dates = pd.date_range("2020-01-01", periods=96, freq="15min")
    solar_kw = [0.0 if i < 28 or i > 76 else 2.0 for i in range(96)]
    load_kw = [1.0] * 96

    df = pd.DataFrame({
        "timestamp": dates,
        "solar_kw": solar_kw,
        "load_kw": load_kw,
    })

    sim = EnergySimulator()
    res = sim.run_simulation(df, RuleBasedEMS())

    assert len(res["timeseries"]) == 96
    metrics = res["metrics"]
    assert metrics["controller"] == "Rule-Based EMS"
    assert metrics["total_electricity_cost"] > 0
    assert metrics["grid_energy_consumed_kwh"] > 0
    assert metrics["solar_utilization_pct"] > 0
    assert metrics["final_battery_soc"] >= 0.20
    assert metrics["final_battery_soc"] <= 0.90
