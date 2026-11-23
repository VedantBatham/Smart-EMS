"""
Unit tests for Power Balance calculations
Validates that: P_solar + P_battery + P_grid = P_load
"""

import pytest
from energy_system.power_balance import calculate_power_balance


def test_power_balance_solar_exceeds_load():
    # Load = 1.0 kW, Solar = 2.5 kW, Battery charging with excess solar (-1.0 kW)
    flow = calculate_power_balance(
        load_kw=1.0,
        solar_kw=2.5,
        battery_power_kw=-1.0,  # charging 1 kW
        allow_grid_export=False,
        allow_grid_charging=True,
    )
    assert flow["solar_to_load_kw"] == 1.0
    assert flow["solar_to_battery_kw"] == 1.0
    assert flow["solar_curtailed_kw"] == 0.5
    assert flow["grid_power_kw"] == 0.0
    assert pytest.approx(flow["balance_error"], abs=1e-5) == 0.0


def test_power_balance_battery_discharging():
    # Load = 2.0 kW, Solar = 0.5 kW, Battery discharging 1.0 kW
    flow = calculate_power_balance(
        load_kw=2.0,
        solar_kw=0.5,
        battery_power_kw=1.0,  # discharging 1 kW
        allow_grid_export=False,
    )
    # Remaining net load = 2.0 - 0.5 - 1.0 = 0.5 kW from grid
    assert flow["solar_to_load_kw"] == 0.5
    assert flow["battery_to_load_kw"] == 1.0
    assert pytest.approx(flow["grid_power_kw"], rel=1e-4) == 0.5
    assert pytest.approx(flow["balance_error"], abs=1e-5) == 0.0


def test_power_balance_grid_charging():
    # Load = 1.0 kW, Solar = 0.0 kW, Battery charging from grid (-1.5 kW)
    flow = calculate_power_balance(
        load_kw=1.0,
        solar_kw=0.0,
        battery_power_kw=-1.5,
        allow_grid_charging=True,
    )
    # Grid must provide load (1.0 kW) + battery charge (1.5 kW) = 2.5 kW
    assert flow["grid_power_kw"] == 2.5
    assert flow["grid_to_battery_kw"] == 1.5
    assert flow["grid_to_load_kw"] == 1.0
    assert pytest.approx(flow["balance_error"], abs=1e-5) == 0.0
