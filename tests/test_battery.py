"""
Unit tests for Battery Energy Storage System (BESS)
"""

import pytest
from energy_system.battery import Battery


def test_battery_initialization():
    battery = Battery(
        capacity_kwh=5.0,
        initial_soc=0.50,
        min_soc=0.20,
        max_soc=0.90,
        max_charge_kw=1.5,
        max_discharge_kw=1.5,
    )
    assert battery.soc == 0.50
    assert battery.capacity_kwh == 5.0
    assert battery.min_soc == 0.20
    assert battery.max_soc == 0.90


def test_battery_charge_dynamics():
    # dt = 0.25 h, eta_c = 0.95, E = 5.0 kWh
    # delta_soc = (0.95 * 1.0 * 0.25) / 5.0 = 0.2375 / 5.0 = 0.0475
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50, charge_efficiency=0.95, time_step_hours=0.25)
    actual_power, next_soc, info = battery.step(-1.0)  # charge at 1.0 kW

    assert actual_power == -1.0
    assert pytest.approx(next_soc, rel=1e-3) == 0.50 + 0.0475
    assert info["violation"] is False


def test_battery_discharge_dynamics():
    # dt = 0.25 h, eta_d = 0.95, E = 5.0 kWh
    # delta_soc = (1.0 * 0.25) / (0.95 * 5.0) = 0.25 / 4.75 = 0.05263
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50, discharge_efficiency=0.95, time_step_hours=0.25)
    actual_power, next_soc, info = battery.step(1.0)  # discharge at 1.0 kW

    assert actual_power == 1.0
    assert pytest.approx(next_soc, rel=1e-3) == 0.50 - (0.25 / 4.75)
    assert info["violation"] is False


def test_battery_max_charge_limit_clipping():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50, max_charge_kw=1.5)
    # Attempt charging with 3.0 kW (exceeding 1.5 kW max)
    actual_power, next_soc, info = battery.step(-3.0)

    assert actual_power == -1.5
    assert info["violation"] is True


def test_battery_max_discharge_limit_clipping():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50, max_discharge_kw=1.5)
    # Attempt discharging with 3.0 kW (exceeding 1.5 kW max)
    actual_power, next_soc, info = battery.step(3.0)

    assert actual_power == 1.5
    assert info["violation"] is True


def test_battery_min_soc_protection():
    # Starting near min_soc = 0.20
    battery = Battery(capacity_kwh=5.0, initial_soc=0.21, min_soc=0.20)
    # Trying to discharge 1.5 kW would drop SOC far below 0.20
    actual_power, next_soc, info = battery.step(1.5)

    assert next_soc >= 0.20
    assert actual_power < 1.5
    assert info["violation"] is True


def test_battery_max_soc_protection():
    # Starting near max_soc = 0.90
    battery = Battery(capacity_kwh=5.0, initial_soc=0.89, max_soc=0.90)
    # Trying to charge 1.5 kW would exceed 0.90
    actual_power, next_soc, info = battery.step(-1.5)

    assert next_soc <= 0.90
    assert abs(actual_power) < 1.5
    assert info["violation"] is True


def test_battery_reset():
    battery = Battery(capacity_kwh=5.0, initial_soc=0.50)
    battery.step(1.0)
    assert battery.soc != 0.50
    assert battery.total_throughput_kwh > 0

    battery.reset(0.60)
    assert battery.soc == 0.60
    assert battery.total_throughput_kwh == 0.0
