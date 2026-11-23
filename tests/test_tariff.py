"""
Unit tests for Electricity Tariff Model (Mathematical TOU Schedule)
Off-peak: ₹0.08 / kWh (00:00-17:00 & 22:00-24:00)
Peak: ₹0.30 / kWh (17:00-22:00)
"""

from energy_system.tariff import TariffModel


def test_default_tariff_rates():
    tariff = TariffModel()

    # 00:00 - 06:00 -> ₹0.08/kWh
    assert tariff.get_tariff(0, 0) == 0.08
    assert tariff.get_tariff(3, 30) == 0.08
    assert tariff.get_tariff(5, 45) == 0.08

    # 06:00 - 17:00 -> ₹0.08/kWh
    assert tariff.get_tariff(6, 0) == 0.08
    assert tariff.get_tariff(12, 0) == 0.08
    assert tariff.get_tariff(16, 45) == 0.08

    # 17:00 - 22:00 -> ₹0.30/kWh
    assert tariff.get_tariff(17, 0) == 0.30
    assert tariff.get_tariff(19, 30) == 0.30
    assert tariff.get_tariff(21, 45) == 0.30

    # 22:00 - 24:00 -> ₹0.08/kWh
    assert tariff.get_tariff(22, 0) == 0.08
    assert tariff.get_tariff(23, 45) == 0.08


def test_tariff_cost_calculation():
    tariff = TariffModel()
    # 2 kW for 0.25 h at ₹0.30/kWh = 2 * 0.25 * 0.30 = ₹0.15
    cost = tariff.get_cost(power_kw=2.0, duration_hours=0.25, hour=18, minute=0)
    assert round(cost, 4) == 0.15


def test_future_tariff_lookahead():
    tariff = TariffModel()
    # At 16:30, looking ahead 4 steps (16:45, 17:00, 17:15, 17:30)
    # 16:45 is ₹0.08, 17:00 is ₹0.30, 17:15 is ₹0.30, 17:30 is ₹0.30
    future = tariff.get_future_tariffs(hour=16, minute=30, horizon_steps=4, step_minutes=15)
    assert future == [0.08, 0.30, 0.30, 0.30]

    max_t = tariff.get_next_hour_max_tariff(hour=16, minute=30)
    assert max_t == 0.30
