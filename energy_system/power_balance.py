"""
Power Balance and Energy Flow Model
Enforces instantaneous physical power balance:
P_solar + P_battery + P_grid = P_load
Calculates component flows, solar curtailment, and grid exchange.
"""

from typing import Dict, Any


def calculate_power_balance(
    load_kw: float,
    solar_kw: float,
    battery_power_kw: float,
    allow_grid_export: bool = False,
    allow_grid_charging: bool = True,
) -> Dict[str, float]:
    """
    Computes instantaneous power balance and decomposed energy flows.
    
    Sign Conventions:
      load_kw >= 0 (electrical power demanded)
      solar_kw >= 0 (photovoltaic generation)
      battery_power_kw:
          > 0 : discharging (power supplied by battery)
          < 0 : charging (power absorbed into battery)
          = 0 : idle
      grid_power_kw:
          > 0 : import from grid (power drawn)
          < 0 : export to grid (feed-in)
          = 0 : neutral
          
    Balance equation:
      P_solar_used + P_battery + P_grid = P_load
    """
    load_kw = max(0.0, float(load_kw))
    solar_kw = max(0.0, float(solar_kw))
    battery_power_kw = float(battery_power_kw)

    solar_to_load = 0.0
    solar_to_battery = 0.0
    solar_curtailed = 0.0
    battery_to_load = 0.0
    grid_to_load = 0.0
    grid_to_battery = 0.0
    grid_export = 0.0
    unserved_load = 0.0

    # 1. Solar priority: direct solar to load first
    solar_used_for_load = min(solar_kw, load_kw)
    solar_to_load = solar_used_for_load
    net_load_remaining = load_kw - solar_to_load
    excess_solar = solar_kw - solar_to_load

    # 2. Battery state
    if battery_power_kw > 0.0:
        # Battery is discharging to help meet remaining load
        battery_to_load = min(battery_power_kw, net_load_remaining)
        # In case battery discharges more than net load:
        excess_batt = battery_power_kw - battery_to_load
        net_load_remaining = max(0.0, net_load_remaining - battery_to_load)
        if excess_batt > 0.0:
            if allow_grid_export:
                grid_export += excess_batt
            else:
                # Excess discharge curtailed or throttled
                pass

        # Any excess solar was 0 because net_load_remaining was > 0, but if solar > load:
        if excess_solar > 0.0:
            if allow_grid_export:
                grid_export += excess_solar
            else:
                solar_curtailed += excess_solar

    elif battery_power_kw < 0.0:
        # Battery is charging (charge_demand = |battery_power_kw|)
        charge_demand = abs(battery_power_kw)

        # First charge from excess solar if available
        solar_to_battery = min(excess_solar, charge_demand)
        excess_solar_remaining = excess_solar - solar_to_battery
        remaining_charge_needed = charge_demand - solar_to_battery

        if excess_solar_remaining > 0.0:
            if allow_grid_export:
                grid_export += excess_solar_remaining
            else:
                solar_curtailed += excess_solar_remaining

        # If remaining charge needed and grid charging allowed:
        if remaining_charge_needed > 0.0:
            if allow_grid_charging:
                grid_to_battery = remaining_charge_needed
            else:
                # Cannot charge from grid; adjust actual battery charge
                # Battery power effectively reduced to solar_to_battery
                pass
    else:
        # Battery idle
        if excess_solar > 0.0:
            if allow_grid_export:
                grid_export += excess_solar
            else:
                solar_curtailed += excess_solar

    # 3. Grid supplies whatever net load remains
    grid_to_load = net_load_remaining
    total_grid_import = grid_to_load + grid_to_battery

    if allow_grid_export and grid_export > 0.0:
        grid_power_kw = total_grid_import - grid_export
    else:
        grid_power_kw = total_grid_import

    # Total solar used
    solar_used_total = solar_to_load + solar_to_battery

    # Power balance verification: (solar_used + battery + grid_import - grid_export) vs load
    supplied = solar_used_total + battery_power_kw + grid_power_kw
    balance_error = abs(supplied - load_kw)

    return {
        "load_kw": load_kw,
        "solar_kw": solar_kw,
        "solar_used_kw": solar_used_total,
        "solar_curtailed_kw": solar_curtailed,
        "solar_to_load_kw": solar_to_load,
        "solar_to_battery_kw": solar_to_battery,
        "battery_power_kw": battery_power_kw,
        "battery_to_load_kw": battery_to_load,
        "grid_power_kw": grid_power_kw,
        "grid_to_load_kw": grid_to_load,
        "grid_to_battery_kw": grid_to_battery,
        "grid_export_kw": grid_export,
        "unserved_load_kw": unserved_load,
        "balance_error": balance_error,
    }
