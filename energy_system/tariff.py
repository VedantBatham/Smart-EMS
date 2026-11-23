"""
Electricity Tariff Model
Configurable Time-of-Use (ToU) electricity pricing model.
Future tariff values are deterministically known from the predefined schedule.
"""

from typing import List, Dict, Any, Optional
import yaml
import os


class TariffModel:
    """
    Time-of-Use (ToU) Tariff Model.
    
    Default Schedule:
      00:00 - 06:00: ₹3.0 / kWh (Off-Peak Night)
      06:00 - 17:00: ₹5.0 / kWh (Standard Day)
      17:00 - 22:00: ₹10.0 / kWh (Peak Evening)
      22:00 - 24:00: ₹4.0 / kWh (Late Evening Off-Peak)
    """

    def __init__(self, config_path: Optional[str] = "config.yaml", custom_slots: Optional[List[Dict[str, Any]]] = None):
        self.slots = []
        self.currency = "₹"

        if custom_slots is not None:
            self.slots = custom_slots
        elif config_path is not None and os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                tariff_cfg = cfg.get("tariff", {})
                self.slots = tariff_cfg.get("slots", [])
                self.currency = cfg.get("simulation", {}).get("currency_symbol", "₹")
        else:
            # Fallback default slots (Time-of-Use in INR: ₹0.08 off-peak, ₹0.30 peak)
            self.slots = [
                {"start_hour": 0, "end_hour": 17, "rate": 0.08, "label": "Off-Peak Hours"},
                {"start_hour": 17, "end_hour": 22, "rate": 0.30, "label": "Evening Peak Hours"},
                {"start_hour": 22, "end_hour": 24, "rate": 0.08, "label": "Off-Peak Hours"},
            ]

    def get_tariff(self, hour: int, minute: int = 0) -> float:
        """
        Returns electricity tariff rate (in ₹/kWh) for a given hour and minute.
        """
        decimal_hour = hour + minute / 60.0
        # Handle wrap around or boundary at 24.0
        if decimal_hour >= 24.0:
            decimal_hour = decimal_hour % 24.0

        for slot in self.slots:
            if slot["start_hour"] <= decimal_hour < slot["end_hour"]:
                return float(slot["rate"])

        # Default fallback
        return 0.08

    def get_future_tariffs(self, hour: int, minute: int, horizon_steps: int = 8, step_minutes: int = 15) -> List[float]:
        """
        Returns list of deterministic future tariffs for the given horizon.
        Default 8 steps = 2 hours ahead.
        """
        rates = []
        total_minutes = hour * 60 + minute
        for s in range(1, horizon_steps + 1):
            fut_min_total = (total_minutes + s * step_minutes) % (24 * 60)
            fut_h = fut_min_total // 60
            fut_m = fut_min_total % 60
            rates.append(self.get_tariff(fut_h, fut_m))
        return rates

    def get_next_hour_avg_tariff(self, hour: int, minute: int) -> float:
        """Average tariff over the next 1 hour (4 steps of 15 min)."""
        future_rates = self.get_future_tariffs(hour, minute, horizon_steps=4, step_minutes=15)
        return float(sum(future_rates) / len(future_rates)) if future_rates else self.get_tariff(hour, minute)

    def get_next_hour_max_tariff(self, hour: int, minute: int) -> float:
        """Maximum tariff over the next 1 hour."""
        future_rates = self.get_future_tariffs(hour, minute, horizon_steps=4, step_minutes=15)
        return float(max(future_rates)) if future_rates else self.get_tariff(hour, minute)

    def get_cost(self, power_kw: float, duration_hours: float, hour: int, minute: int = 0) -> float:
        """
        Calculates electricity cost in currency units.
        Cost = power_kw * duration_hours * tariff_rate
        """
        rate = self.get_tariff(hour, minute)
        return float(max(0.0, power_kw) * duration_hours * rate)

    def get_average_tariff(self) -> float:
        """Calculates time-weighted average tariff rate over 24 hours."""
        if not self.slots:
            return 0.15
        total_rate_hours = sum(slot["rate"] * (slot["end_hour"] - slot["start_hour"]) for slot in self.slots)
        return float(total_rate_hours / 24.0)

    def is_peak_hour(self, hour: int, minute: int = 0) -> bool:
        """Determines if the given time falls within the peak tariff window."""
        t = self.get_tariff(hour, minute)
        max_rate = max((slot["rate"] for slot in self.slots), default=0.30)
        return t >= (max_rate * 0.85)

    def get_tariff_schedule(self) -> List[Dict[str, Any]]:
        """Returns the full configured tariff schedule."""
        return self.slots
