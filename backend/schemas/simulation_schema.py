"""
Pydantic Schemas for API Requests and Responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class TariffSlotSchema(BaseModel):
    start_hour: int
    end_hour: int
    rate: float
    label: str


class BatteryConfigSchema(BaseModel):
    capacity_kwh: float = Field(5.0, ge=1.0, le=50.0, description="Battery capacity in kWh")
    initial_soc: float = Field(0.50, ge=0.0, le=1.0, description="Initial State of Charge")
    min_soc: float = Field(0.20, ge=0.0, le=0.5, description="Minimum State of Charge")
    max_soc: float = Field(0.90, ge=0.5, le=1.0, description="Maximum State of Charge")
    max_charge_kw: float = Field(1.5, ge=0.1, le=20.0, description="Max charging rate in kW")
    max_discharge_kw: float = Field(1.5, ge=0.1, le=20.0, description="Max discharging rate in kW")
    allow_grid_charging: bool = Field(True, description="Enable battery charging from grid")
    allow_grid_export: bool = Field(False, description="Enable excess solar export to grid")


class SimulationRequest(BaseModel):
    date: Optional[str] = Field(None, description="Selected date YYYY-MM-DD")
    start_date: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD")
    mode: Optional[str] = Field("day", description="Simulation mode: 'day', '7day', or '14day'")
    initial_soc: Optional[float] = Field(0.50, ge=0.0, le=1.0)
    battery_capacity: Optional[float] = Field(5.0, ge=1.0, le=50.0)
    allow_grid_charging: Optional[bool] = Field(True)


class PlaybackStepRequest(BaseModel):
    timestamp: str
    controller: str = "Forecast-Assisted PPO"
