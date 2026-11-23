"""
System and Configuration API routes.
"""

from fastapi import APIRouter
from typing import Dict, Any
import yaml
import os

from backend.services.simulation_service import BackendSimulationService
from backend.schemas.simulation_schema import BatteryConfigSchema

router = APIRouter(prefix="/api", tags=["System & Config"])
sim_service = BackendSimulationService()


@router.get("/system/current")
def get_current_system_status() -> Dict[str, Any]:
    """
    Returns live overview snapshot (most recent interval or initial test point).
    """
    dates_data = sim_service.get_available_dates()
    dates_list = dates_data.get("dates", [])
    sample_date = dates_data.get("default_date") or (dates_list[0] if dates_list else "2020-01-01")
    slice_df = sim_service.get_data_slice(date_str=sample_date)
    first_row = slice_df.iloc[0] if not slice_df.empty else {}

    return {
        "status": "ONLINE",
        "current_date": sample_date,
        "current_solar_kw": float(first_row.get("solar_kw", 0.0)),
        "current_load_kw": float(first_row.get("load_kw", 1.2)),
        "current_battery_soc": 0.50,
        "current_tariff": float(first_row.get("current_tariff", 3.0)),
        "active_controller": "Forecast-Assisted PPO",
    }


@router.get("/data/dates")
def get_available_dates():
    """Returns all dates present in the processed dataset with metadata."""
    return sim_service.get_available_dates()


@router.get("/config")
def get_current_config() -> Dict[str, Any]:
    """Returns currently configured parameters from config.yaml."""
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}


@router.post("/config")
def update_config(config: BatteryConfigSchema) -> Dict[str, Any]:
    """Updates battery configuration parameters in config.yaml."""
    cfg_path = "config.yaml"
    data = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}

    data["battery"] = {
        "capacity_kwh": config.capacity_kwh,
        "initial_soc": config.initial_soc,
        "min_soc": config.min_soc,
        "max_soc": config.max_soc,
        "max_charge_kw": config.max_charge_kw,
        "max_discharge_kw": config.max_discharge_kw,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "target_terminal_soc": 0.50,
    }
    data["grid"] = {
        "allow_grid_charging": config.allow_grid_charging,
        "allow_grid_export": config.allow_grid_export,
    }

    with open(cfg_path, "w") as f:
        yaml.safe_dump(data, f)

    return {"message": "Configuration updated successfully", "battery": data["battery"]}
