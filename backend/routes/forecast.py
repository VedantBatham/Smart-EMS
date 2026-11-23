"""
Forecasting API routes.
"""

from fastapi import APIRouter, Query
from typing import Dict, Any, Optional
from backend.services.forecast_service import BackendForecastService

router = APIRouter(prefix="/api", tags=["Forecasting"])
forecast_service = BackendForecastService()


@router.get("/forecast/solar")
def get_solar_forecast(date: Optional[str] = Query(None, description="Date YYYY-MM-DD")) -> Dict[str, Any]:
    """Returns actual vs predicted solar PV generation."""
    data = forecast_service.get_sample_forecasts(date)
    return {
        "date": date,
        "timestamps": data.get("timestamps", []),
        "actual": data.get("solar_actual", []),
        "predicted": data.get("solar_predicted", []),
    }


@router.get("/forecast/load")
def get_load_forecast(date: Optional[str] = Query(None, description="Date YYYY-MM-DD")) -> Dict[str, Any]:
    """Returns actual vs predicted household electrical load demand."""
    data = forecast_service.get_sample_forecasts(date)
    return {
        "date": date,
        "timestamps": data.get("timestamps", []),
        "actual": data.get("load_actual", []),
        "predicted": data.get("load_predicted", []),
    }


@router.get("/models/metrics")
def get_models_metrics() -> Dict[str, Any]:
    """Returns test evaluation metrics and top feature importances for both models."""
    return forecast_service.get_metrics()
