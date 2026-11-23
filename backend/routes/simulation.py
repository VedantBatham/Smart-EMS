"""
Simulation API routes for running individual controllers.
"""

from fastapi import APIRouter
from typing import Dict, Any
from backend.services.simulation_service import BackendSimulationService
from backend.schemas.simulation_schema import SimulationRequest

router = APIRouter(prefix="/api/run", tags=["Simulation"])
sim_service = BackendSimulationService()


@router.post("/rule-based")
def run_rule_based_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """Runs Rule-Based EMS on selected date/range."""
    return sim_service.run_single_simulation(
        controller_name="Rule-Based EMS",
        date_str=request.date,
        start_date=request.start_date,
        end_date=request.end_date,
        mode=request.mode or "day",
        initial_soc=request.initial_soc or 0.50,
        battery_capacity=request.battery_capacity or 5.0,
        allow_grid_charging=request.allow_grid_charging if request.allow_grid_charging is not None else True,
    )


@router.post("/ppo")
def run_ppo_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """Runs Forecast-Assisted PPO EMS on selected date/range."""
    return sim_service.run_single_simulation(
        controller_name="Forecast-Assisted PPO",
        date_str=request.date,
        start_date=request.start_date,
        end_date=request.end_date,
        mode=request.mode or "day",
        initial_soc=request.initial_soc or 0.50,
        battery_capacity=request.battery_capacity or 5.0,
        allow_grid_charging=request.allow_grid_charging if request.allow_grid_charging is not None else True,
    )


@router.post("/hybrid")
def run_hybrid_simulation(request: SimulationRequest) -> Dict[str, Any]:
    """Runs Hybrid DQN-PPO EMS on selected date/range."""
    return sim_service.run_single_simulation(
        controller_name="Hybrid DQN-PPO",
        date_str=request.date,
        start_date=request.start_date,
        end_date=request.end_date,
        mode=request.mode or "day",
        initial_soc=request.initial_soc or 0.50,
        battery_capacity=request.battery_capacity or 5.0,
        allow_grid_charging=request.allow_grid_charging if request.allow_grid_charging is not None else True,
    )
