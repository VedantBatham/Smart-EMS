"""
Comparison API routes for head-to-head evaluation and decision exploration.
"""

from fastapi import APIRouter, Query
from typing import Dict, Any, Optional
from backend.services.comparison_service import BackendComparisonService
from backend.schemas.simulation_schema import SimulationRequest

router = APIRouter(prefix="/api", tags=["Comparison"])
comp_service = BackendComparisonService()


@router.post("/run/comparison")
def run_controller_comparison(request: SimulationRequest) -> Dict[str, Any]:
    """Runs head-to-head comparison between Rule-Based and PPO on identical data."""
    return comp_service.run_comparison(
        date_str=request.date,
        start_date=request.start_date,
        end_date=request.end_date,
        mode=request.mode or "day",
        initial_soc=request.initial_soc or 0.50,
        battery_capacity=request.battery_capacity or 5.0,
        allow_grid_charging=request.allow_grid_charging if request.allow_grid_charging is not None else True,
    )


@router.get("/results/comparison")
def get_comparison_summary() -> Dict[str, Any]:
    """Returns the most recent comparison metrics and performance table."""
    res = comp_service.get_latest_comparison()
    return {
        "comparison_metrics": res["comparison_metrics"],
        "rule_based_metrics": res["rule_based_metrics"],
        "ppo_metrics": res["ppo_metrics"],
        "hybrid_metrics": res.get("hybrid_metrics", {}),
        "selected_best_rl": res.get("selected_best_rl", "hybrid_dqn_ppo"),
        "selected_best_rl_name": res.get("selected_best_rl_name", "Hybrid DQN-PPO"),
        "comparison_table": res.get("comparison_table", {}),
    }


@router.get("/results/rl_benchmark")
def get_rl_benchmark_table() -> Dict[str, Any]:
    """Returns the complete benchmark table (Rule-Based, PPO, Hybrid DQN-PPO) and selected winner."""
    res = comp_service.get_latest_comparison()
    return {
        "selected_best_rl": res.get("selected_best_rl", "hybrid_dqn_ppo"),
        "selected_best_rl_name": res.get("selected_best_rl_name", "Hybrid DQN-PPO"),
        "comparison_table": res.get("comparison_table", {}),
        "rule_based": res.get("rule_based_metrics", {}),
        "ppo": res.get("ppo_metrics", {}),
        "hybrid_dqn_ppo": res.get("hybrid_metrics", {}),
    }


@router.get("/results/timeseries")
def get_comparison_timeseries() -> Dict[str, Any]:
    """Returns the complete synchronized time-series records for all comparison charts."""
    res = comp_service.get_latest_comparison()
    return {"timeseries": res["timeseries"]}


@router.get("/results/decision")
def get_decision_detail(timestamp: str = Query(..., description="Timestamp to inspect")) -> Dict[str, Any]:
    """Returns step-by-step decision details and explanation for the chosen timestamp."""
    return comp_service.get_decision_at_timestamp(timestamp)

