"""
Scenario Generation API routes for hybrid training inspection.
Provides endpoints for comparing base historical profiles with randomized PPO training profiles.
"""

import os
import yaml
import pandas as pd
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query
from environment.scenario_generator import ScenarioGenerator

router = APIRouter(prefix="/api/scenario", tags=["Scenario Generator"])

METHODOLOGY_STATEMENT = (
    "Historical solar-generation and load-demand profiles are used as realistic base trajectories. "
    "During PPO training, controlled stochastic variations are introduced in renewable generation, "
    "load demand, and initial battery SOC to create diverse scenarios and reduce memorization of "
    "fixed historical sequences. The trained controller is finally evaluated on unseen, unmodified "
    "historical data and compared with a conventional rule-based EMS under identical operating conditions."
)

_cached_train_df: Optional[pd.DataFrame] = None
_scenario_gen: Optional[ScenarioGenerator] = None


def get_scenario_generator() -> ScenarioGenerator:
    global _scenario_gen
    if _scenario_gen is None:
        _scenario_gen = ScenarioGenerator(config_path="config.yaml")
    return _scenario_gen


def get_train_df() -> pd.DataFrame:
    global _cached_train_df
    if _cached_train_df is None:
        path = "data/processed/train_data.csv"
        if not os.path.exists(path):
            path = "data/processed/processed_data.csv"
        _cached_train_df = pd.read_csv(path)
    return _cached_train_df


@router.get("/sample")
def get_scenario_sample(
    day_idx: Optional[int] = Query(None, description="Optional historical day index to sample"),
    seed: Optional[int] = Query(None, description="Optional random seed for reproducible scenario generation"),
) -> Dict[str, Any]:
    """
    Generates a side-by-side comparison of a historical base day vs. its randomized PPO training scenario.
    """
    train_df = get_train_df()
    sg = get_scenario_generator()
    if seed is not None:
        import numpy as np
        sg.rng = np.random.default_rng(seed)

    sample = sg.generate_comparison_sample(train_df, day_idx=day_idx)
    sample["methodology_statement"] = METHODOLOGY_STATEMENT
    return sample


@router.get("/config")
def get_scenario_config() -> Dict[str, Any]:
    """
    Returns current scenario randomization hyper-parameters from config.yaml.
    """
    cfg_path = "config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            return {
                "scenario_randomization": cfg.get("scenario_randomization", {}),
                "methodology_statement": METHODOLOGY_STATEMENT,
            }
    return {"scenario_randomization": {}, "methodology_statement": METHODOLOGY_STATEMENT}
