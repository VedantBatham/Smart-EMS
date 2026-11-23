"""
Integration tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_dates_endpoint():
    response = client.get("/api/data/dates")
    assert response.status_code == 200
    data = response.json()
    assert "dates" in data
    assert isinstance(data["dates"], list)
    assert len(data["dates"]) > 0


def test_api_system_current():
    response = client.get("/api/system/current")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert "current_solar_kw" in data
    assert "current_load_kw" in data


def test_api_solar_forecast():
    response = client.get("/api/forecast/solar")
    assert response.status_code == 200
    data = response.json()
    assert "actual" in data
    assert "predicted" in data
    assert len(data["actual"]) == len(data["predicted"])


def test_api_load_forecast():
    response = client.get("/api/forecast/load")
    assert response.status_code == 200
    data = response.json()
    assert "actual" in data
    assert "predicted" in data
    assert len(data["actual"]) == len(data["predicted"])


def test_api_models_metrics():
    response = client.get("/api/models/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "solar_forecast" in data
    assert "load_forecast" in data
    assert "r2_score" in data["solar_forecast"]
    assert "mae_kw" in data["load_forecast"]


def test_api_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "battery" in data
    assert "tariff" in data


def test_api_run_comparison():
    payload = {
        "date": "2020-01-01",
        "initial_soc": 0.50,
        "battery_capacity": 5.0,
        "allow_grid_charging": True,
    }
    response = client.post("/api/run/comparison", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "comparison_metrics" in data
    assert "rule_based_metrics" in data
    assert "ppo_metrics" in data
    assert "timeseries" in data
    assert len(data["timeseries"]) == 96  # 1 full day = 96 steps


def test_api_run_comparison_7day():
    payload = {
        "mode": "7day",
        "initial_soc": 0.50,
        "battery_capacity": 5.0,
        "allow_grid_charging": True,
    }
    response = client.post("/api/run/comparison", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "comparison_metrics" in data
    assert len(data["timeseries"]) == 672  # 7 days = 672 steps


def test_api_scenario_sample():
    response = client.get("/api/scenario/sample?day_idx=0")
    assert response.status_code == 200
    data = response.json()
    assert "time_labels" in data
    assert "base_solar_kw" in data
    assert "randomized_solar_kw" in data
    assert "base_load_kw" in data
    assert "randomized_load_kw" in data
    assert "methodology_statement" in data
    assert len(data["time_labels"]) == 96


def test_api_scenario_config():
    response = client.get("/api/scenario/config")
    assert response.status_code == 200
    data = response.json()
    assert "scenario_randomization" in data
    assert "methodology_statement" in data

