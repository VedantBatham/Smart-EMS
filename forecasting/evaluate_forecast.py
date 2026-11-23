"""
Forecast Evaluation Module
Evaluates trained XGBoost solar and load models on held-out test data,
calculates standardized metrics (MAE, RMSE, MAPE, R²), and generates comparison plots.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_forecasting_models(
    test_path: str = "data/processed/test_data.csv",
    solar_model_path: str = "models/solar_xgb.pkl",
    load_model_path: str = "models/load_xgb.pkl",
    output_metrics_path: str = "results/metrics/forecast_metrics.json",
    plots_dir: str = "results/plots",
) -> dict:
    """
    Evaluates both XGBoost models on the test dataset.
    """
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    test_df = pd.read_csv(test_path)
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    solar_data = joblib.load(solar_model_path)
    load_data = joblib.load(load_model_path)

    solar_model = solar_data["model"]
    solar_features = solar_data["features"]

    load_model = load_data["model"]
    load_features = load_data["features"]

    # Filter clean rows
    clean_test = test_df.dropna(subset=solar_features + load_features + ["target_solar_next_15m", "target_load_next_15m"])

    # 1. Solar Predictions
    solar_actual = clean_test["target_solar_next_15m"].values
    solar_preds = np.clip(solar_model.predict(clean_test[solar_features]), 0.0, None)

    solar_mae = float(mean_absolute_error(solar_actual, solar_preds))
    solar_rmse = float(root_mean_squared_error(solar_actual, solar_preds))
    solar_r2 = float(r2_score(solar_actual, solar_preds))

    # 2. Load Predictions
    load_actual = clean_test["target_load_next_15m"].values
    load_preds = np.clip(load_model.predict(clean_test[load_features]), 0.05, None)

    load_mae = float(mean_absolute_error(load_actual, load_preds))
    load_rmse = float(root_mean_squared_error(load_actual, load_preds))
    load_r2 = float(r2_score(load_actual, load_preds))
    load_valid = load_actual > 0.10
    load_mape = float(np.mean(np.abs((load_actual[load_valid] - load_preds[load_valid]) / load_actual[load_valid])) * 100.0)

    metrics = {
        "solar_forecast": {
            "mae_kw": round(solar_mae, 4),
            "rmse_kw": round(solar_rmse, 4),
            "r2_score": round(solar_r2, 4),
            "test_samples": len(solar_actual),
            "top_features": solar_data.get("top_features", {}),
        },
        "load_forecast": {
            "mae_kw": round(load_mae, 4),
            "rmse_kw": round(load_rmse, 4),
            "mape_pct": round(load_mape, 2),
            "r2_score": round(load_r2, 4),
            "test_samples": len(load_actual),
            "top_features": load_data.get("top_features", {}),
        },
    }

    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved forecasting test metrics to {output_metrics_path}")

    # Generate Evaluation Plots for a representative 3-day test slice (288 steps)
    plot_slice = clean_test.iloc[96 : 96 + 288].copy()
    time_index = plot_slice["timestamp"]

    # Solar Forecast Plot
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_index, plot_slice["target_solar_next_15m"], label="Actual Solar PV (kW)", color="#eab308", lw=2)
    plt.plot(time_index, np.clip(solar_model.predict(plot_slice[solar_features]), 0, None),
             label="XGBoost Predicted Solar (kW)", color="#ca8a04", linestyle="--", lw=1.8)
    plt.title(f"Solar PV Generation Forecasting — Test Period (R² = {solar_r2:.3f}, MAE = {solar_mae:.3f} kW)", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Power (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    solar_plot_path = os.path.join(plots_dir, "solar_forecast.png")
    plt.savefig(solar_plot_path)
    plt.close()

    # Load Forecast Plot
    plt.figure(figsize=(12, 4.5), dpi=150)
    plt.plot(time_index, plot_slice["target_load_next_15m"], label="Actual Household Load (kW)", color="#06b6d4", lw=2)
    plt.plot(time_index, np.clip(load_model.predict(plot_slice[load_features]), 0.05, None),
             label="XGBoost Predicted Load (kW)", color="#0284c7", linestyle="--", lw=1.8)
    plt.title(f"Household Load Demand Forecasting — Test Period (R² = {load_r2:.3f}, MAE = {load_mae:.3f} kW, MAPE = {load_mape:.1f}%)", fontsize=13, fontweight="bold")
    plt.xlabel("Date & Time")
    plt.ylabel("Power (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    load_plot_path = os.path.join(plots_dir, "load_forecast.png")
    plt.savefig(load_plot_path)
    plt.close()

    logger.info(f"Forecast plots saved to {solar_plot_path} and {load_plot_path}")
    return metrics


if __name__ == "__main__":
    evaluate_forecasting_models()
