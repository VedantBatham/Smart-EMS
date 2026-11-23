"""
XGBoost Regressor for Short-Term Household Load Demand Forecasting.
Predicts next 15-minute electrical demand: P_load,t+1
Saves trained model and feature importance to models/load_xgb.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_load_forecaster(
    train_path: str = "data/processed/train_data.csv",
    val_path: str = "data/processed/validation_data.csv",
    model_output_path: str = "models/load_xgb.pkl",
) -> dict:
    """
    Trains XGBoost regressor for residential load forecasting with chronological validation.
    Calculates MAE, RMSE, MAPE (where valid), and R².
    """
    logger.info("Loading training and validation sets for Load forecaster...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    from preprocessing.feature_engineering import get_feature_columns
    _, load_features = get_feature_columns()

    train_clean = train_df.dropna(subset=load_features + ["target_load_next_15m"])
    val_clean = val_df.dropna(subset=load_features + ["target_load_next_15m"])

    X_train = train_clean[load_features]
    y_train = train_clean["target_load_next_15m"]

    X_val = val_clean[load_features]
    y_val = val_clean["target_load_next_15m"]

    logger.info(f"Training Load XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features...")
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Validation evaluation
    val_preds = np.clip(model.predict(X_val), 0.05, None)
    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(root_mean_squared_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))

    # MAPE where load > 0.1 kW to avoid near-zero distortions
    valid_mask = y_val > 0.10
    mape = float(np.mean(np.abs((y_val[valid_mask] - val_preds[valid_mask]) / y_val[valid_mask])) * 100.0)

    logger.info(f"Load Model Validation -> MAE: {mae:.4f} kW, RMSE: {rmse:.4f} kW, MAPE: {mape:.2f}%, R²: {r2:.4f}")

    # Feature importances
    feature_importance = dict(zip(load_features, [float(x) for x in model.feature_importances_]))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:10])

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    payload = {
        "model": model,
        "features": load_features,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape, 2),
            "r2": round(r2, 4),
        },
        "top_features": sorted_importance,
    }
    joblib.dump(payload, model_output_path)
    logger.info(f"Load model saved successfully to {model_output_path}")

    return payload["metrics"]


if __name__ == "__main__":
    train_load_forecaster()
