"""
XGBoost Regressor for Short-Term Solar PV Generation Forecasting.
Predicts next 15-minute generation: P_solar,t+1
Saves trained model and feature importance to models/solar_xgb.pkl
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


def train_solar_forecaster(
    train_path: str = "data/processed/train_data.csv",
    val_path: str = "data/processed/validation_data.csv",
    model_output_path: str = "models/solar_xgb.pkl",
) -> dict:
    """
    Trains XGBoost regressor for solar forecasting with chronological validation.
    Nighttime zero-generation intervals are handled without division-by-zero MAPE artifacts.
    """
    logger.info("Loading training and validation sets for Solar forecaster...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    from preprocessing.feature_engineering import get_feature_columns
    solar_features, _ = get_feature_columns()

    # Drop any lingering rows with NaN in features or target
    train_clean = train_df.dropna(subset=solar_features + ["target_solar_next_15m"])
    val_clean = val_df.dropna(subset=solar_features + ["target_solar_next_15m"])

    X_train = train_clean[solar_features]
    y_train = train_clean["target_solar_next_15m"]

    X_val = val_clean[solar_features]
    y_val = val_clean["target_solar_next_15m"]

    logger.info(f"Training Solar XGBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features...")
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
    val_preds = np.clip(model.predict(X_val), 0.0, None)
    mae = float(mean_absolute_error(y_val, val_preds))
    rmse = float(root_mean_squared_error(y_val, val_preds))
    r2 = float(r2_score(y_val, val_preds))

    # Daylight-only MAE (where y_val > 0.05 kW)
    daylight_mask = y_val > 0.05
    daylight_mae = float(mean_absolute_error(y_val[daylight_mask], val_preds[daylight_mask])) if daylight_mask.sum() > 0 else mae

    logger.info(f"Solar Model Validation -> MAE: {mae:.4f} kW, RMSE: {rmse:.4f} kW, R²: {r2:.4f}, Daylight MAE: {daylight_mae:.4f} kW")

    # Feature importances
    feature_importance = dict(zip(solar_features, [float(x) for x in model.feature_importances_]))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:10])

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    payload = {
        "model": model,
        "features": solar_features,
        "metrics": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "daylight_mae": round(daylight_mae, 4),
        },
        "top_features": sorted_importance,
    }
    joblib.dump(payload, model_output_path)
    logger.info(f"Solar model saved successfully to {model_output_path}")

    return payload["metrics"]


if __name__ == "__main__":
    train_solar_forecaster()
