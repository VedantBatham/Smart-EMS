"""
Feature Engineering Module
Constructs lag features, rolling statistics, cyclical temporal encodings,
and deterministic tariff schedule features without data leakage.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from energy_system.tariff import TariffModel


def create_engineered_features(
    df: pd.DataFrame,
    tariff_model: TariffModel = None,
) -> pd.DataFrame:
    """
    Computes time-series lags, rolling window statistics, temporal encodings,
    and forward-looking tariff features.
    
    Lags (15-min intervals):
      lag_1 = 15m (1 step)
      lag_2 = 30m (2 steps)
      lag_4 = 1h (4 steps)
      lag_8 = 2h (8 steps)
      lag_12 = 3h (12 steps)
      lag_96 = 24h (96 steps, same time yesterday)
    """
    if tariff_model is None:
        tariff_model = TariffModel()

    feat_df = df.copy()

    # Ensure timestamp is datetime
    if not np.issubdtype(feat_df["timestamp"].dtype, np.datetime64):
        feat_df["timestamp"] = pd.to_datetime(feat_df["timestamp"])

    # 1. Base Temporal & Cyclical Features
    feat_df["hour"] = feat_df["timestamp"].dt.hour
    feat_df["minute"] = feat_df["timestamp"].dt.minute
    feat_df["day_of_week"] = feat_df["timestamp"].dt.dayofweek
    feat_df["month"] = feat_df["timestamp"].dt.month
    feat_df["weekend"] = (feat_df["day_of_week"] >= 5).astype(int)

    # Cyclical sin/cos encodings
    feat_df["sin_hour"] = np.sin(2 * np.pi * (feat_df["hour"] + feat_df["minute"] / 60.0) / 24.0)
    feat_df["cos_hour"] = np.cos(2 * np.pi * (feat_df["hour"] + feat_df["minute"] / 60.0) / 24.0)
    feat_df["sin_month"] = np.sin(2 * np.pi * feat_df["month"] / 12.0)
    feat_df["cos_month"] = np.cos(2 * np.pi * feat_df["month"] / 12.0)

    # 2. Lag Features for Solar and Load
    lags = [1, 2, 4, 8, 12, 96]
    for lag in lags:
        feat_df[f"solar_lag_{lag}"] = feat_df["solar_kw"].shift(lag)
        feat_df[f"load_lag_{lag}"] = feat_df["load_kw"].shift(lag)

    # 3. Rolling Window Statistics (1h=4, 2h=8, 6h=24, 24h=96)
    # Using shift(1) to avoid target leakage with current observation
    rolling_windows = {
        "1h": 4,
        "2h": 8,
        "6h": 24,
        "24h": 96,
    }
    for label, window in rolling_windows.items():
        feat_df[f"solar_rolling_mean_{label}"] = (
            feat_df["solar_kw"].shift(1).rolling(window=window, min_periods=1).mean()
        )
        feat_df[f"load_rolling_mean_{label}"] = (
            feat_df["load_kw"].shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Rolling standard deviations
    feat_df["solar_rolling_std_1h"] = (
        feat_df["solar_kw"].shift(1).rolling(window=4, min_periods=1).std().fillna(0.0)
    )
    feat_df["solar_rolling_std_6h"] = (
        feat_df["solar_kw"].shift(1).rolling(window=24, min_periods=1).std().fillna(0.0)
    )
    feat_df["load_rolling_std_1h"] = (
        feat_df["load_kw"].shift(1).rolling(window=4, min_periods=1).std().fillna(0.0)
    )
    feat_df["load_rolling_std_6h"] = (
        feat_df["load_kw"].shift(1).rolling(window=24, min_periods=1).std().fillna(0.0)
    )

    # 4. Deterministic Electricity Tariff Features
    feat_df["current_tariff"] = feat_df.apply(
        lambda row: tariff_model.get_tariff(int(row["hour"]), int(row["minute"])), axis=1
    )
    feat_df["next_hour_avg_tariff"] = feat_df.apply(
        lambda row: tariff_model.get_next_hour_avg_tariff(int(row["hour"]), int(row["minute"])), axis=1
    )
    feat_df["next_hour_max_tariff"] = feat_df.apply(
        lambda row: tariff_model.get_next_hour_max_tariff(int(row["hour"]), int(row["minute"])), axis=1
    )

    # 5. Future Targets and Summary Horizons (Shifted backwards for training ML models)
    feat_df["target_solar_next_15m"] = feat_df["solar_kw"].shift(-1)
    feat_df["target_load_next_15m"] = feat_df["load_kw"].shift(-1)

    # Rolling future summaries (Next 1h = 4 steps, Next 2h = 8 steps)
    # Using inverted rolling window or shifted mean
    # Next 1 hour average: steps t+1 to t+4
    indexer_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=4)
    feat_df["future_solar_1h_avg"] = feat_df["solar_kw"].shift(-1).rolling(window=indexer_1h, min_periods=1).mean()
    feat_df["future_load_1h_avg"] = feat_df["load_kw"].shift(-1).rolling(window=indexer_1h, min_periods=1).mean()

    indexer_2h = pd.api.indexers.FixedForwardWindowIndexer(window_size=8)
    feat_df["future_solar_2h_avg"] = feat_df["solar_kw"].shift(-1).rolling(window=indexer_2h, min_periods=1).mean()
    feat_df["future_load_2h_avg"] = feat_df["load_kw"].shift(-1).rolling(window=indexer_2h, min_periods=1).mean()
    feat_df["future_load_2h_peak"] = feat_df["load_kw"].shift(-1).rolling(window=indexer_2h, min_periods=1).max()

    return feat_df


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Returns the feature column names used for training solar and load XGBoost models.
    """
    solar_features = [
        "solar_kw",
        "hour",
        "minute",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "solar_lag_1",
        "solar_lag_2",
        "solar_lag_4",
        "solar_lag_8",
        "solar_lag_12",
        "solar_lag_96",
        "solar_rolling_mean_1h",
        "solar_rolling_mean_2h",
        "solar_rolling_mean_6h",
        "solar_rolling_mean_24h",
        "solar_rolling_std_1h",
        "solar_rolling_std_6h",
    ]

    load_features = [
        "load_kw",
        "hour",
        "minute",
        "day_of_week",
        "weekend",
        "month",
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "current_tariff",
        "next_hour_avg_tariff",
        "next_hour_max_tariff",
        "load_lag_1",
        "load_lag_2",
        "load_lag_4",
        "load_lag_8",
        "load_lag_12",
        "load_lag_96",
        "load_rolling_mean_1h",
        "load_rolling_mean_2h",
        "load_rolling_mean_6h",
        "load_rolling_mean_24h",
        "load_rolling_std_1h",
        "load_rolling_std_6h",
    ]

    return solar_features, load_features
