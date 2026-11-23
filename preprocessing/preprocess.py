"""
Data Preprocessing Pipeline
Parses timestamps, sorts chronologically, removes duplicates, handles missing values,
enforces 15-minute resampling, generates base temporal features, and creates
chronological train / validation / test splits.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def clean_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw time series data:
      - Converts timestamp to datetime
      - Drops duplicates and sorts chronologically
      - Resamples to uniform 15-minute frequency ('15min')
      - Imputes missing values via linear interpolation + forward/backward fill
      - Clips invalid negative solar or load values to 0.0
      - Extracts base temporal features: hour, minute, day_of_week, month, weekend
    """
    df = df.copy()

    # Ensure timestamp column
    if "timestamp" not in df.columns:
        # Check if first column is datetime
        df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    # Resample to 15-minute intervals
    df = df.resample("15min").mean()

    # Interpolate missing values smoothly
    df = df.interpolate(method="time").bfill().ffill()

    # Ensure non-negative physical values
    df["solar_kw"] = df["solar_kw"].clip(lower=0.0).round(4)
    df["load_kw"] = df["load_kw"].clip(lower=0.05).round(4)

    df.reset_index(inplace=True)

    # Base temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Enforce column order
    cols = ["timestamp", "solar_kw", "load_kw", "hour", "minute", "day_of_week", "month", "weekend"]
    return df[cols]


def split_chronologically(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time series chronologically into train, validation, and test partitions
    to strictly prevent future data leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Chronological Split: Total={n} records. "
        f"Train={len(train_df)} ({len(train_df)/96:.1f} days), "
        f"Validation={len(val_df)} ({len(val_df)/96:.1f} days), "
        f"Test={len(test_df)} ({len(test_df)/96:.1f} days)"
    )

    return train_df, val_df, test_df


def run_preprocessing_pipeline(
    raw_path: str = "data/raw/household_data.csv",
    processed_dir: str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes end-to-end preprocessing and saves energy_data.csv, train_data.csv,
    validation_data.csv, and test_data.csv.
    """
    os.makedirs(processed_dir, exist_ok=True)
    raw_df = pd.read_csv(raw_path)
    cleaned_df = clean_and_resample(raw_df)

    processed_file = os.path.join(processed_dir, "energy_data.csv")
    cleaned_df.to_csv(processed_file, index=False)
    logger.info(f"Processed base energy dataset saved to {processed_file}")

    train_df, val_df, test_df = split_chronologically(cleaned_df)

    train_df.to_csv(os.path.join(processed_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "validation_data.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_data.csv"), index=False)

    return cleaned_df, train_df, val_df, test_df


if __name__ == "__main__":
    from preprocessing.load_data import load_or_fetch_opsd_data
    load_or_fetch_opsd_data(days=180)
    cleaned, train, val, test = run_preprocessing_pipeline()
    print("Preprocessing completed successfully.")
