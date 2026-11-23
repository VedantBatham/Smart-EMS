"""
Script to execute complete dataset acquisition, preprocessing, feature engineering,
and train/validation/test chronological splitting.
"""

import os
import sys
import logging

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.load_data import load_or_fetch_opsd_data
from preprocessing.preprocess import clean_and_resample, split_chronologically
from preprocessing.feature_engineering import create_engineered_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Acquiring / Generating OPSD Household Dataset ===")
    raw_df = load_or_fetch_opsd_data(
        raw_path="data/raw/household_data.csv",
        try_download=False,  # Set to False to reliably load calibrated 180-day microgrid data
        days=180,
    )
    logger.info(f"Raw dataset shape: {raw_df.shape}")

    logger.info("=== STEP 2: Cleaning and Resampling to 15-Minute Intervals ===")
    cleaned_df = clean_and_resample(raw_df)
    logger.info(f"Cleaned dataset shape: {cleaned_df.shape}")

    logger.info("=== STEP 3: Engineering Lags, Rolling Statistics, Temporal & Tariff Features ===")
    engineered_df = create_engineered_features(cleaned_df)
    # Forward fill remaining initial lag NaNs to ensure no missing values
    engineered_df = engineered_df.bfill().ffill()
    logger.info(f"Engineered dataset shape: {engineered_df.shape}")

    logger.info("=== STEP 4: Chronological Splitting (Train: 70%, Val: 15%, Test: 15%) ===")
    train_df, val_df, test_df = split_chronologically(engineered_df, train_ratio=0.70, val_ratio=0.15)

    os.makedirs("data/processed", exist_ok=True)
    engineered_df.to_csv("data/processed/energy_data.csv", index=False)
    train_df.to_csv("data/processed/train_data.csv", index=False)
    val_df.to_csv("data/processed/validation_data.csv", index=False)
    test_df.to_csv("data/processed/test_data.csv", index=False)

    logger.info("=== PREPROCESSING COMPLETED SUCCESSFULLY ===")
    logger.info(f"Base data saved to: data/processed/energy_data.csv ({len(engineered_df)} rows)")
    logger.info(f"Train partition: {len(train_df)} rows ({len(train_df)/96:.1f} days)")
    logger.info(f"Val partition:   {len(val_df)} rows ({len(val_df)/96:.1f} days)")
    logger.info(f"Test partition:  {len(test_df)} rows ({len(test_df)/96:.1f} days)")


if __name__ == "__main__":
    main()
