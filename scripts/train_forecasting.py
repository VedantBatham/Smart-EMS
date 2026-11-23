"""
Execution script to train and evaluate XGBoost short-term solar and load forecasters.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forecasting.train_solar_model import train_solar_forecaster
from forecasting.train_load_model import train_load_forecaster
from forecasting.evaluate_forecast import evaluate_forecasting_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Training Solar PV XGBoost Model ===")
    from forecasting.train_solar_model import train_solar_forecaster
    solar_metrics = train_solar_forecaster()
    logger.info(f"Solar training finished: {solar_metrics}")

    logger.info("=== STEP 2: Training Household Load XGBoost Model ===")
    load_metrics = train_load_forecaster()
    logger.info(f"Load training finished: {load_metrics}")

    logger.info("=== STEP 3: Evaluating On Held-Out Test Set ===")
    test_metrics = evaluate_forecasting_models()
    logger.info("=== FORECASTING PIPELINE COMPLETED SUCCESSFULLY ===")
    print("Forecasting Metrics Summary:")
    import json
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
