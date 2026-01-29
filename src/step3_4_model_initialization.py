"""
Step 3.4: Model Initialization
Initializes the four regression models used in the study with specified hyperparameters.
"""

import pandas as pd
import logging
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_models():
    logger.info("="*80)
    logger.info("Step 3.4: Model Initialization")
    logger.info("="*80)

    # 1. Load Data (Just to verify existence for next steps)
    if not os.path.exists('X_selected.csv') or not os.path.exists('y_data.csv'):
        logger.error("Data files not found! Please run previous steps.")
        return

    logger.info("Data files verified (X_selected.csv, y_data.csv).")

    # 2. Initialize Models
    logger.info("\nInitializing models...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Bagging Regressor": BaggingRegressor(random_state=42)
    }

    # 3. Print Confirmation
    logger.info(f"Initialized {len(models)} models successfully:")
    for name, model in models.items():
        logger.info(f"  - {name}: {model}")

    logger.info("\n" + "="*80)
    logger.info("Models are ready for training.")
    logger.info("="*80)

    return models

if __name__ == "__main__":
    initialize_models()
