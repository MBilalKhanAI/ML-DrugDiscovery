"""
Step 3.1: Initial Data Cleaning
Forces numeric conversion of molecular descriptors and verifies data integrity.
"""

import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_data():
    logger.info("="*80)
    logger.info("Step 3.1: Data Cleaning (Force Numeric)")
    logger.info("="*80)

    # 1. Load Data
    if not os.path.exists('X_descriptors.csv') or not os.path.exists('y_data.csv'):
        logger.error("Input files not found!")
        return

    logger.info("Loading data...")
    X = pd.read_csv('X_descriptors.csv', low_memory=False) # low_memory=False to avoid warning on mixed types
    y = pd.read_csv('y_data.csv')
    
    logger.info(f"Loaded X shape: {X.shape}")
    logger.info(f"Loaded y shape: {y.shape}")

    # 2. Force Numeric Conversion
    logger.info("\nForcing numeric conversion on X descriptors...")
    
    # Identify non-numeric errors before conversion for reporting
    # This is just for information
    
    # Convert all columns to numeric, coercing errors to NaN
    # This handles "DivideByZero", "Reals", strings, etc.
    X_numeric = X.apply(pd.to_numeric, errors='coerce')
    
    # Check how many NaNs we introduced (indicates how many values were non-numeric)
    nans_before = X.isna().sum().sum()
    nans_after = X_numeric.isna().sum().sum()
    new_nans = nans_after - nans_before
    
    if new_nans > 0:
        logger.info(f"Converted {new_nans} non-numeric values (errors) to NaN.")
    else:
        logger.info("No non-numeric error strings found (clean conversion).")

    # 3. Verify Alignment
    if len(X_numeric) != len(y):
        logger.error(f"CRITICAL ERROR: Shape mismatch! X: {len(X_numeric)}, y: {len(y)}")
        return
    
    # 4. Save Output
    logger.info("\nSaving cleaned numeric data...")
    output_filename = 'X_numeric_raw.csv'
    X_numeric.to_csv(output_filename, index=False)
    
    logger.info(f"✓ Saved {output_filename}")
    logger.info(f"Final X shape: {X_numeric.shape}")
    logger.info("All columns are now strictly numeric (float/int).")
    
    # Optional: Report columns with high missing values
    missing_counts = X_numeric.isna().sum()
    high_missing = missing_counts[missing_counts > 0]
    if not high_missing.empty:
        logger.info(f"\nNote: {len(high_missing)} columns contain NaN values (failed calculations).")
        logger.info("These will need to be handled in Step 3.2 (Feature Selection).")

    logger.info("="*80)
    logger.info("Step 3.1 completed successfully.")

if __name__ == "__main__":
    clean_data()
