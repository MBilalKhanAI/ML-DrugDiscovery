"""
Step 3.2: Imputation and Noise Removal
Handles missing values (mean imputation) and removes zero-variance (constant) features.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def impute_and_clean():
    logger.info("="*80)
    logger.info("Step 3.2: Imputation and Noise Removal")
    logger.info("="*80)

    input_file = 'X_numeric_raw.csv'
    output_file = 'X_cleaned.csv'

    # 1. Load Data
    if not os.path.exists(input_file):
        logger.error(f"Input file '{input_file}' not found!")
        return

    logger.info(f"Loading {input_file}...")
    X = pd.read_csv(input_file)
    original_shape = X.shape
    logger.info(f"Original shape: {original_shape}")

    # 2. Imputation (Mean)
    logger.info("\nChecking for missing values...")
    n_missing = X.isna().sum().sum()
    if n_missing > 0:
        logger.info(f"Found {n_missing} missing values. Performing mean imputation...")
        imputer = SimpleImputer(strategy='mean')
        # preserve column names
        X_imputed_array = imputer.fit_transform(X)
        
        # Check if columns were dropped
        if X_imputed_array.shape[1] != len(X.columns):
            logger.warning(f"Imputer dropped {len(X.columns) - X_imputed_array.shape[1]} columns (likely all-NaN).")
            # We need to figure out which columns are left if we want to keep names
            # But SimpleImputer doesn't easily tell us which ones it kept if it dropped them implicitly.
            # Best approach: Use get_feature_names_out if available (sklearn v1.0+), or just use simple imputer carefully.
            
            # Alternative: Ensure we keep all descriptors even if NaN for now (keep_empty_features if version supports)
            # Or better: Manually drop all-NaN columns BEFORE imputation.
            
            logger.info("Dropping all-NaN columns before imputation to fix alignment...")
            X_clean_nan = X.dropna(axis=1, how='all')
            dropped_nan_cols = len(X.columns) - len(X_clean_nan.columns)
            logger.info(f"Dropped {dropped_nan_cols} columns that were completely empty.")
            
            X_imputed_array = imputer.fit_transform(X_clean_nan)
            X = pd.DataFrame(X_imputed_array, columns=X_clean_nan.columns)
        else:
            X = pd.DataFrame(X_imputed_array, columns=X.columns)
            
        logger.info("Imputation complete. NaNs remaining: " + str(X.isna().sum().sum()))
    else:
        logger.info("No missing values found. Skipping imputation.")

    # 3. Remove Constant Columns (Variance = 0)
    logger.info("\nRemoving constant columns (variance = 0.0)...")
    
    # Calculate variance manually to identify which columns to drop (for logging)
    variances = X.var()
    constant_cols = variances[variances == 0].index
    num_constant = len(constant_cols)
    
    if num_constant > 0:
        logger.info(f"Found {num_constant} constant features to drop.")
        # Actually drop them
        X_cleaned = X.drop(columns=constant_cols)
        
        # Verify with VarianceThreshold for robustness (sanity check)
        selector = VarianceThreshold(threshold=0)
        selector.fit(X) # just fitting to check
        # assert len(X.columns) - np.sum(selector.get_support()) == num_constant
    else:
        logger.info("No constant columns found.")
        X_cleaned = X

    # 4. Save Output
    logger.info("\nSaving cleaned data...")
    X_cleaned.to_csv(output_file, index=False)
    
    final_shape = X_cleaned.shape
    dropped_cols = original_shape[1] - final_shape[1]
    
    logger.info(f"✓ Saved {output_file}")
    logger.info("-" * 40)
    logger.info(f"Original shape: {original_shape}")
    logger.info(f"Final shape:    {final_shape}")
    logger.info(f"Dropped features: {dropped_cols}")
    logger.info("-" * 40)
    
    logger.info("\nStep 3.2 completed successfully.")

if __name__ == "__main__":
    impute_and_clean()
