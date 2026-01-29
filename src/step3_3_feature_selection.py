"""
Step 3.3: Feature Selection (Univariate Regression)
Selects the top 50 most relevant descriptors based on f_regression scores.
Methodology: "Best descriptors are shortlisted using univariate regression."
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def select_features():
    logger.info("="*80)
    logger.info("Step 3.3: Feature Selection (Univariate f_regression)")
    logger.info("="*80)

    input_X = 'X_cleaned.csv'
    input_y = 'y_data.csv'
    output_file = 'X_selected.csv'
    K_FEATURES = 50

    # 1. Load Data
    if not os.path.exists(input_X) or not os.path.exists(input_y):
        logger.error("Input files not found!")
        return

    logger.info("Loading data...")
    X = pd.read_csv(input_X)
    y = pd.read_csv(input_y)
    
    # Ensure y is a 1D array
    y_values = y.iloc[:, 0].values
    
    logger.info(f"X shape: {X.shape}")
    logger.info(f"y shape: {y.shape}")

    # 2. Select Top 50 Features
    logger.info(f"\nSelecting top {K_FEATURES} features using f_regression...")
    
    selector = SelectKBest(score_func=f_regression, k=K_FEATURES)
    selector.fit(X, y_values)
    
    # Get selected feature names and scores
    cols = X.columns
    selected_mask = selector.get_support()
    selected_features = cols[selected_mask]
    selected_scores = selector.scores_[selected_mask]
    
    # Create DataFrame with selected features
    X_selected = X[selected_features]
    
    # 3. Print Top 5 Features
    logger.info("\nTop 5 Selected Features (by F-score):")
    
    # Combine names and scores for sorting
    feature_score_pairs = list(zip(selected_features, selected_scores))
    # Sort by score descending
    feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, score) in enumerate(feature_score_pairs[:5]):
        logger.info(f"  {i+1}. {name:<30} (Score: {score:.2f})")

    # 4. Save Output
    logger.info(f"\nSaving reduced dataset to {output_file}...")
    X_selected.to_csv(output_file, index=False)
    
    logger.info(f"✓ Saved {output_file}")
    logger.info("-" * 40)
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Selected features: {X_selected.shape[1]}")
    logger.info("-" * 40)
    
    logger.info("\nStep 3.3 completed successfully.")

if __name__ == "__main__":
    select_features()
