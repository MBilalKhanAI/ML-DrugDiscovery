"""
Step 3.6: Validation and Feature Importance
Performs 10-Fold Cross-Validation and extracts Feature Importance from Random Forest.
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def validate_and_analyze():
    print("="*80)
    print("Step 3.6: 10-Fold Cross-Validation & Feature Importance")
    print("="*80)

    # 1. Load Data
    X = pd.read_csv('X_selected.csv')
    y = pd.read_csv('y_data.csv').values.ravel()
    
    print(f"Data Loaded: {X.shape} samples")

    # 2. Define Models
    # Note: Using pipelines for scaling where appropriate (Linear Regression)
    # Tree-based models (RF, GBR) generally don't require scaling but it doesn't hurt.
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Bagging Regressor": BaggingRegressor(random_state=42)
    }

    # Paper reported values (approximate from user request)
    paper_values = {
        "Linear Regression": 0.58,
        "Random Forest": 0.84,
        "Gradient Boosting": 0.77,
        "Bagging Regressor": 0.80
    }

    # 3. 10-Fold Cross-Validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    print("\n" + "-"*95)
    print(f"{'Model':<20} | {'Mean R2':<10} | {'Mean RMSE':<10} | {'Paper R2 (Ref)':<15} | {'Diff'}")
    print("-"*95)

    results_data = []

    for name, model in models.items():
        # R2 Score
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        mean_r2 = r2_scores.mean()
        
        # RMSE Score (Negative RMSE needed from cross_val_score)
        rmse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')
        mean_rmse = -rmse_scores.mean()
        
        paper_ref = paper_values.get(name, 0.0)
        diff = mean_r2 - paper_ref
        
        print(f"{name:<20} | {mean_r2:.4f}     | {mean_rmse:.4f}     | {paper_ref:<15} | {diff:+.4f}")
        results_data.append((name, mean_r2, mean_rmse))

    print("-"*95)

    # 4. Feature Importance (Random Forest)
    print("\n" + "="*80)
    print("Feature Importance Analysis (Random Forest)")
    print("="*80)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1] # Sort descending
    
    print("Top 5 Most Important Descriptors:")
    print(f"{'Rank':<5} | {'Descriptor':<30} | {'Importance Score'}")
    print("-" * 60)
    
    for f in range(5):
        idx = indices[f]
        name = X.columns[idx]
        score = importances[idx]
        print(f"{f+1:<5} | {name:<30} | {score:.4f}")
        
    print("-" * 60)
    
    # Save validation results to file
    with open("validation_results.txt", "w") as f:
        f.write("10-Fold Cross-Validation Results:\n")
        f.write(f"{'Model':<20} | {'Mean R2':<10} | {'Mean RMSE':<10}\n")
        for name, r2, rmse in results_data:
            f.write(f"{name:<20} | {r2:.4f}     | {rmse:.4f}\n")
            
        f.write("\nTop 5 Features:\n")
        f.write(f"{'Descriptor':<30} | {'Importance'}\n")
        for f_idx in range(5):
            idx = indices[f_idx]
            f.write(f"{X.columns[idx]:<30}: {importances[idx]:.4f}\n")

if __name__ == "__main__":
    validate_and_analyze()
