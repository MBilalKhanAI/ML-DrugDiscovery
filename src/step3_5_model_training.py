"""
Step 3.5: Model Training & Evaluation
Trains Linear Regression, Random Forest, Gradient Boosting, and Bagging models.
Evaluates using R2 score and RMSE on a test set (20%).
"""

import pandas as pd
import numpy as np
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(messages)s' # Simplified format for results table
)
logger = logging.getLogger(__name__)

def train_and_evaluate():
    print("="*80)
    print("Step 3.5: Model Training & Evaluation (Train/Test Split)")
    print("="*80)

    # 1. Load Data
    X = pd.read_csv('X_selected.csv')
    y = pd.read_csv('y_data.csv')
    y = y.values.ravel() # Convert to 1D array

    print(f"Loaded Data: X={X.shape}, y={y.shape}")

    # 2. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")

    # 3. Scaling (Important for Linear Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Initialize Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Bagging Regressor": BaggingRegressor(random_state=42)
    }

    # 5. Train and Evaluate
    results = []
    
    print("\n" + "-"*65)
    print(f"{'Model':<25} | {'R2 Score':<10} | {'RMSE':<10}")
    print("-"*65)

    best_model_name = ""
    best_r2 = -float('inf')

    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results.append((name, r2, rmse))
        
        print(f"{name:<25} | {r2:.4f}     | {rmse:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print("-"*65)
    print(f"\nBest Performing Model: {best_model_name} (R2 = {best_r2:.4f})")
    print("="*80)

    # Save results to file
    with open("model_results.txt", "w") as f:
        f.write(f"Model Training Results:\n")
        f.write(f"{'Model':<25} | {'R2 Score':<10} | {'RMSE':<10}\n")
        f.write("-" * 50 + "\n")
        for name, r2, rmse in results:
             f.write(f"{name:<25} | {r2:.4f}     | {rmse:.4f}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Best: {best_model_name} (R2={best_r2:.4f})\n")

    return results

if __name__ == "__main__":
    train_and_evaluate()
