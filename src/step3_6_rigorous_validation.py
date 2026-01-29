import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
print("Loading data...")
try:
    X = pd.read_csv('X_selected.csv')
    y = pd.read_csv('y_data.csv').values.ravel()
    print(f"Data Loaded: {X.shape[0]} samples, {X.shape[1]} features")
except FileNotFoundError:
    print("Error: Files not found. Ensure X_selected.csv and y_data.csv are in the folder.")
    exit()

# 2. Hyperparameter Tuning (Paper Section 2.3)
# The paper states: "The GridSearchCV library in Scikit-learn is used to tune hyperparameters."
print("\n[1/3] Tuning Hyperparameters (GridSearchCV)...")
print("      Optimizing Random Forest (this may take a minute)...")

# Parameter grid for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=rf_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
rf_grid.fit(X, y)
best_rf = rf_grid.best_estimator_
print(f"      Best RF Params: {rf_grid.best_params_}")

# Optional: Tune GB as well since it was the second best
print("      Optimizing Gradient Boosting...")
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5]
}
gb_grid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=gb_params,
    cv=5,
    scoring='r2',
    n_jobs=-1
)
gb_grid.fit(X, y)
best_gb = gb_grid.best_estimator_

# 3. 10-Fold Cross-Validation (Paper Section 2.3)
print("\n[2/3] Running 10-Fold Cross-Validation...")
# "A 10-fold CV shows higher performance."

# Define the models (Using the TUNED versions)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest (Tuned)": best_rf,
    "Gradient Boosting (Tuned)": best_gb,
    "Bagging Regressor": BaggingRegressor(random_state=42)
}

cv = KFold(n_splits=10, shuffle=True, random_state=42)

results_table = []

print(f"{'Model':<25} | {'Mean R2':<10} | {'Mean RMSE':<10} | {'Paper R2 (Ref)'}")
print("-" * 65)
paper_refs = {
    "Linear Regression": 0.58,
    "Random Forest (Tuned)": 0.84,
    "Gradient Boosting (Tuned)": 0.77,
    "Bagging Regressor": 0.80
}

# Save output to a file string as well
output_log = []
header = f"{'Model':<25} | {'Mean R2':<10} | {'Mean RMSE':<10} | {'Paper R2 (Ref)'}"
output_log.append(header)
output_log.append("-" * 65)

for name, model in models.items():
    # R2 Scores
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    mean_r2 = r2_scores.mean()
    
    # RMSE Scores (Negative MSE -> RMSE)
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-mse_scores)
    mean_rmse = rmse_scores.mean()
    
    ref = paper_refs.get(name, "-")
    row = f"{name:<25} | {mean_r2:.4f}     | {mean_rmse:.4f}     | {ref}"
    print(row)
    output_log.append(row)

# 4. Feature Importance (Paper Figure 4)
print("\n[3/3] Feature Importance (from Best Tuned RF)...")
# Fit on full data
best_rf.fit(X, y)
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 5 Features:")
output_log.append("\nTop 5 Features (Tuned RF):")
for i in range(5):
    idx = indices[i]
    feat_str = f"  {i+1}. {X.columns[idx]:<15} ({importances[idx]:.4f})"
    print(feat_str)
    output_log.append(feat_str)

print("\nValidation Complete. Update your report with these 'Mean R2' values.")

# Write results to file for the agent to read
with open("rigorous_validation_results.txt", "w") as f:
    f.write("\n".join(output_log))
