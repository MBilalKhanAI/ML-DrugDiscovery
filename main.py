import argparse
import pandas as pd
import numpy as np
import os
import logging

from src.features.descriptors import generate_full_feature_set
from src.data.splitting import scaffold_split
from src.models.optimize import optimize_xgboost, optimize_lightgbm, optimize_rf
from src.models.train_ensemble import build_stacking_ensemble, train_and_evaluate
from src.visualization.plots import plot_parity, plot_shap_summary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Stanford-Quality ML-DrugDiscovery Pipeline")
    parser.add_argument('--dataset', type=str, default='data/dataset.csv', help='Path to dataset CSV with SMILES and pIC50')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save outputs')
    parser.add_argument('--figures_dir', type=str, default='figures', help='Directory to save figures')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of Optuna trials for hyperparameter tuning')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    
    # 1. Load Data
    logger.info(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset)
    
    # 2. Generate Features (if not already cached)
    feature_cache_path = os.path.join(args.output_dir, 'features_cached.csv')
    if os.path.exists(feature_cache_path):
        logger.info("Loading cached features...")
        X = pd.read_csv(feature_cache_path)
    else:
        logger.info("Generating features (Morgan + RDKit 2D)...")
        X = generate_full_feature_set(df['SMILES'].tolist())
        # Clean infinite and NaN values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        X.to_csv(feature_cache_path, index=False)
        
    y = df['pIC50']
    
    # Combine for splitting to keep indices aligned
    full_df = pd.concat([df[['SMILES']], X, y], axis=1)
    
    # 3. Scaffold Split
    logger.info("Performing Bemis-Murcko Scaffold Split...")
    train_df, valid_df, test_df = scaffold_split(full_df, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    
    X_train = train_df.drop(columns=['SMILES', 'pIC50'])
    y_train = train_df['pIC50']
    X_valid = valid_df.drop(columns=['SMILES', 'pIC50'])
    y_valid = valid_df['pIC50']
    X_test = test_df.drop(columns=['SMILES', 'pIC50'])
    y_test = test_df['pIC50']
    
    # Feature Selection based on Variance to remove constant features
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)
    X_train_sel = pd.DataFrame(selector.fit_transform(X_train), columns=X_train.columns[selector.get_support()])
    X_valid_sel = pd.DataFrame(selector.transform(X_valid), columns=X_valid.columns[selector.get_support()])
    X_test_sel = pd.DataFrame(selector.transform(X_test), columns=X_test.columns[selector.get_support()])
    
    logger.info(f"Features reduced from {X_train.shape[1]} to {X_train_sel.shape[1]}")
    
    # 4. Hyperparameter Optimization
    # We combine train and valid for CV-based optimization
    X_opt = pd.concat([X_train_sel, X_valid_sel], axis=0)
    y_opt = pd.concat([y_train, y_valid], axis=0)
    
    xgb_best = optimize_xgboost(X_opt, y_opt, n_trials=args.n_trials)
    lgbm_best = optimize_lightgbm(X_opt, y_opt, n_trials=args.n_trials)
    rf_best = optimize_rf(X_opt, y_opt, n_trials=args.n_trials)
    
    # 5. Build and Train Ensemble
    stacking_model = build_stacking_ensemble(xgb_best, lgbm_best, rf_best)
    
    model, preds, r2, rmse = train_and_evaluate(stacking_model, X_opt, y_opt, X_test_sel, y_test)
    
    # Save Results Summary
    results_txt = os.path.join(args.output_dir, 'academic_results.txt')
    with open(results_txt, 'w') as f:
        f.write("=== ML-DrugDiscovery Stacking Ensemble Results ===\n")
        f.write(f"Test Set R2: {r2:.4f}\n")
        f.write(f"Test Set RMSE: {rmse:.4f}\n")
        f.write("\n=== Hyperparameters Found by Optuna ===\n")
        f.write(f"XGBoost: {xgb_best}\n")
        f.write(f"LightGBM: {lgbm_best}\n")
        f.write(f"Random Forest: {rf_best}\n")
    logger.info(f"Results written to {results_txt}")
        
    # 6. Generate Visualizations
    parity_path = os.path.join(args.figures_dir, 'fig_parity_plot.png')
    plot_parity(y_test, preds, r2, rmse, parity_path)
    
    shap_path = os.path.join(args.figures_dir, 'fig_shap_summary.png')
    plot_shap_summary(model, X_opt, shap_path)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
