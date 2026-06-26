import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

# Set publication quality styling
plt.style.use('default')
sns.set_theme(style='whitegrid', context='paper', font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def plot_parity(y_true: pd.Series, y_pred: np.ndarray, r2: float, rmse: float, output_path: str):
    """
    Creates a publication-ready Parity Plot (Predicted vs Actual).
    """
    plt.figure(figsize=(8, 8))
    
    # Scatter plot with density coloring or simple aesthetic styling
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, edgecolor='k', s=80, color='#2c7bb6')
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min()) - 0.5
    max_val = max(y_true.max(), y_pred.max()) + 0.5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.text(0.05, 0.95, f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}', 
             transform=plt.gca().transAxes, fontsize=16,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Experimental pIC$_{50}$', fontweight='bold')
    plt.ylabel('Predicted pIC$_{50}$', fontweight='bold')
    plt.title('Model Performance on Independent Test Set', fontweight='bold', pad=15)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved parity plot to {output_path}")

def plot_shap_summary(model, X_train: pd.DataFrame, output_path: str):
    """
    Creates a SHAP summary plot for model interpretability.
    Note: SHAP typically requires a tree-based model. For a StackingRegressor, 
    we often compute SHAP values for the most important base estimator (e.g., XGBoost).
    """
    # Extract the XGBoost model from the stacking ensemble
    xgb_model = None
    if hasattr(model, 'estimators_'):
        for name, est in zip(model.estimators, model.estimators_):
            if name[0] == 'xgb':
                xgb_model = est
                break
    else:
        xgb_model = model # Fallback if it's not a stacking regressor
        
    if xgb_model is None:
        logger.warning("Could not find XGBoost model in ensemble for SHAP analysis. Skipping.")
        return

    logger.info("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_train, show=False, max_display=15)
    
    # Polish the plot
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=14, fontweight='bold')
    plt.title('Top 15 Most Important Features driving Aromatase Inhibition', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved SHAP summary plot to {output_path}")
