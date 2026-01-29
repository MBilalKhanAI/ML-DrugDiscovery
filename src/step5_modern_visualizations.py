"""
Step 5: Modern Visualizations (Analysis)
Generates advanced plots: t-SNE, Residual Analysis, and SHAP (w/ Permutation fallback).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.inspection import permutation_importance
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_data():
    if not os.path.exists('X_selected.csv') or not os.path.exists('y_data.csv'):
        logger.error("Missing input files.")
        return None, None
    X = pd.read_csv('X_selected.csv')
    y = pd.read_csv('y_data.csv')
    return X, y

def generate_tsne(X, y):
    logger.info("Generating Modern Figure 1: t-SNE Chemical Space...")
    
    # Scale data for t-SNE
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    tsne_df = pd.DataFrame(X_embedded, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['pIC50'] = y.values
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(
        data=tsne_df, x='Dimension 1', y='Dimension 2',
        hue='pIC50', palette='viridis', alpha=0.8,
        s=80, edgecolor='w'
    )
    plt.title('t-SNE Chemical Space Distribution', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    
    # Custom colorbar
    norm = plt.Normalize(tsne_df['pIC50'].min(), tsne_df['pIC50'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='pIC50 (Bioactivity)', ax=plt.gca())
    scatter.legend_.remove() # Correct way to remove legend from seaborn scatter
    
    plt.tight_layout()
    plt.savefig('Modern_Fig_TSNE.png', dpi=300)
    plt.close()

def generate_residuals(X, y):
    logger.info("Generating Modern Figure 2: Residual Analysis...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Get predictions via CV to avoid overfitting
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    y_true = y.values.ravel()
    y_pred = cross_val_predict(rf, X, y_true, cv=cv, n_jobs=-1)
    
    residuals = y_pred - y_true
    
    # Setup Grid
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 4)
    ax1 = fig.add_subplot(gs[0, :3]) # Scatter takes 3/4 width
    ax2 = fig.add_subplot(gs[0, 3])  # Hist takes 1/4 width
    
    # Plot 1: Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, ax=ax1, color='steelblue', alpha=0.6, edgecolor='w')
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted pIC50', fontsize=12)
    ax1.set_ylabel('Residuals (Predicted - True)', fontsize=12)
    ax1.set_title('Residuals vs. Predicted Values', fontsize=14)
    
    # Plot 2: Residual Distribution
    sns.histplot(y=residuals, ax=ax2, color='steelblue', kde=True)
    ax2.axhline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_title('Distribution', fontsize=14)
    
    # Share Y axis visual alignment
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([]) # Hide ticks on histogram y-axis
    
    plt.tight_layout()
    plt.savefig('Modern_Fig_Residuals.png', dpi=300)
    plt.close()

def generate_interpretability(X, y):
    logger.info("Generating Modern Figure 3: Interpretability (SHAP/Permutation)...")
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y.values.ravel())
    
    try:
        import shap
        logger.info("SHAP library found. Generating SHAP Summary Plot.")
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        
        # Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.title('SHAP Feature Importance (Top 20)', fontsize=16)
        plt.tight_layout()
        plt.savefig('Modern_Fig_SHAP.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError:
        logger.warning("SHAP not found. Falling back to Permutation Importance.")
        
        # Calculate Permutation Importance
        result = permutation_importance(rf, X, y.values.ravel(), n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = result.importances_mean.argsort()[::-1][:20] # Top 20
        
        # Prepare Data
        top_feats = X.columns[sorted_idx]
        top_importance = result.importances[sorted_idx].T
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.boxplot(top_importance, vert=False, labels=top_feats)
        plt.title("Permutation Importance (Top 20 Features)", fontsize=16)
        plt.xlabel("Importance Score (Decrease in R2)", fontsize=12)
        plt.tight_layout()
        plt.savefig('Modern_Fig_SHAP.png', dpi=300) # Keep requested filename for consistency
        plt.close()

def main():
    X, y = load_data()
    if X is None: return
    
    generate_tsne(X, y)
    generate_residuals(X, y)
    generate_interpretability(X, y)
    
    logger.info("Modern visualizations complete!")

if __name__ == "__main__":
    main()
