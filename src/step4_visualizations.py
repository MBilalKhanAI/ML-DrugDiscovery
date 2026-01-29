"""
Step 4: Visualization (Figures 2-8)
Generates high-resolution figures matching the paper's methodology.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('ggplot')
sns.set_context("paper")

def load_data():
    if not os.path.exists('X_selected.csv') or not os.path.exists('y_data.csv') or not os.path.exists('dataset.csv'):
        logger.error("Missing input files.")
        return None, None, None
    X = pd.read_csv('X_selected.csv')
    y = pd.read_csv('y_data.csv')
    original_df = pd.read_csv('dataset.csv')
    return X, y, original_df

def get_top_features_by_corr(X, y, n=10):
    """Sort features by absolute correlation with y."""
    data = X.copy()
    data['target'] = y.values
    corr = data.corr()['target'].drop('target')
    top_indices = corr.abs().sort_values(ascending=False).index[:n]
    return top_indices

def generate_figure_2(X, y):
    logger.info("Generating Figure 2: Distribution of Descriptors & pIC50...")
    top_9_feats = get_top_features_by_corr(X, y, n=9)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot Target
    sns.histplot(y['pIC50'], kde=True, ax=axes[0], color='black')
    axes[0].set_title('pIC50 (Target)')
    
    # Plot Top 9 Features
    for i, feat in enumerate(top_9_feats):
        sns.histplot(X[feat], kde=True, ax=axes[i+1])
        axes[i+1].set_title(feat)
        
    # Hide unused subplots
    for j in range(len(top_9_feats)+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig('Figure_2.png', dpi=300)
    plt.close()

def generate_figure_3(X, y):
    logger.info("Generating Figure 3: Correlation Heatmap...")
    top_10_feats = get_top_features_by_corr(X, y, n=10)
    data_subset = X[top_10_feats].copy()
    data_subset['pIC50'] = y.values
    
    plt.figure(figsize=(10, 8))
    corr_matrix = data_subset.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Pearson Correlation Heatmap (Top 10 Features + pIC50)")
    plt.tight_layout()
    plt.savefig('Figure_3.png', dpi=300)
    plt.close()

def generate_figure_4(X, y):
    logger.info("Generating Figure 4: Feature Importance (RF)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y.values.ravel())
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], align='center', color='teal')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Top 15 Descriptors (Random Forest)')
    plt.gca().invert_yaxis() # Highest importance at top
    plt.tight_layout()
    plt.savefig('Figure_4.png', dpi=300)
    plt.close()
    
    return rf # Return trained model for later use

def generate_figure_5(X, y):
    logger.info("Generating Figure 5: True vs Predicted Scatter Plots (10-Fold CV)...")
    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Bagging": BaggingRegressor(random_state=42)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    y_true = y.values.ravel()
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(models.items()):
        y_pred = cross_val_predict(model, X, y_true, cv=cv)
        
        # Plot
        ax = axes[i]
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='w')
        
        # Parity line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        
        ax.set_xlabel('Measured pIC50')
        ax.set_ylabel('Predicted pIC50')
        ax.set_title(f'{name}')
        ax.text(0.05, 0.95, f"Method: 10-Fold CV", transform=ax.transAxes, verticalalignment='top')
        
    plt.tight_layout()
    plt.savefig('Figure_5.png', dpi=300)
    plt.close()

def generate_figure_6(rf_model, X):
    logger.info("Generating Figure 6: Predicted pIC50 Distribution (Simulated Mining)...")
    # Simulate 5000 molecules based on feature statistics
    n_samples = 5000
    means = X.mean()
    stds = X.std()
    
    # Generate synthetic data
    X_synthetic = np.random.normal(loc=means, scale=stds, size=(n_samples, X.shape[1]))
    X_synthetic_df = pd.DataFrame(X_synthetic, columns=X.columns)
    
    # Predict
    y_pred_synthetic = rf_model.predict(X_synthetic_df)
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_synthetic, bins=50, color='purple', edgecolor='black', alpha=0.7)
    plt.title('Predicted pIC50 Distribution for 5000 Synthetic Molecules')
    plt.xlabel('Predicted pIC50')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('Figure_6.png', dpi=300)
    plt.close()

def generate_figure_7(original_df):
    logger.info("Generating Figure 7: Top 20 Molecules (Best Candidates)...")
    # Identify top 20 by pIC50
    if 'pIC50' not in original_df.columns:
        logger.error("pIC50 missing in original dataset")
        return

    top_20 = original_df.nlargest(20, 'pIC50')
    mols = []
    legends = []
    
    for idx, row in top_20.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"Idx: {idx}\npIC50: {row['pIC50']:.2f}")
            
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
        img.save('Figure_7.png')
    else:
        logger.warning("No molecules could be drawn for Figure 7.")

def generate_figure_8(original_df):
    logger.info("Generating Figure 8: Top 5 Training Molecules...")
    # Top 5 best pIC50
    top_5 = original_df.nlargest(5, 'pIC50')
    mols = []
    legends = []
    
    for idx, row in top_5.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            mols.append(mol)
            legends.append(f"pIC50: {row['pIC50']:.2f}")
            
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(300, 300), legends=legends)
        img.save('Figure_8.png')
    else:
        logger.warning("No molecules could be drawn for Figure 8.")

def main():
    X, y, df = load_data()
    if X is None: return
    
    generate_figure_2(X, y)
    generate_figure_3(X, y)
    rf_model = generate_figure_4(X, y) # Returns fitted model
    generate_figure_5(X, y)
    generate_figure_6(rf_model, X)
    generate_figure_7(df)
    generate_figure_8(df)
    
    logger.info("Visualization generation complete!")

if __name__ == "__main__":
    main()
