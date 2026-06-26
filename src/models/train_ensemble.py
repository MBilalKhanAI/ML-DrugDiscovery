import numpy as np
import pandas as pd
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)

def build_stacking_ensemble(xgb_params: dict, lgbm_params: dict, rf_params: dict) -> StackingRegressor:
    """
    Builds a Stacking Regressor combining XGBoost, LightGBM, and Random Forest.
    """
    estimators = [
        ('xgb', XGBRegressor(**xgb_params, random_state=42)),
        ('lgbm', LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)),
        ('rf', RandomForestRegressor(**rf_params, random_state=42))
    ]
    
    # Meta-learner
    final_estimator = Ridge(alpha=1.0)
    
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    
    return stacking_regressor

def train_and_evaluate(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Trains the model and evaluates on the test set.
    """
    logger.info("Training the Stacking Ensemble model...")
    model.fit(X_train, y_train)
    
    logger.info("Evaluating on Test Set...")
    preds = model.predict(X_test)
    
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    logger.info(f"Test R2 Score: {r2:.4f}")
    logger.info(f"Test RMSE: {rmse:.4f}")
    
    return model, preds, r2, rmse
