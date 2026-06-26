import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)

def optimize_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> dict:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Optimizing XGBoost with Optuna...")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best XGBoost Params: {study.best_params}")
    return study.best_params

def optimize_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> dict:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Optimizing LightGBM with Optuna...")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best LightGBM Params: {study.best_params}")
    return study.best_params

def optimize_rf(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 20) -> dict:
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = RandomForestRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(mean_squared_error(y_val, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Optimizing Random Forest with Optuna...")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info(f"Best RF Params: {study.best_params}")
    return study.best_params
