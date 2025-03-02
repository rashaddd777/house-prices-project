#!/usr/bin/env python3
"""
Module: models.py
Purpose: Define model architectures and training routines for the House Prices project.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a baseline Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_ridge_model(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Ridge:
    """
    Train a Ridge regression model with specified alpha.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_lasso_model(X_train: pd.DataFrame, y_train: pd.Series, alpha: float = 1.0) -> Lasso:
    """
    Train a Lasso regression model with specified alpha.
    """
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None, num_rounds: int = 100) -> xgb.Booster:
    """
    Train an XGBoost regressor.
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    booster = xgb.train(params, dtrain, num_boost_round=num_rounds)
    return booster

def predict_model(model, X, model_type: str = 'sklearn'):
    """
    Generate predictions using the provided model.
    
    Parameters:
      - model: the trained model.
      - X: features for prediction.
      - model_type: 'sklearn' for scikit-learn models or 'xgboost' for XGBoost.
    
    Returns:
      - Predicted values as a NumPy array.
    """
    if model_type.lower() == 'xgboost':
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)
    else:
        preds = model.predict(X)
    return preds

def evaluate_model(model, X, y_true, model_type: str = 'sklearn') -> float:
    """
    Evaluate the model using RMSE.
    
    Returns:
      - RMSE value.
    """
    y_pred = predict_model(model, X, model_type=model_type)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def save_model(model, file_path: str):
    """
    Save the trained model to disk using joblib.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path: str):
    """
    Load a saved model from disk.
    """
    model = joblib.load(file_path)
    return model

if __name__ == "__main__":
    # Example usage with synthetic regression data
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Train a baseline linear model
    linear_model = train_linear_model(X, y)
    rmse = evaluate_model(linear_model, X, y)
    print(f"Baseline Linear Model RMSE: {rmse:.2f}")
    
    # Save the trained model
    save_model(linear_model, "linear_model.pkl")
