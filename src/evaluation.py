#!/usr/bin/env python3
"""
Module: evaluation.py
Purpose: Implements evaluation metrics and visualization functions for regression models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def compute_rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between actual and predicted values.
    
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    
    Returns:
        float: The RMSE value.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse

def compute_r2(y_true, y_pred):
    """
    Compute the R² (coefficient of determination) between actual and predicted values.
    
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    
    Returns:
        float: The R² score.
    """
    r2 = r2_score(y_true, y_pred)
    return r2

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted", save_path=None):
    """
    Plot a scatter plot of actual vs. predicted values with a reference diagonal line.
    
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the plot to the given file path.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
    
    # Draw diagonal line for perfect predictions
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals Distribution", save_path=None):
    """
    Plot the distribution of residuals (errors) between actual and predicted values.
    
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
        title (str): Title of the plot.
        save_path (str, optional): If provided, saves the plot to the given file path.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    y_true = np.random.normal(loc=100, scale=10, size=100)
    y_pred = y_true + np.random.normal(loc=0, scale=5, size=100)
    
    rmse = compute_rmse(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)
    print(f"Example RMSE: {rmse:.2f}")
    print(f"Example R²: {r2:.2f}")
    
    # Example plots (will display and save if save_path provided)
    plot_actual_vs_predicted(y_true, y_pred, title="Example Actual vs. Predicted")
    plot_residuals(y_true, y_pred, title="Example Residuals Distribution")
