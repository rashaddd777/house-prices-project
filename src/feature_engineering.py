#!/usr/bin/env python3
"""
Module: feature_engineering.py
Purpose: Modular functions for feature creation and transformation.
"""

import pandas as pd
import numpy as np

def create_house_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column 'HouseAge' based on 'YrSold' - 'YearBuilt'.
    """
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    return df

def create_years_since_remodel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column 'YearsSinceRemodel' based on 'YrSold' - 'YearRemodAdd'.
    """
    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
    return df

def create_total_sf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column 'TotalSF' by summing 1stFlrSF, 2ndFlrSF, and TotalBsmtSF.
    """
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    return df

def log_transform_column(df: pd.DataFrame, col_name: str, new_col: str) -> pd.DataFrame:
    """
    Applies log1p transformation to a specified column and stores it in a new column.
    """
    df[new_col] = np.log1p(df[col_name])
    return df

def has_garage_flag(df: pd.DataFrame, garage_col: str = 'GarageType') -> pd.DataFrame:
    """
    Creates a binary 'HasGarage' flag based on whether the garage column is 'NA' or not.
    """
    df['HasGarage'] = df[garage_col].apply(lambda x: 0 if str(x) == 'NA' else 1)
    return df

def encode_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """
    One-hot encodes the specified list of categorical columns and returns the transformed DataFrame.
    """
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A single function to apply a series of feature engineering steps.
    Adjust the steps as needed for your project.
    """
    df = create_house_age(df)
    df = create_years_since_remodel(df)
    df = create_total_sf(df)
    df = log_transform_column(df, col_name='GrLivArea', new_col='LogGrLivArea')
    df = has_garage_flag(df, garage_col='GarageType')
    return df

if __name__ == "__main__":
    # Example usage
    import os

    sample_path = "../data/processed/train_cleaned.csv"
    if os.path.exists(sample_path):
        df_sample = pd.read_csv(sample_path)
        df_transformed = engineer_features(df_sample)
        # Optionally encode some columns
        df_encoded = encode_categorical(df_transformed, cat_cols=['MSZoning', 'Neighborhood'])
        output_path = "../data/processed/train_feature_engineered.csv"
        df_encoded.to_csv(output_path, index=False)
        print(f"Feature-engineered data saved to: {output_path}")
    else:
        print("Sample cleaned file not found.")
