#!/usr/bin/env python3
"""
Module: data_preprocessing.py
Purpose: Load raw data and apply cleaning functions.
"""

import pandas as pd
import numpy as np

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw CSV data into a DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in a DataFrame.
    - Numeric columns: fill with median.
    - Categorical columns: fill with mode.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    return df

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Removes outliers from a specified column using the IQR method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column name from which to remove outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_clean

def preprocess_data(file_path: str, outlier_column: str = "SalePrice") -> pd.DataFrame:
    """
    End-to-end preprocessing: load data, impute missing values, and remove outliers.
    
    Parameters:
        file_path (str): Path to the raw CSV file.
        outlier_column (str): Column on which to perform outlier removal.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = load_raw_data(file_path)
    df = impute_missing_values(df)
    df_clean = remove_outliers_iqr(df, column=outlier_column)
    return df_clean

if __name__ == "__main__":
    # Example usage: preprocess the train data and save the cleaned version.
    raw_file = "../data/raw/train.csv"
    cleaned_df = preprocess_data(raw_file)
    
    output_path = "../data/processed/train_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
