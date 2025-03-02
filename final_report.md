# Final Report: House Prices Prediction Project

## 1. Overview

This project was developed for the Kaggle “House Prices: Advanced Regression Techniques” competition. It demonstrates a complete data science pipeline—from data exploration and cleaning, through advanced feature engineering and model building, to hyperparameter tuning, ensemble modeling, and final evaluation. The project is fully reproducible and available on GitHub.

## 2. Data Exploration & Cleaning

- **Raw Data:**
  - **Shape:** 1460 rows × 81 columns  
  - Detailed raw data information is available in `reports/text/raw_data_info.txt`.

- **Cleaned Data:**
  - **Shape after Outlier Removal:** 1399 rows × 81 columns  
  - Summaries of the cleaned dataset are stored in `reports/text/basic_info.txt` and `reports/text/cleaned_data_shape.txt`.

- **Missing & Categorical Analysis:**  
  - Full missing value summaries and categorical counts are documented in `reports/text/missing_values.txt` and `reports/text/categorical_value_counts.txt`.

## 3. Feature Engineering

Key features were derived to boost model performance. For example, the following features were created:
- **HouseAge:** Calculated as `YrSold - YearBuilt`
- **YearsSinceRemodel:** Calculated as `YrSold - YearRemodAdd`
- **TotalSF:** Sum of `1stFlrSF`, `2ndFlrSF`, and `TotalBsmtSF`
- **LogGrLivArea:** Log-transformed `GrLivArea` to reduce skewness
- **HasGarage:** A binary flag indicating garage presence

A summary of key feature statistics is provided in `reports/text/key_features_summary.txt`.

## 4. Modeling & Hyperparameter Tuning

### Baseline Model
- **Model:** Linear Regression  
- **Validation RMSE:** 19,859.58  
  (See `reports/text/baseline_model_evaluation.txt` for details)

### Ensemble Model
- **Model:** Stacking Ensemble (combining Linear Regression, Random Forest, and XGBoost with Ridge as the meta-model)
- **Validation RMSE:** 19,354.04  
- **Validation R²:** 0.88  
  (Detailed evaluation is in `reports/text/final_evaluation_report.txt` and `reports/text/final_evaluation_summary.txt`)

### Random Forest Grid Search
- **Best CV RMSE:** 21,693.82  
- **Best Parameters:**  
  - `max_depth`: 20  
  - `min_samples_split`: 2  
  - `n_estimators`: 200  
  (Full grid search results in `reports/text/rf_grid_search_results.txt`)
- **Random Forest Validation RMSE:** 21,478.68  
  (See `reports/text/rf_validation_evaluation.txt`)

## 5. Submission Preparation

The final submission file (`submission.csv`) was generated after processing the test data with the same pipeline and contains the predicted `SalePrice` for each test instance. The submission file follows the competition format with columns `Id` and `SalePrice`.

## 6. Conclusions & Future Work

- **Results:**  
  The ensemble model achieved a lower validation RMSE (19,354.04) and an R² of 0.88, which is an improvement over the baseline model (RMSE 19,859.58). While the Random Forest grid search provided competitive parameters, its validation RMSE was higher (21,478.68), indicating that the stacking ensemble benefits from combining multiple model types.

- **Future Work:**  
  - Explore additional feature transformations and interaction effects.  
  - Experiment with alternative ensembling methods or more advanced deep learning approaches.  
  - Extend hyperparameter tuning on a larger grid and incorporate cross-validation strategies for further performance gains.

## 7. Repository & Contact

The complete project—including all code, notebooks, and this report—is available on [GitHub](https://github.com/YourUsername/house-prices-project).

For any questions or further details, please contact me at [your.email@example.com].
