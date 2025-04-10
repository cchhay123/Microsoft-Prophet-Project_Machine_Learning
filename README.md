# Stock Price Forecasting with Machine Learning and Prophet

This project showcases my work applying machine learning models to forecast future stock prices. Using historical financial data from IBM, I explored multiple predictive approaches—including **XGBoost**, **Random Forest**, and the **Prophet** model from Meta —to evaluate which methods offer the best performance for short-term stock price prediction.

---

##  Data Sources

- **GoldmineIBM.csv**: Primary dataset with engineered financial indicators and historical prices.
- **VolatilityIBM.csv**: Supplementary volatility data for model enrichment.
- **IBMFinancialStatements.csv**: Used for feature generation and broader financial context.

---

## Project Goals

- Predict the 30-day future average buy price using regression models.
- Compare machine learning methods (e.g., XGBoost, Random Forest) with time series models like Prophet.
- Tune model hyperparameters and evaluate performance using RMSE and visual inspection.

---

## Machine Learning Models

### Linear Regression + Backward Selection
- Built a baseline linear model using all predictors.
- Applied backward selection with `step()` based on AIC to improve model simplicity and performance.

### Random Forest
- Used `randomForest` package to create ensemble models.
- Tuned parameters like `ntree` and `nodesize` for best performance.
- Visualized model accuracy across parameter combinations using heatmaps.

### XGBoost
- Applied `xgboost` with advanced hyperparameter tuning:
  - `max_depth`, `min_child_weight`, `gamma`
  - `subsample` and `colsample_bytree`
- Used cross-validation to optimize and reduce overfitting.
- Visualized results with `ggplot2`.

---

## Time Series Forecasting with Prophet

- Implemented the `prophet` package to model time series data.
- Incorporated trend, seasonality, and holiday effects.
- Compared Prophet forecasts with XGBoost and Random Forest outputs.

---

## Model Evaluation

- **Accuracy Metrics**: Root Mean Squared Error (RMSE), out-of-bag error, and cross-validation scores.
- **Visualization**: Line plots comparing actual vs predicted values across models.
- **Insights**:
  - Prophet handles long-term trends and seasonality well.
  - XGBoost performs better for short-term predictions with engineered features.

---

## Technologies Used

- **R packages**: `xgboost`, `randomForest`, `prophet`, `ggplot2`, `dplyr`, `caret`, `Metrics`
- **Techniques**: Feature engineering, time series alignment, model tuning, cross-validation

---

## Outcome

- Built and compared multiple predictive models for stock price forecasting.
- Demonstrated strengths of different modeling techniques across time horizons.
- Delivered a flexible and reproducible pipeline for financial forecasting tasks.


[Project Final Report.pdf](https://github.com/user-attachments/files/19654892/Project.Final.Report.pdf)
