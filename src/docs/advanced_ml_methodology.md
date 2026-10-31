# RetailIQ V1.0 Advanced ML/AI Methodology

RetailIQ V1.0 adds the remaining planned analytics paths while keeping the system lightweight and reproducible.

## 1. Anomaly detection
Isolation Forest is applied to standardized numeric features after median imputation. The contamination parameter is user-configurable. Flagged rows receive `ML_Anomaly` and `Anomaly_Score`.

## 2. Advanced predictive analytics
Gradient Boosting is added as an advanced predictive baseline for classification and regression. A fixed random seed (42) and a hold-out evaluation are used. Numeric features are median-imputed and standardized. Permutation importance is reported to expose influential variables.

## 3. Time-series forecasting
When a date-like column and numeric measure are detected, RetailIQ aggregates observations to an inferred frequency and constructs lag/rolling features. Ridge Lag and Gradient Boosting Lag models are compared using MAE, RMSE and R2 on a chronological hold-out period. The selected model is then refit on the available history for recursive horizon forecasting. Prophet is not required.

## 4. ML-guided recommendations
Recommendations are generated from measured model performance, feature influence, anomaly counts and forecast direction. Each recommendation records its evidence source, making the output traceable rather than opaque.

## 5. Natural-language insight generation
RetailIQ converts observed descriptive statistics, data-quality signals, anomaly counts and forecast changes into deterministic narrative statements. This module is intentionally non-generative and does not claim to replace an external large language model.

## Research evaluation
The V1.0 research suite retains the original 10-dataset, 5-repeat × 5-fold benchmark for supervised/clustering evaluation and adds separate reproducible tracks for advanced predictive modeling, anomaly detection and synthetic retail forecasting. Synthetic results are clearly labeled as synthetic evidence.
