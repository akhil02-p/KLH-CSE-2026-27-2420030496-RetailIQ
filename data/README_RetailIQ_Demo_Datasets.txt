RetailIQ – External Faculty Demonstration Datasets

This ZIP contains two compact demonstration datasets.

1) RetailIQ_Master_Test_Dataset.csv
   - 367 raw records and 18 columns.
   - Intended for general retail analytics demonstrations.
   - Suggested numerical target: sales (regression).
   - Includes churn and is_anomaly labels for relevant classification/anomaly-analysis demonstrations.
   - Contains a small number of missing values and duplicate records intentionally, useful for demonstrating data-quality checks and cleaning.
   - Caution: cost, profit and profit_margin are derived/related to sales. Do not use them as predictors when demonstrating a realistic pre-sale sales forecast.

2) RetailIQ_Customer_Churn_Demo.csv
   - Derived from the same source records, retaining customer and transaction attributes plus churn.
   - Suggested target: churn (classification; 0 = not churned, 1 = churned).
   - Excludes sales, cost, profit and profit_margin to reduce obvious target leakage for the churn demonstration.
   - Contains missing values as present in the source; demonstrate preprocessing before modelling.

These are project demonstration/test datasets, not production business records.
Use the first dataset for a broad walkthrough and the second when you want a focused classification example.
