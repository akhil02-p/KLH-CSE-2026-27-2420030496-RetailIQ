# DAV Alignment Status

This document is a project self-assessment, not an external certification. The repository contains no official course/institutional DAV rubric, so formal compliance cannot be established from the checked-in evidence alone.

RetailIQ implements a churn-oriented problem definition, dataset profiling, preprocessing, classification and regression model comparison, hybrid prediction, SHAP/LIME explanations, recommendations, and Responsible AI guidance. The app also supports general uploaded datasets and currently exposes 26 navigation destinations. Sales prediction excludes known post-sale predictors; raw and cleaned duplicate counts are reported independently.

The latest backend and interactive audit also covered nullable data, rare-class splitting, regression explainability, sales leakage, categorical preprocessing, chart contrast, and clickable branded navigation. The full test suite reports **16 passed**. The requested `RetailIQ_Master_Test_Dataset(1).csv`, `RetailIQ_Bar_Chart_Test_Dataset.csv`, and `RetailIQ_Customer_Churn_Demo.csv` files were not present, so their exact behavior remains unverified. See the [backend and DAV audit report](../RETAILIQ_BACKEND_AND_DAV_AUDIT_REPORT.md) for root causes, validation evidence, and remaining limitations.

Do not claim full external DAV compliance until the official rubric is obtained and every criterion is mapped to validated evidence. The checked-in research manifest contains 10 benchmark datasets but does not contain an `online_retail_customer_churn.csv` dataset or a `Target_Churn` column.
