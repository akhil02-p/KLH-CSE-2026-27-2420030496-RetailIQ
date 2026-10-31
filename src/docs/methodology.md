# RetailIQ Methodology

## 1. Data Ingestion
Accept CSV/XLSX/XLS files and load them into Pandas.

## 2. Dataset Profiling
Measure rows, columns, missing cells, duplicates, unique values, data types and likely field roles.

## 3. Data Quality Assessment
Identify missing-value problems, fully empty columns, constant columns and duplicates.

## 4. Cleaning & Pre-processing
Standardize column names, remove fully empty columns and exact duplicates, trim text and detect strong date-like fields. All actions are logged.

## 5. Adaptive Recommendation
Recommend EDA, dashboards, time-series analysis and segmentation based on detected analytical signals.

## 6. Visualization
Generate interactive Plotly charts when suitable fields are available.

## 7. Export
Allow users to download the cleaned dataset as CSV.
