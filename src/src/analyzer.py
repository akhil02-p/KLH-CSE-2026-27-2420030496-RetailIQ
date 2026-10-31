import pandas as pd
import numpy as np

def _date_like_columns(df):
    result = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            result.append(col)
        elif pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            sample = s.dropna().astype(str).head(100)
            if len(sample) >= 5:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().mean() >= 0.80:
                    result.append(col)
    return result

def _role(dtype, unique_count, row_count):
    if pd.api.types.is_numeric_dtype(dtype):
        return "Numeric / Measure"
    if unique_count <= max(20, int(row_count * 0.05)):
        return "Categorical / Dimension"
    return "Text / Identifier"

def analyze_dataset(df):
    rows, columns = df.shape
    profiles = []
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["object","string","category","bool"]).columns.tolist()
    dates = _date_like_columns(df)

    for col in df.columns:
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique(dropna=True))
        profiles.append({
            "Column": col,
            "Data Type": str(df[col].dtype),
            "Non-Null": int(df[col].notna().sum()),
            "Missing": missing,
            "Missing %": round(missing / rows * 100, 2) if rows else 0,
            "Unique Values": unique,
            "Potential Role": _role(df[col].dtype, unique, rows)
        })

    quality = []
    for r in profiles:
        if r["Missing"] > 0:
            severity = "High" if r["Missing %"] >= 50 else ("Medium" if r["Missing %"] >= 10 else "Low")
            action = "Review/remove column" if severity == "High" else ("Consider imputation" if severity == "Medium" else "Consider targeted imputation")
            quality.append({
                "Column": r["Column"],
                "Severity": severity,
                "Issue": f"{r['Missing %']}% missing values",
                "Suggested Action": action,
                "Missing Values": r["Missing"]
            })
        elif r["Unique Values"] <= 1:
            quality.append({
                "Column": r["Column"],
                "Severity": "High",
                "Issue": "Constant / no variation",
                "Suggested Action": "Consider removing constant column",
                "Missing Values": 0
            })

    return {
        "rows": rows,
        "columns": columns,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "date_columns": dates,
        "column_profile": pd.DataFrame(profiles),
        "quality_table": pd.DataFrame(quality)
    }
