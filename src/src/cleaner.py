import re
import pandas as pd

def _standardize_name(name):
    name = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", name).strip("_") or "column"

def build_cleaning_plan(df):
    plan, seen = [], set()
    for col in df.columns:
        new = _standardize_name(col)
        base, i = new, 2
        while new in seen:
            new = f"{base}_{i}"
            i += 1
        seen.add(new)
        plan.append((col, new))
    return plan

def clean_dataset(df, plan):
    cleaned = df.copy()
    log = []
    rename_map = dict(plan)
    if any(a != b for a,b in plan):
        cleaned = cleaned.rename(columns=rename_map)
        for a,b in plan:
            if a != b:
                log.append({"Action":"Renamed","Column":f"{a} → {b}","Reason":"Standardized column naming"})

    empty = [c for c in cleaned.columns if cleaned[c].isna().all()]
    for c in empty:
        log.append({"Action":"Dropped","Column":c,"Reason":"Column contains 100% missing values"})
    if empty:
        cleaned = cleaned.drop(columns=empty)

    dup = int(cleaned.duplicated().sum())
    if dup:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
        log.append({"Action":"Removed duplicates","Column":"-","Reason":f"Removed {dup} exact duplicate rows"})

    for c in cleaned.select_dtypes(include=["object","string"]).columns:
        before = cleaned[c].astype("string")
        cleaned[c] = before.str.strip()
        if (before.fillna("__NA__") != cleaned[c].fillna("__NA__")).any():
            log.append({"Action":"Trimmed text","Column":c,"Reason":"Removed leading/trailing whitespace"})

    for c in cleaned.select_dtypes(include=["object"]).columns:
        sample = cleaned[c].dropna().astype(str).head(100)
        if len(sample) >= 5:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.90:
                cleaned[c] = pd.to_datetime(cleaned[c], errors="coerce")
                log.append({"Action":"Converted","Column":c,"Reason":"Detected date-like values"})
    return cleaned, log
