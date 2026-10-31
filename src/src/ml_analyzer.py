"""Adaptive machine-learning utilities for RetailIQ V0.7.

V0.7 adds conservative target detection, explicit supervised/unsupervised
routing, permutation-based feature importance, reproducible experiments,
and dataset suitability diagnostics. The module remains a transparent
scikit-learn baseline rather than a claim of full AutoML.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import logging
import re

import numpy as np
import pandas as pd
import warnings

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# Only names with strong semantic evidence are allowed to drive automatic
# target selection. Generic numeric fields such as "value" are deliberately
# not treated as targets because they are often legitimate features.
STRONG_TARGET_EXACT = {
    "target", "label", "class", "outcome", "churn", "default", "response",
    "sales", "revenue", "profit", "demand", "prediction", "target_value",
}
STRONG_TARGET_TOKENS = (
    "target", "label", "outcome", "churn", "default", "response",
    "sales", "revenue", "profit", "demand", "prediction", "forecast",
)
WEAK_TARGET_TOKENS = ("rating", "score", "price", "amount", "quantity", "total", "value")

RANDOM_STATE = 42
logger = logging.getLogger(__name__)


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(100)
        if len(sample) < 5:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        return bool(parsed.notna().mean() >= 0.80)
    return False


def _datetime_expand(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if _is_datetime_like(out[col]):
            dt = pd.to_datetime(out[col], errors="coerce")
            out[f"{col}__year"] = dt.dt.year
            out[f"{col}__month"] = dt.dt.month
            out[f"{col}__day"] = dt.dt.day
            out[f"{col}__dayofweek"] = dt.dt.dayofweek
            out.drop(columns=[col], inplace=True)
    return out


def _looks_identifier(name: str, series: pd.Series) -> bool:
    name_l = str(name).strip().lower()
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(len(series), 1)
    tokens = set(re.findall(r"[a-z0-9]+", name_l))
    id_token = bool(tokens.intersection({"id", "uuid", "key", "code", "number", "identifier"}))
    text_like = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    return (
        (id_token and ratio >= 0.70)
        or nunique <= 1
        or (text_like and nunique >= 20 and ratio >= 0.98)
    )


def _target_reason(col: str, df: pd.DataFrame) -> tuple[float, str]:
    name = str(col).strip().lower()
    series = df[col]
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(len(df), 1)
    if nunique < 2 or _is_datetime_like(series):
        return -1.0, "not a suitable target candidate"

    # A strong semantic target name overrides high-cardinality heuristics.
    # For example, a continuous `sales` column can legitimately have one
    # unique value per row and must not be mistaken for an identifier.
    if name in STRONG_TARGET_EXACT:
        return 1.00, "strong semantic target name"
    if any(tok in name for tok in STRONG_TARGET_TOKENS):
        return 0.90, "strong semantic target-name match"
    if _looks_identifier(col, series):
        return -1.0, "not a suitable target candidate"

    # Weak target names are only suggestions for human review, never an
    # automatic target. This prevents monetary_value from hijacking
    # unsupervised datasets.
    if any(tok in name for tok in WEAK_TARGET_TOKENS):
        return 0.45, "weak target-name hint; manual confirmation recommended"

    # A low-cardinality categorical field can be a target, but only as a
    # manual suggestion. Automatic routing remains conservative.
    if (pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)) and 2 <= nunique <= min(10, max(2, int(len(df) * 0.20))):
        return 0.35, "low-cardinality categorical field; manual confirmation recommended"
    return 0.0, "no target-specific evidence"


def target_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return ranked target suggestions with confidence and reasons."""
    rows = []
    for col in df.columns:
        score, reason = _target_reason(col, df)
        if score > 0:
            rows.append({"Column": col, "Confidence": score, "Reason": reason})
    if not rows:
        return pd.DataFrame(columns=["Column", "Confidence", "Reason"])
    return pd.DataFrame(rows).sort_values(["Confidence", "Column"], ascending=[False, True]).reset_index(drop=True)


def detect_target(df: pd.DataFrame) -> str | None:
    """Auto-select only high-confidence semantic targets; otherwise None."""
    candidates = target_candidates(df)
    if candidates.empty:
        return None
    top = candidates.iloc[0]
    return str(top["Column"]) if float(top["Confidence"]) >= 0.80 else None


def infer_task(y: pd.Series) -> str:
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "unsupported"
    if (pd.api.types.is_bool_dtype(y_nonnull) or pd.api.types.is_object_dtype(y_nonnull)
            or pd.api.types.is_string_dtype(y_nonnull) or isinstance(y_nonnull.dtype, pd.CategoricalDtype)):
        return "classification" if y_nonnull.nunique() >= 2 else "unsupported"
    unique = y_nonnull.nunique()
    if unique <= 10 and unique / max(len(y_nonnull), 1) <= 0.10:
        return "classification"
    return "regression"


def _drop_unusable_features(
    X: pd.DataFrame,
    target: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    out = _datetime_expand(X)
    dropped = []
    if target and str(target).strip().lower() == "sales":
        post_outcome_columns = {
            "cost", "profit", "profit_margin", "net_profit", "gross_profit",
        }
        leakage = [
            column for column in out.columns
            if re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
            in post_outcome_columns
        ]
        dropped.extend(leakage)
        out.drop(columns=leakage, inplace=True)
    for c in list(out.columns):
        if _looks_identifier(c, out[c]):
            dropped.append(c)
            out.drop(columns=[c], inplace=True)
    numeric_columns = out.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in out.columns if column not in numeric_columns]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(np.float64)
    for column in categorical_columns:
        values = out[column].astype(object)
        out[column] = values.where(pd.notna(values), np.nan)
    return out, dropped


def _normalize_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y.dtype):
        return pd.Series(
            pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan),
            index=y.index,
            name=y.name,
        )
    return y.astype(object)


def _unsupervised_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric predictors while excluding known labels and identifiers."""
    features = df.select_dtypes(include=np.number).copy()
    known_labels = {
        "target", "label", "class", "outcome", "is_anomaly", "anomaly",
        "anomaly_label", "ground_truth", "true_label",
    }
    detected_target = detect_target(df)
    excluded = [
        column for column in features.columns
        if column == detected_target
        or str(column).strip().lower() in known_labels
        or _looks_identifier(str(column), features[column])
    ]
    return features.drop(columns=excluded)


def _preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def _safe_split(X, y, task, test_size=0.20, random_state=RANDOM_STATE):
    if not 0 < test_size < 1:
        raise ValueError("The test-set proportion must be between 0 and 1.")
    if len(y) < 2:
        raise ValueError("At least two labeled rows are required for a train/test split.")
    stratify = None
    if task == "classification":
        counts = y.value_counts()
        test_rows = int(np.ceil(len(y) * test_size))
        train_rows = len(y) - test_rows
        if len(counts) > 1 and counts.min() >= 2:
            if test_rows >= len(counts) and train_rows >= len(counts):
                stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def _model_specs(task: str):
    if task == "classification":
        return {
            "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
            "Decision Tree": (DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE), False),
            "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
        }
    return {
        "Linear Regression": (LinearRegression(), False),
        "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
    }


def _permutation_importance(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, task: str, top_n: int = 12) -> pd.DataFrame:
    """Measure importance on original input columns to avoid scale/one-hot bias."""
    try:
        scoring = "f1_weighted" if task == "classification" else "neg_root_mean_squared_error"
        perm = permutation_importance(
            pipeline, X_test, y_test, scoring=scoring,
            n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        )
        result = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": perm.importances_mean,
            "Std": perm.importances_std,
        }).sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)
        return result
    except Exception:
        logger.exception("Permutation importance failed for task %s", task)
        return pd.DataFrame(columns=["Feature", "Importance", "Std"])


def _diagnostics(df: pd.DataFrame, target: str, task: str) -> list[str]:
    notes = []
    if len(df) < 100:
        notes.append("Small dataset: treat benchmark scores as preliminary and prefer cross-validation for final claims.")
    missing_pct = float(df.isna().mean().mean() * 100)
    if missing_pct > 10:
        notes.append(f"Missing-cell rate is {missing_pct:.1f}%; imputation may affect model results.")
    if task == "classification":
        counts = df[target].value_counts(normalize=True)
        if not counts.empty and float(counts.min()) < 0.10:
            notes.append("Class imbalance detected; weighted metrics are reported, but per-class metrics should be inspected for final evaluation.")
    return notes


def run_supervised(df: pd.DataFrame, target: str, task: str | None = None, test_size: float = 0.20) -> dict:
    if target not in df.columns:
        return {"ok": False, "error": "Selected target column was not found."}

    work = df.copy().dropna(subset=[target])
    y = _normalize_target(work[target])
    inferred = infer_task(y)
    task = task or inferred
    if task not in {"classification", "regression"}:
        return {"ok": False, "error": "The selected target is not suitable for supervised learning."}
    if len(work) < 30:
        return {"ok": False, "error": f"At least 30 usable rows are recommended for a meaningful ML benchmark; this dataset has {len(work)}."}
    if task == "classification" and y.nunique() < 2:
        return {"ok": False, "error": "Classification requires at least two target classes."}
    if task == "regression" and y.nunique() < 5:
        return {"ok": False, "error": "Regression requires more variation in the target."}

    X = work.drop(columns=[target])
    X, dropped = _drop_unusable_features(X, target=target)
    if X.shape[1] == 0:
        return {"ok": False, "error": "No usable feature columns remain after removing identifier-like/constant fields."}

    X_train, X_test, y_train, y_test = _safe_split(X, y, task, test_size=test_size)
    results, pipelines, importances = [], {}, {}
    for name, (model, scale) in _model_specs(task).items():
        pipe = Pipeline([
            ("preprocessor", _preprocessor(X_train, scale_numeric=scale)),
            ("model", model),
        ])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            if task == "classification":
                metrics = {
                    "Accuracy": accuracy_score(y_test, pred),
                    "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                    "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
            else:
                metrics = {
                    "MAE": mean_absolute_error(y_test, pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                    "R2": r2_score(y_test, pred),
                }
            results.append({"Model": name, **metrics})
            pipelines[name] = pipe
            importances[name] = _permutation_importance(pipe, X_test, y_test, task)
        except Exception:
            logger.exception("Supervised model %s failed for target %s", name, target)
            results.append({"Model": name, "Error": "Model training failed; see application logs."})

    result_df = pd.DataFrame(results)
    result = {
        "ok": bool(pipelines),
        "task": task,
        "target": target,
        "rows_used": len(work),
        "features_used": X.shape[1],
        "dropped_features": dropped,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "results": result_df,
        "pipelines": pipelines,
        "importances": importances,
        "diagnostics": _diagnostics(work, target, task),
        "random_state": RANDOM_STATE,
        "test_size": test_size,
    }
    if not pipelines:
        result["error"] = "No candidate model could be trained. Review the target, class distribution, and predictors."
    return result


def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> dict:
    X = _unsupervised_numeric_features(df)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] < 2 or len(X) < 20:
        return {"ok": False, "error": "Clustering requires at least 20 rows and two varying numeric features."}
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    X_scaled = StandardScaler().fit_transform(X)
    k = max(2, min(int(n_clusters), len(X) - 1))
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else np.nan
    assignments = X.copy()
    assignments["Cluster"] = labels
    return {
        "ok": True, "model": model, "silhouette": score,
        "assignments": assignments, "features": X.columns.tolist(),
        "cluster_sizes": pd.Series(labels).value_counts().sort_index().to_dict(),
        "random_state": RANDOM_STATE,
    }


def run_anomaly_detection(df: pd.DataFrame, contamination: float = 0.05) -> dict:
    if not 0 < contamination <= 0.5:
        return {"ok": False, "error": "Contamination must be greater than 0 and no greater than 0.5."}
    X = _unsupervised_numeric_features(df)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] < 1 or len(X) < 20:
        return {"ok": False, "error": "Anomaly detection requires at least 20 rows and one varying numeric feature."}
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    scores = model.decision_function(X_scaled)
    out = df.copy()
    out["ML_Anomaly"] = np.where(labels == -1, "Anomaly", "Normal")
    out["Anomaly_Score"] = scores
    return {
        "ok": True, "model": model, "data": out,
        "anomalies": int((labels == -1).sum()), "features": X.columns.tolist(),
        "contamination": contamination, "random_state": RANDOM_STATE,
    }


def log_experiment(result: dict, path: str | Path = "results/ml_experiments.csv") -> Path:
    """Append a compact, reproducible record of a supervised benchmark."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = result.get("results", pd.DataFrame()).copy()
    if rows.empty:
        return p
    best = None
    valid = rows.dropna(how="all", subset=[c for c in rows.columns if c != "Model" and c != "Error"])
    if not valid.empty:
        if result.get("task") == "classification" and "F1" in valid:
            best = str(valid.loc[valid["F1"].idxmax(), "Model"])
        elif result.get("task") == "regression" and "RMSE" in valid:
            best = str(valid.loc[valid["RMSE"].idxmin(), "Model"])
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": result.get("task"), "target": result.get("target"),
        "rows_used": result.get("rows_used"), "train_rows": result.get("train_rows"),
        "test_rows": result.get("test_rows"), "features_used": result.get("features_used"),
        "dropped_features": ", ".join(result.get("dropped_features", [])),
        "test_size": result.get("test_size"), "random_state": result.get("random_state"),
        "best_baseline": best,
    }
    for _, row in rows.iterrows():
        rec = record.copy()
        rec.update({k: v for k, v in row.to_dict().items()})
        header = not p.exists()
        pd.DataFrame([rec]).to_csv(p, mode="a", header=header, index=False)
    return p


# ---------------- Research evaluation ----------------

def _cv_model_specs(task: str):
    return _model_specs(task)


def _make_pipeline(X: pd.DataFrame, model, scale_numeric: bool) -> Pipeline:
    return Pipeline([
        ("preprocessor", _preprocessor(X, scale_numeric=scale_numeric)),
        ("model", model),
    ])


def _score_predictions(task: str, y_true, y_pred) -> dict:
    if task == "classification":
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_supervised(df: pd.DataFrame, target: str, task: str | None = None, folds: int = 5) -> dict:
    """Run leakage-safe K-fold CV. Preprocessing is fitted independently inside each fold."""
    from sklearn.model_selection import KFold, StratifiedKFold

    if target not in df.columns:
        return {"ok": False, "error": "Target column not found."}
    work = df.copy().dropna(subset=[target])
    y = work[target]
    task = task or infer_task(y)
    if task not in {"classification", "regression"}:
        return {"ok": False, "error": "Unsupported supervised task."}
    if len(work) < folds * 5:
        return {"ok": False, "error": f"At least {folds*5} usable rows are recommended for {folds}-fold CV."}
    X = work.drop(columns=[target])
    X, dropped = _drop_unusable_features(X)
    if X.shape[1] == 0:
        return {"ok": False, "error": "No usable feature columns remain."}

    if task == "classification":
        min_class = int(y.value_counts().min())
        if min_class < folds:
            return {"ok": False, "error": f"The smallest class has only {min_class} rows; cannot perform {folds}-fold stratified CV."}
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(X)

    fold_rows = []
    specs = _cv_model_specs(task)
    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for name, (model, scale) in specs.items():
            pipe = _make_pipeline(X_train, model, scale)
            try:
                pipe.fit(X_train, y_train)
                pred = pipe.predict(X_test)
                metrics = _score_predictions(task, y_test, pred)
                fold_rows.append({"Fold": fold, "Model": name, **metrics})
            except Exception as exc:
                fold_rows.append({"Fold": fold, "Model": name, "Error": str(exc)})

    fold_df = pd.DataFrame(fold_rows)
    metric = "F1" if task == "classification" else "RMSE"
    summary = fold_df.groupby("Model", dropna=False).agg({
        **({"Accuracy": "mean", "Precision": "mean", "Recall": "mean", "F1": "mean"} if task == "classification" else {"MAE": "mean", "RMSE": "mean", "R2": "mean"})
    }).reset_index()
    std_cols = [c for c in summary.columns if c != "Model"]
    std_df = fold_df.groupby("Model")[std_cols].std(ddof=1).reset_index()
    std_df = std_df.rename(columns={c: f"{c}_Std" for c in std_cols})
    summary = summary.merge(std_df, on="Model", how="left")
    return {
        "ok": True, "task": task, "target": target, "rows_used": len(work),
        "features_used": X.shape[1], "dropped_features": dropped, "folds": folds,
        "fold_results": fold_df, "summary": summary, "selection_metric": metric,
        "random_state": RANDOM_STATE,
    }


def evaluate_clustering_k_range(df: pd.DataFrame, k_values=range(2, 6)) -> dict:
    """Evaluate K-Means across a small predefined K range for reproducible comparison."""
    X = _unsupervised_numeric_features(df)
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] < 2 or len(X) < 20:
        return {"ok": False, "error": "Clustering requires at least 20 rows and two varying numeric features."}
    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    X_scaled = StandardScaler().fit_transform(X)
    rows = []
    for k in k_values:
        if k >= len(X):
            continue
        model = KMeans(n_clusters=int(k), random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        rows.append({"K": int(k), "Silhouette": float(score), "Inertia": float(model.inertia_)})
    result = pd.DataFrame(rows)
    return {"ok": not result.empty, "results": result, "features": X.columns.tolist(), "random_state": RANDOM_STATE}


def run_routing_evaluation() -> dict:
    """Evaluate the automatic routing rule on diverse built-in sklearn datasets.

    This is a routing benchmark, not a predictive-accuracy claim about RetailIQ.
    """
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

    datasets = []
    bc = load_breast_cancer(as_frame=True)
    d = bc.frame.copy(); d["target"] = d["target"].map({0: "malignant", 1: "benign"})
    datasets.append(("Breast Cancer", d, "classification"))
    w = load_wine(as_frame=True)
    d = w.frame.copy(); d["target"] = d["target"].astype(str)
    datasets.append(("Wine", d, "classification"))
    di = load_diabetes(as_frame=True)
    d = di.frame.copy(); d.rename(columns={"target": "target_value"}, inplace=True)
    datasets.append(("Diabetes", d, "regression"))
    ir = load_iris(as_frame=True)
    d = ir.frame.copy(); d.drop(columns=["target"], inplace=True)
    datasets.append(("Iris (target removed)", d, "unsupervised"))

    rows = []
    for name, frame, expected in datasets:
        detected = detect_target(frame)
        if detected:
            actual = infer_task(frame[detected])
        else:
            actual = "unsupervised"
        rows.append({
            "Dataset": name, "Rows": len(frame), "Expected_Task": expected,
            "Detected_Target": detected or "None", "Detected_Task": actual,
            "Correct": bool(actual == expected),
        })
    result = pd.DataFrame(rows)
    return {"ok": True, "results": result, "accuracy": float(result["Correct"].mean()), "random_state": RANDOM_STATE}


def run_research_benchmark(output_dir: str | Path = "results", folds: int = 5) -> dict:
    """Run the complete reproducible research benchmark and save CSV artifacts."""
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine, load_iris
    import time

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    routing = run_routing_evaluation()
    routing["results"].to_csv(out / "routing_evaluation.csv", index=False)

    supervised_runs = []
    datasets = []
    bc = load_breast_cancer(as_frame=True); d = bc.frame.copy(); d["target"] = d["target"].map({0:"malignant",1:"benign"}); datasets.append(("Breast Cancer", d, "target", "classification"))
    w = load_wine(as_frame=True); d = w.frame.copy(); d["target"] = d["target"].astype(str); datasets.append(("Wine", d, "target", "classification"))
    di = load_diabetes(as_frame=True); d = di.frame.copy(); d.rename(columns={"target":"target_value"}, inplace=True); datasets.append(("Diabetes", d, "target_value", "regression"))

    for dataset_name, frame, target, task in datasets:
        start = time.perf_counter()
        result = cross_validate_supervised(frame, target, task, folds=folds)
        runtime = time.perf_counter() - start
        if result["ok"]:
            summary = result["summary"].copy()
            summary.insert(0, "Dataset", dataset_name)
            summary["Runtime_Sec"] = runtime
            supervised_runs.append(summary)
            result["fold_results"].assign(Dataset=dataset_name).to_csv(out / f"{dataset_name.lower().replace(' ','_')}_fold_results.csv", index=False)

    cv_summary = pd.concat(supervised_runs, ignore_index=True) if supervised_runs else pd.DataFrame()
    cv_summary.to_csv(out / "supervised_cv_results.csv", index=False)

    # Unsupervised benchmark on Iris with the class label intentionally removed.
    iris = load_iris(as_frame=True).frame.drop(columns=["target"])
    clustering = evaluate_clustering_k_range(iris, range(2, 6))
    if clustering["ok"]:
        clustering["results"].to_csv(out / "clustering_results.csv", index=False)

    # Statistical comparison: Friedman test across CV folds for each dataset/task.
    stat_rows = []
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
        if not cv_summary.empty:
            for dataset_name, _, _, task in datasets:
                fold_file = out / f"{dataset_name.lower().replace(' ','_')}_fold_results.csv"
                if not fold_file.exists():
                    continue
                fdf = pd.read_csv(fold_file)
                metric = "F1" if task == "classification" else "RMSE"
                pivot = fdf.pivot(index="Fold", columns="Model", values=metric).dropna()
                if pivot.shape[1] >= 3 and pivot.shape[0] >= 3:
                    arrays = [pivot[c].values for c in pivot.columns]
                    stat, p = friedmanchisquare(*arrays)
                    stat_rows.append({"Dataset": dataset_name, "Test":"Friedman", "Metric":metric, "Statistic":float(stat), "P_Value":float(p)})
                    # Pairwise Wilcoxon only if Friedman is significant; otherwise record no pairwise tests.
                    if p < 0.05:
                        cols = list(pivot.columns)
                        for i in range(len(cols)):
                            for j in range(i+1, len(cols)):
                                try:
                                    ws, wp = wilcoxon(pivot[cols[i]], pivot[cols[j]], zero_method="wilcox", alternative="two-sided")
                                    stat_rows.append({"Dataset":dataset_name, "Test":f"Wilcoxon: {cols[i]} vs {cols[j]}", "Metric":metric, "Statistic":float(ws), "P_Value":float(wp)})
                                except Exception:
                                    pass
    except ImportError:
        stat_rows.append({"Dataset":"All","Test":"Statistical tests unavailable","Metric":"","Statistic":np.nan,"P_Value":np.nan})
    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(out / "statistical_tests.csv", index=False)

    summary = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "routing_accuracy": routing["accuracy"], "folds": folds,
        "random_state": RANDOM_STATE, "classification_datasets": 2,
        "regression_datasets": 1, "unsupervised_datasets": 1,
    }])
    summary.to_csv(out / "research_run_summary.csv", index=False)
    return {"ok": True, "routing": routing, "cv_summary": cv_summary, "clustering": clustering, "statistics": stat_df, "output_dir": str(out)}
