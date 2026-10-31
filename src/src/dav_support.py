from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
)
from sklearn.base import is_classifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.ml_analyzer import _drop_unusable_features, _safe_split, infer_task

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:  # pragma: no cover
    LimeTabularExplainer = None

logger = logging.getLogger(__name__)


def build_project_definition() -> dict[str, Any]:
    return {
        "problem_statement": "RetailIQ addresses customer churn risk in retail analytics by turning transactional and customer behavior data into a transparent, explainable decision-support system.",
        "business_objective": "Reduce preventable customer churn, prioritize intervention efforts, and improve retention decisions using reliable predictive analytics.",
        "analytics_objective": "Identify churn drivers, compare predictive models, explain classification decisions, and surface data-backed recommendations for customer retention.",
        "research_objective": "Evaluate whether a hybrid ensemble improves churn prediction compared with baseline and standard comparison models while preserving explainability.",
        "target_variable": "churn",
        "prediction_task": "binary classification",
        "kpis": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "PR-AUC",
            "Customer churn rate",
            "High-risk customer share",
        ],
        "research_questions": [
            "What customer characteristics are associated with churn?",
            "Which customer segments have the highest churn risk?",
            "Which factors contribute most to churn?",
            "Which model predicts churn most reliably?",
            "Does the proposed hybrid model outperform individual baseline models?",
            "Why did the model classify a particular customer as high risk?",
        ],
        "business_questions": [
            "Which customers should receive retention outreach first?",
            "Which factors justify a personalized offer or intervention?",
            "How should the business segment risk to optimize support resources?",
        ],
        "constraints_and_limitations": [
            "Model quality depends on the quality and completeness of the uploaded dataset.",
            "If the source data lacks churn labels or relevant customer behavior features, predictive performance will be limited.",
            "Recommendations should be treated as decision-support rather than guaranteed outcomes.",
        ],
    }


def _is_date_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(200)
        if len(sample) < 5:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.80
    return False


def summarize_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    df = df.copy()
    rows, cols = df.shape
    numeric_columns = list(df.select_dtypes(include=[np.number]).columns)
    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns and (pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(df[col]))
    ]
    date_columns = [col for col in df.columns if _is_date_like(df[col])]
    missing_by_column = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    inferred_target = target_column or _infer_target_column(df)
    target_distribution = None
    class_imbalance = None
    if inferred_target and inferred_target in df.columns:
        y = df[inferred_target].dropna()
        target_distribution = y.value_counts(dropna=False).to_dict()
        if pd.api.types.is_numeric_dtype(y) and y.nunique() <= 10 and y.nunique() >= 2:
            counts = y.value_counts(normalize=True)
            class_imbalance = float(counts.min() / max(counts.max(), 1e-9))
        elif pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(y):
            counts = y.value_counts(normalize=True)
            class_imbalance = float(counts.min() / max(counts.max(), 1e-9))
    summary = {
        "rows": rows,
        "columns": cols,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "missing_values": missing_by_column,
        "missing_total": missing_total,
        "duplicate_rows": duplicate_rows,
        "target_column": inferred_target,
        "target_distribution": target_distribution,
        "class_imbalance": class_imbalance,
        "data_types": {col: str(df[col].dtype) for col in df.columns},
        "unique_values": {col: int(df[col].nunique(dropna=True)) for col in df.columns},
        "descriptive_statistics": df.describe(include="all").T.to_dict(),
    }
    return summary


def build_target_distribution_data(target: pd.Series) -> tuple[str, pd.DataFrame]:
    """Prepare raw numeric values for regression histograms or class counts."""
    values = target.dropna()
    task = infer_task(values)
    if task == "regression":
        numeric_values = pd.to_numeric(values, errors="coerce").dropna()
        return task, numeric_values.rename("Target value").reset_index(drop=True).to_frame()
    distribution = values.value_counts(dropna=True).rename_axis("Target class").reset_index(name="Rows")
    return task, distribution


def _infer_target_column(df: pd.DataFrame) -> str | None:
    strong_names = [
        "target", "label", "class", "outcome", "churn", "default", "response",
        "sales", "revenue", "profit", "demand", "prediction", "forecast",
    ]
    for col in df.columns:
        name = str(col).lower()
        if any(token in name for token in strong_names):
            return col
    if "churn" in df.columns:
        return "churn"
    return None


def build_preprocessing_decisions(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    target = target_column or _infer_target_column(df)
    predictors = df.drop(columns=[target]) if target in df.columns else df.copy()
    predictors, dropped_features = _drop_unusable_features(predictors, target=target)
    numeric_cols = list(predictors.select_dtypes(include=[np.number]).columns)
    categorical_cols = [
        col for col in predictors.columns if col not in numeric_cols and (pd.api.types.is_object_dtype(predictors[col]) or pd.api.types.is_string_dtype(predictors[col]) or isinstance(predictors[col].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(predictors[col]))
    ]
    decisions = []
    leakage_features = [
        column for column in dropped_features
        if str(target).strip().lower() == "sales"
        and str(column).strip().lower().replace(" ", "_") in {
            "cost", "profit", "profit_margin", "net_profit", "gross_profit",
        }
    ]
    if leakage_features:
        decisions.append({
            "step": "Target leakage",
            "method": "Exclude post-sale fields from sales predictors",
            "reason": "Cost, profit, and profit margin are generally known only after a sale and can reveal the future sales outcome.",
            "columns": leakage_features,
        })
    if numeric_cols:
        decisions.append({
            "step": "Missing data",
            "method": "Median imputation for numeric columns",
            "reason": "Median is robust to outliers and preserves scale without introducing extreme values.",
            "columns": numeric_cols,
        })
    if categorical_cols:
        decisions.append({
            "step": "Missing data",
            "method": "Most-frequent imputation for categorical columns",
            "reason": "Mode-based imputation preserves the dominant category when values are missing.",
            "columns": categorical_cols,
        })
    if numeric_cols:
        if target and target in df.columns and df[target].nunique(dropna=True) <= 10:
            decisions.append({
                "step": "Scaling",
                "method": "StandardScaler",
                "reason": "Standard scaling helps linear models and supports stable coefficient interpretation.",
                "columns": numeric_cols,
            })
        else:
            decisions.append({
                "step": "Scaling",
                "method": "RobustScaler",
                "reason": "Robust scaling reduces the effect of outliers when the data distribution is skewed.",
                "columns": numeric_cols,
            })
    if categorical_cols:
        decisions.append({
            "step": "Encoding",
            "method": "One-hot encoding",
            "reason": "This avoids imposing arbitrary ordinal assumptions on nominal customer attributes.",
            "columns": categorical_cols,
        })
    if target and target in df.columns and df[target].nunique() == 2:
        ratio = float(df[target].value_counts(normalize=True).min() / max(df[target].value_counts(normalize=True).max(), 1e-9))
        if ratio < 0.75:
            decisions.append({
                "step": "Imbalance handling",
                "method": "Class weights",
                "reason": "A balanced class weighting policy reduces the tendency to over-prioritize the majority class.",
                "columns": [target],
            })
    return {
        "target": target,
        "decisions": decisions,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dropped_features": dropped_features,
    }


def _create_feature_if(df: pd.DataFrame, new_name: str, source_columns: list[str], func) -> tuple[pd.DataFrame, str | None]:
    available = [col for col in source_columns if col in df.columns]
    if not available:
        return df, None
    if len(available) == 1:
        value = func(df[available[0]])
    else:
        series = pd.concat([df[col].astype(float) for col in available], axis=1)
        value = func(series)
    df[new_name] = value
    return df, new_name


def engineer_features(df: pd.DataFrame, target_column: str | None = None) -> pd.DataFrame:
    target = target_column if target_column in df.columns else _infer_target_column(df)
    target_values = df[target].copy() if target else None
    target_position = int(df.columns.get_loc(target)) if target else 0
    work = df.drop(columns=[target]).copy() if target else df.copy()
    descriptions: dict[str, str] = {}

    if "customer_tenure" not in work.columns and any(col in work.columns for col in ["tenure_months", "customer_tenure"]):
        source = "tenure_months" if "tenure_months" in work.columns else "customer_tenure"
        work["customer_tenure"] = pd.to_numeric(work[source], errors="coerce")
        descriptions["customer_tenure"] = "Customer tenure in months derived from the available tenure field."

    if "purchase_frequency" not in work.columns:
        for source in ["purchase_frequency", "orders_last_6m", "orders_last_12m", "num_orders", "order_count"]:
            if source in work.columns:
                work["purchase_frequency"] = pd.to_numeric(work[source], errors="coerce")
                descriptions["purchase_frequency"] = "How often the customer places orders over the observed period."
                break

    if "avg_order_value" not in work.columns:
        if "avg_order_value" in work.columns:
            pass
        elif "total_spend" in work.columns and "purchase_frequency" in work.columns:
            work["avg_order_value"] = work["total_spend"] / work["purchase_frequency"].replace(0, np.nan)
            descriptions["avg_order_value"] = "Average spend per purchase, derived from total spending and purchase frequency."
        elif any(col in work.columns for col in ["sales", "revenue", "amount", "total_sales", "value"]):
            source = next(col for col in ["sales", "revenue", "amount", "total_sales", "value"] if col in work.columns)
            work["avg_order_value"] = pd.to_numeric(work[source], errors="coerce")
            descriptions["avg_order_value"] = "Average transactional value from the available spending metric."

    if "total_spend" not in work.columns:
        for source in ["total_spend", "revenue", "sales", "amount", "total_sales"]:
            if source in work.columns:
                work["total_spend"] = pd.to_numeric(work[source], errors="coerce")
                descriptions["total_spend"] = "Total customer spend measured over the period available in the dataset."
                break

    if "recency_days" not in work.columns:
        for source in ["recency_days", "days_since_last_order", "days_since_last_purchase", "recency"]:
            if source in work.columns:
                work["recency_days"] = pd.to_numeric(work[source], errors="coerce")
                descriptions["recency_days"] = "Time since the most recent observed purchase or engagement event."
                break

    if "engagement_score" not in work.columns:
        score_cols = [col for col in work.columns if any(token in str(col).lower() for token in ["visit", "engagement", "session", "login", "interaction"]) ]
        if score_cols:
            score_cols = [col for col in score_cols if pd.api.types.is_numeric_dtype(work[col])]
            if score_cols:
                work["engagement_score"] = work[score_cols].sum(axis=1, skipna=True)
                descriptions["engagement_score"] = "Composite engagement score produced from available interaction/activity metrics."

    if "support_indicator" not in work.columns:
        for source in ["support_calls", "support_requests", "complaints", "complaint_count", "service_calls"]:
            if source in work.columns:
                work["support_indicator"] = pd.to_numeric(work[source], errors="coerce")
                descriptions["support_indicator"] = "Customer support burden or complaint volume indicator."
                break

    if "return_rate" not in work.columns:
        if "returns" in work.columns and "orders" in work.columns:
            work["return_rate"] = work["returns"] / work["orders"].replace(0, np.nan)
            descriptions["return_rate"] = "Return ratio derived from returns and total number of orders."

    if "monitoring_index" not in work.columns and "purchase_frequency" in work.columns and "support_indicator" in work.columns:
        work["monitoring_index"] = work["purchase_frequency"] - work["support_indicator"].fillna(0)
        descriptions["monitoring_index"] = "A simple behavioral index combining order cadence and support burden."

    if target_values is not None:
        work.insert(target_position, target, target_values)
    work.attrs["feature_descriptions"] = descriptions
    if target and target in work.columns:
        work.attrs["target_column"] = target
    return work


def _normalize_model_features(X: pd.DataFrame) -> pd.DataFrame:
    normalized = X.copy()
    numeric_columns = normalized.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in normalized.columns if column not in numeric_columns]

    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype(np.float64)
    for column in categorical_columns:
        values = normalized[column].astype(object)
        normalized[column] = values.where(pd.notna(values), np.nan)

    if any(pd.api.types.is_extension_array_dtype(dtype) for dtype in normalized.dtypes):
        raise ValueError("Feature preprocessing could not normalize pandas nullable dtypes.")
    if any(
        value is pd.NA
        for column in categorical_columns
        for value in normalized[column].to_numpy(dtype=object, copy=False)
    ):
        raise ValueError("Feature preprocessing could not normalize pandas nullable values.")
    return normalized


def _prepare_model_data(df: pd.DataFrame, target_column: str | None = None) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    target = target_column or _infer_target_column(df)
    if not target or target not in df.columns:
        raise ValueError("A valid target column is required for supervised modeling.")
    work = engineer_features(df, target_column=target)
    work = work.dropna(subset=[target]).copy()
    X, dropped_features = _drop_unusable_features(
        work.drop(columns=[target]), target=target
    )
    X = _normalize_model_features(X)
    y = work[target]
    if pd.api.types.is_numeric_dtype(y.dtype):
        y = pd.Series(
            pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan),
            index=y.index,
            name=y.name,
        )
    else:
        y = y.astype(object)
    if X.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two distinct values.")

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = [col for col in X.columns if col not in num_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ],
        remainder="drop",
    )
    info = {
        "target": target,
        "task": infer_task(y),
        "dropped_features": dropped_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": preprocessor,
    }
    return X, y, info


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None, is_classifier: bool = True) -> dict[str, float | None]:
    if not is_classifier:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
        }
    metrics: dict[str, float | None] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if is_classifier and y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = _safe_roc_auc(y_true, y_proba)
        metrics["PR-AUC"] = _safe_pr_auc(y_true, y_proba)
    return metrics


def _safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float | None:
    if y_true.nunique() != 2:
        return None
    values = y_true.astype(str)
    unique = sorted(values.unique())
    mapping = {unique[0]: 0, unique[1]: 1}
    y_bin = values.map(mapping)
    prob = np.asarray(y_proba)
    if prob.ndim > 1:
        prob = prob[:, 1]
    try:
        return float(roc_auc_score(y_bin, prob))
    except Exception:
        return None


def _safe_pr_auc(y_true: pd.Series, y_proba: np.ndarray) -> float | None:
    if y_true.nunique() != 2:
        return None
    values = y_true.astype(str)
    unique = sorted(values.unique())
    mapping = {unique[0]: 0, unique[1]: 1}
    y_bin = values.map(mapping)
    prob = np.asarray(y_proba)
    if prob.ndim > 1:
        prob = prob[:, 1]
    try:
        return float(average_precision_score(y_bin, prob))
    except Exception:
        return None


def build_hybrid_model(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    target = target or _infer_target_column(df)
    if not target:
        return {"ok": False, "error": "No target column could be inferred for the hybrid model."}
    try:
        X, y, info = _prepare_model_data(df, target_column=target)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    task = info["task"]
    try:
        X_train, X_test, y_train, y_test = _safe_split(
            X, y, task, test_size=0.25, random_state=42
        )
    except ValueError as exc:
        return {"ok": False, "error": f"Hybrid model split failed: {exc}"}
    preprocessor = info["preprocessor"]
    try:
        if task == "classification":
            base_models = [
                ("lr", Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])),
                ("rf", Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"))])),
                ("gb", Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingClassifier(random_state=42))])),
            ]
            min_class_count = int(y_train.value_counts().min())
            if min_class_count >= 2:
                hybrid = StackingClassifier(
                    estimators=base_models,
                    final_estimator=LogisticRegression(max_iter=1000),
                    stack_method="predict_proba",
                    cv=min(5, min_class_count),
                    passthrough=False,
                )
            else:
                hybrid = VotingClassifier(estimators=base_models, voting="soft")
        else:
            base_models = [
                ("linear", Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])),
                ("rf", Pipeline([("preprocessor", preprocessor), ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))])),
                ("gb", Pipeline([("preprocessor", preprocessor), ("model", GradientBoostingRegressor(random_state=42))])),
            ]
            hybrid = StackingRegressor(
                estimators=base_models,
                final_estimator=Ridge(),
                cv=min(5, max(2, len(X_train) // 2)),
                passthrough=False,
            )
        started = time.perf_counter()
        hybrid.fit(X_train, y_train)
        training_time = time.perf_counter() - started

        preds = hybrid.predict(X_test)
        proba = hybrid.predict_proba(X_test) if task == "classification" else None
        metrics = _evaluate_predictions(y_test, preds, proba, is_classifier=task == "classification")
        metrics["Training Time"] = float(training_time)
        result = {
            "ok": True,
            "model_name": "RetailIQ Proposed Hybrid Ensemble",
            "model": hybrid,
            "target": target,
            "task": task,
            "metrics": metrics,
            "predictions": preds,
            "probabilities": proba,
            "training_time": training_time,
            "y_test": y_test,
            "X_train": X_train,
            "X_test": X_test,
            "dropped_features": info["dropped_features"],
        }
        if task == "classification":
            result["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
        return result
    except Exception:
        logger.exception("Hybrid model training failed for target %s", target)
        return {
            "ok": False,
            "error": "Hybrid model training failed. Check the dataset's target and class distribution, then review application logs.",
        }


def _single_model_metrics(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    task: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    metrics = {"Model": model_name, "Training Time": float(elapsed)}
    metrics.update(_evaluate_predictions(y_test, preds, proba, is_classifier=task == "classification"))
    if task == "classification":
        metrics["ROC-AUC"] = _safe_roc_auc(y_test, proba) if proba is not None else None
        metrics["PR-AUC"] = _safe_pr_auc(y_test, proba) if proba is not None else None
        metrics["Confusion Matrix"] = confusion_matrix(y_test, preds).tolist()
    return metrics


def compare_models(df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
    target = target or _infer_target_column(df)
    if not target or target not in df.columns:
        return {"ok": False, "error": "A valid target column was not found."}
    try:
        X, y, info = _prepare_model_data(df, target)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    task = info["task"]
    try:
        X_train, X_test, y_train, y_test = _safe_split(
            X, y, task, test_size=0.25, random_state=42
        )
    except ValueError as exc:
        return {"ok": False, "error": f"Model comparison split failed: {exc}"}
    if task == "classification":
        specs = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
    else:
        specs = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
    model_specs = {
        name: Pipeline([("preprocessor", info["preprocessor"]), ("model", estimator)])
        for name, estimator in specs.items()
    }
    results = []
    successful_models = 0
    for name, model in model_specs.items():
        try:
            results.append(_single_model_metrics(model, X_train, X_test, y_train, y_test, name, task))
            successful_models += 1
        except Exception:
            logger.exception("Comparison model %s failed for target %s", name, target)
            results.append({"Model": name, "Error": "Model training failed; see application logs."})

    hybrid = build_hybrid_model(df, target=target)
    if hybrid.get("ok"):
        hybrid_row = {
            "Model": "Proposed Hybrid",
            "Training Time": hybrid["metrics"].get("Training Time"),
            **hybrid["metrics"],
        }
        if task == "classification":
            hybrid_row["Confusion Matrix"] = hybrid["confusion_matrix"]
        results.append(hybrid_row)
        successful_models += 1
    comparison_df = pd.DataFrame(results)
    if successful_models == 0:
        return {
            "ok": False,
            "error": hybrid.get("error", "No comparison models could be trained."),
        }
    comparison_df["Proposed Hybrid"] = comparison_df["Model"].eq("Proposed Hybrid").astype(int)
    if task == "classification":
        best_overall = comparison_df.dropna(subset=["F1"]).sort_values(
            ["F1", "ROC-AUC"], ascending=[False, False], na_position="last"
        ).iloc[0].to_dict()
    else:
        best_overall = comparison_df.dropna(subset=["RMSE"]).sort_values(
            ["RMSE", "MAE"], ascending=[True, True]
        ).iloc[0].to_dict()
    return {
        "ok": True,
        "target": target,
        "task": task,
        "models": comparison_df,
        "best_overall": best_overall,
        "highlight": "RetailIQ Proposed Hybrid Ensemble",
        "hybrid": hybrid,
    }


def _explanation_features(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    if isinstance(model, Pipeline) and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        explanation_model = model.steps[-1][1]
        train_features = preprocessor.transform(_normalize_model_features(X_train))
        test_features = preprocessor.transform(_normalize_model_features(X_test))
        if hasattr(train_features, "toarray"):
            train_features = train_features.toarray()
        if hasattr(test_features, "toarray"):
            test_features = test_features.toarray()
        return (
            explanation_model,
            np.asarray(train_features),
            np.asarray(test_features),
            list(preprocessor.get_feature_names_out()),
        )
    return model, X_train.to_numpy(), X_test.to_numpy(), list(X_test.columns)


def fit_explanation_model(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.25,
) -> dict[str, Any]:
    try:
        X, y, info = _prepare_model_data(df, target)
        task = info["task"]
        X_train, X_test, y_train, y_test = _safe_split(
            X, y, task, test_size=test_size, random_state=42
        )
        estimator = (
            LogisticRegression(max_iter=1000, class_weight="balanced")
            if task == "classification"
            else RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        )
        model = Pipeline([
            ("preprocessor", info["preprocessor"]),
            ("model", estimator),
        ])
        model.fit(X_train, y_train)
        return {
            "ok": True,
            "model": model,
            "task": task,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "dropped_features": info["dropped_features"],
        }
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:
        logger.exception("Could not fit the explainability model for target %s", target)
        return {
            "ok": False,
            "error": "The explainability model could not be trained for this target. Review the target and feature data.",
        }


def generate_shap_explanation(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    customer_index: int = 0,
    class_index: int | None = None,
) -> dict[str, Any]:
    if shap is None:
        return {"ok": False, "error": "SHAP is not installed. Install shap to enable explainable AI."}
    try:
        if not 0 <= customer_index < len(X_test):
            return {"ok": False, "error": "The selected observation is outside the evaluation data."}
        explanation_model, train_features, test_features, feature_names = _explanation_features(model, X_train, X_test)
        classifier = is_classifier(explanation_model)
        if classifier:
            explainer = shap.Explainer(
                explanation_model, train_features, feature_names=feature_names
            )
            values = explainer(test_features)
        else:
            explainer = shap.Explainer(
                explanation_model.predict,
                train_features,
                feature_names=feature_names,
                algorithm="permutation",
            )
            values = explainer(
                test_features, max_evals=2 * len(feature_names) + 1
            )
        feature_values = np.asarray(values.values[customer_index])
        if feature_values.ndim == 2:
            selected_class = -1 if class_index is None else class_index
            if classifier and not -feature_values.shape[1] <= selected_class < feature_values.shape[1]:
                return {"ok": False, "error": "The selected class is not supported by this SHAP explanation."}
            feature_values = feature_values[:, selected_class] if classifier else feature_values[:, 0]
        if feature_values.ndim != 1 or len(feature_values) != len(feature_names):
            return {"ok": False, "error": "SHAP returned an unsupported explanation shape for this model."}
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "contribution": np.asarray(feature_values, dtype=float),
        }).sort_values("contribution", ascending=False)
        top_positive = feature_importance.head(3).to_dict("records")
        top_negative = feature_importance.tail(3).sort_values("contribution").to_dict("records")
        predicted = model.predict(X_test.iloc[[customer_index]])[0]
        result = {
            "ok": True,
            "task": "classification" if classifier else "regression",
            "prediction": str(predicted) if classifier else float(predicted),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "feature_importance": feature_importance,
            "summary": "SHAP shows how each feature contributes to this model's prediction; contributions are not causal effects.",
        }
        if classifier and hasattr(model, "predict_proba"):
            result["probabilities"] = model.predict_proba(X_test.iloc[[customer_index]])[0].tolist()
            result["classes"] = model.named_steps["model"].classes_.tolist()
        return result
    except Exception:
        logger.exception("SHAP explanation failed")
        return {
            "ok": False,
            "error": "SHAP could not explain this model and dataset combination. Review application logs for details.",
        }


def generate_lime_explanation(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, customer_index: int = 0, target_name: str | None = None) -> dict[str, Any]:
    if LimeTabularExplainer is None:
        return {"ok": False, "error": "LIME is not installed. Install lime to enable local explanations."}
    try:
        if not 0 <= customer_index < len(X_test):
            return {"ok": False, "error": "The selected observation is outside the evaluation data."}
        explanation_model, train_features, test_features, feature_names = _explanation_features(model, X_train, X_test)
        classifier = is_classifier(explanation_model)
        class_names = (
            [str(value) for value in explanation_model.classes_]
            if classifier and hasattr(explanation_model, "classes_")
            else None
        )
        explainer = LimeTabularExplainer(
            train_features,
            feature_names=feature_names,
            class_names=class_names,
            mode="classification" if classifier else "regression",
        )
        instance = test_features[customer_index]
        predict_fn = explanation_model.predict_proba if classifier else explanation_model.predict
        observation = X_test.iloc[[customer_index]]
        class_index = (
            int(np.argmax(model.predict_proba(observation)[0]))
            if classifier and hasattr(model, "predict_proba")
            else None
        )
        explanation = explainer.explain_instance(
            instance,
            predict_fn,
            labels=(class_index,) if class_index is not None else (1,),
            num_features=5,
            num_samples=500,
        )
        predicted = model.predict(observation)[0]
        result = {
            "ok": True,
            "task": "classification" if classifier else "regression",
            "prediction": str(predicted) if classifier else float(predicted),
            "top_features": explanation.as_list(
                label=class_index
            ) if class_index is not None else explanation.as_list(),
            "summary": "LIME locally approximates the fitted model around this observation; the explanation is not causal.",
            "target_name": target_name,
        }
        if classifier and hasattr(model, "predict_proba"):
            result["probabilities"] = model.predict_proba(observation)[0].tolist()
            result["classes"] = class_names
        return result
    except Exception:
        logger.exception("LIME explanation failed")
        return {
            "ok": False,
            "error": "LIME could not explain this model and dataset combination. Review application logs for details.",
        }


def build_data_story(df: pd.DataFrame, model_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summarize_dataset(df)
    risk_share = 0.0
    if summary["target_column"] and summary["target_column"] in df.columns:
        counts = df[summary["target_column"]].value_counts(normalize=True)
        if len(counts) >= 2:
            risk_share = float(counts.iloc[1])
    story = {
        "what_happened": f"The current dataset contains {summary['rows']} rows and {summary['columns']} columns, with {summary['missing_total']} missing cells and {summary['duplicate_rows']} duplicate rows detected.",
        "why_it_happened": "The strongest business drivers are expected to emerge from customer behavior variables, spending patterns, and support activity when they are present in the uploaded dataset.",
        "what_is_likely_to_happen": "The model comparison will report task-appropriate holdout metrics for the selected target.",
        "what_should_we_do": "Prioritize retention outreach for high-risk customers, validate feature trends with business stakeholders, and monitor the model as more data becomes available.",
        "risk_share": risk_share,
    }
    if model_summary and model_summary.get("models") is not None:
        models = model_summary["models"]
        if not models.empty:
            if model_summary.get("task") == "regression" and "RMSE" in models.columns:
                valid_models = models.dropna(subset=["RMSE"]).sort_values("RMSE")
                if not valid_models.empty:
                    best = valid_models.iloc[0]
                    story["what_is_likely_to_happen"] = (
                        f"The lowest holdout RMSE is from {best['Model']} "
                        f"({best['RMSE']:.3f}); validate this estimate on representative future data."
                    )
            elif "F1" in models.columns:
                valid_models = models.dropna(subset=["F1"]).sort_values(
                    ["F1", "ROC-AUC"], ascending=[False, False], na_position="last"
                )
                if not valid_models.empty:
                    best = valid_models.iloc[0]
                    story["what_is_likely_to_happen"] = (
                        f"The highest-performing classifier is {best['Model']} with "
                        f"weighted F1 {best['F1']:.3f} on the holdout set."
                    )
    return story


def build_customer_recommendation(probability: float, risk_label: str, drivers: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    drivers = drivers or []
    intensity = "High Risk" if probability >= 0.65 else "Medium Risk" if probability >= 0.35 else "Low Risk"
    if intensity == "High Risk":
        action = "Launch retention outreach, offer personalized incentives, and escalate to customer support follow-up."
    elif intensity == "Medium Risk":
        action = "Monitor engagement closely and trigger targeted retention messaging if usage continues to decline."
    else:
        action = "Maintain the current relationship plan and focus upsell or loyalty actions on the healthiest segments."
    return {
        "risk_level": intensity,
        "probability": float(probability),
        "action": action,
        "drivers": drivers,
    }


def positive_risk_class_index(classes: list[Any]) -> int | None:
    """Return a semantically identified positive class instead of guessing."""
    positive_labels = {
        "1", "1.0", "yes", "true", "churn", "churned", "high", "high risk",
        "at risk", "positive", "default", "fraud", "attrited", "exited",
        "left", "delinquent",
    }
    normalized = [
        " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
        for value in classes
    ]
    matching = [index for index, value in enumerate(normalized) if value in positive_labels]
    return matching[0] if len(matching) == 1 else None
