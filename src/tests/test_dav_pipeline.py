import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.dav_support import (
    build_project_definition,
    summarize_dataset,
    engineer_features,
    _prepare_model_data,
    build_hybrid_model,
    build_preprocessing_decisions,
    compare_models,
    build_target_distribution_data,
    fit_explanation_model,
    _safe_pr_auc,
    _safe_roc_auc,
    generate_lime_explanation,
    generate_shap_explanation,
    positive_risk_class_index,
)
from src.advanced_ml import run_advanced_predictive
from src.ml_analyzer import _safe_split, infer_task, run_anomaly_detection, run_clustering, run_supervised
from src.cleaner import build_cleaning_plan, clean_dataset


def _synthetic_churn_data():
    rows = []
    for i in range(200):
        tenure = 6 + (i % 24)
        spend = 100 + (i % 10) * 25 + (i % 4) * 18
        support_calls = (i % 5)
        purchase_freq = 2 + (i % 6)
        churn = 1 if (tenure < 12 and support_calls >= 3) or (i % 11 == 0) else 0
        rows.append({
            "customer_id": f"C{i:03d}",
            "tenure_months": tenure,
            "avg_order_value": spend,
            "total_spend": spend * purchase_freq,
            "support_calls": support_calls,
            "purchase_frequency": purchase_freq,
            "segment": "High Value" if i % 3 == 0 else "Standard",
            "churn": churn,
        })
    return pd.DataFrame(rows)


def test_project_definition_has_required_sections():
    definition = build_project_definition()
    assert "problem_statement" in definition
    assert "business_objective" in definition
    assert "target_variable" in definition
    assert "research_questions" in definition


def test_dataset_summary_and_feature_engineering_work():
    df = _synthetic_churn_data()
    summary = summarize_dataset(df)
    assert summary["rows"] == 200
    assert summary["target_column"] == "churn"
    engineered = engineer_features(df)
    assert "customer_tenure" in engineered.columns or "tenure_months" in engineered.columns
    assert "churn" in engineered.columns


def test_hybrid_model_build_and_comparison_are_valid():
    df = _synthetic_churn_data()
    result = build_hybrid_model(df, target="churn")
    assert result["ok"] is True
    assert "RetailIQ Proposed Hybrid Ensemble" in result["model_name"]
    y_test = result["y_test"]
    predictions = result["predictions"]
    probabilities = result["probabilities"]
    assert result["confusion_matrix"] == confusion_matrix(y_test, predictions).tolist()
    assert result["metrics"]["Accuracy"] == accuracy_score(y_test, predictions)
    assert result["metrics"]["Precision"] == precision_score(y_test, predictions, average="weighted", zero_division=0)
    assert result["metrics"]["Recall"] == recall_score(y_test, predictions, average="weighted", zero_division=0)
    assert result["metrics"]["F1"] == f1_score(y_test, predictions, average="weighted", zero_division=0)
    assert result["metrics"]["ROC-AUC"] == _safe_roc_auc(y_test, probabilities)
    assert result["metrics"]["PR-AUC"] == _safe_pr_auc(y_test, probabilities)
    comparison = compare_models(df, target="churn")
    assert comparison["ok"] is True
    assert "Proposed Hybrid" in comparison["models"].columns
    comparison_hybrid = comparison["hybrid"]
    hybrid_row = comparison["models"].set_index("Model").loc["Proposed Hybrid"]
    assert comparison_hybrid["confusion_matrix"] == result["confusion_matrix"]
    for metric in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"):
        assert comparison_hybrid["metrics"][metric] == result["metrics"][metric]
        assert hybrid_row[metric] == result["metrics"][metric]


def test_model_preparation_normalizes_pandas_nullable_values():
    df = pd.DataFrame({
        "churn": ["yes", "no"] * 20,
        "spend": pd.Series([100, pd.NA, 120, 130] * 10, dtype="Int64"),
        "segment": pd.Series(["value", pd.NA, "standard", "value"] * 10, dtype="string"),
    })

    X, y, info = _prepare_model_data(df, target_column="churn")
    transformed = info["preprocessor"].fit_transform(X, y)

    assert X["spend"].dtype == "float64"
    assert X["segment"].dtype == object
    assert not any(pd.api.types.is_extension_array_dtype(dtype) for dtype in X.dtypes)
    assert not any(value is pd.NA for value in X["segment"].to_numpy(dtype=object))
    assert transformed.shape[0] == len(df)


def test_pipeline_explanations_support_encoded_categorical_features():
    df = _synthetic_churn_data().head(80)
    X, y, info = _prepare_model_data(df, target_column="churn")
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("preprocessor", info["preprocessor"]),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)

    shap_result = generate_shap_explanation(model, X_train, X_test)
    lime_result = generate_lime_explanation(model, X_train, X_test)

    assert shap_result["ok"] is True
    assert lime_result["ok"] is True


def test_target_is_never_reused_as_an_engineered_predictor():
    df = pd.DataFrame({
        "sales": [100.0, 150.0, 200.0, 250.0],
        "orders_last_6m": [1, 2, 3, 4],
        "support_calls": [0, 1, 0, 2],
    })

    engineered = engineer_features(df)

    assert "sales" in engineered.columns
    assert engineered.attrs["target_column"] == "sales"
    assert "avg_order_value" not in engineered.columns
    assert "total_spend" not in engineered.columns
    assert "sales" not in build_preprocessing_decisions(df, "sales")["numeric_columns"]


def test_advanced_model_fits_imputer_only_on_training_rows(monkeypatch):
    from sklearn.impute import SimpleImputer

    train_sizes = []
    original_fit = SimpleImputer.fit

    def recording_fit(self, values, y=None):
        train_sizes.append(len(values))
        return original_fit(self, values, y)

    monkeypatch.setattr(SimpleImputer, "fit", recording_fit)
    df = pd.DataFrame({
        "metric": [value % 10 for value in range(80)],
        "segment": pd.Series(["North", "South", pd.NA, "West"] * 20, dtype="string"),
        "target": [value * 1.5 + (value % 7) for value in range(80)],
    })

    result = run_advanced_predictive(df, target="target", task="regression")

    assert result["ok"] is True
    assert len(train_sizes) == 2
    assert set(train_sizes) == {64}
    assert set(result["importance"]["Feature"]) == {"metric", "segment"}


def test_unsupervised_models_exclude_reference_labels_and_identifiers():
    df = pd.DataFrame({
        "customer_id": range(100),
        "feature_a": [float(value % 13) for value in range(100)],
        "feature_b": [float((value * 7) % 31) for value in range(100)],
        "is_anomaly": [1 if value % 10 == 0 else 0 for value in range(100)],
    })

    anomaly = run_anomaly_detection(df)
    clustering = run_clustering(df, n_clusters=3)

    assert anomaly["ok"] is True
    assert clustering["ok"] is True
    assert "is_anomaly" not in anomaly["features"]
    assert "customer_id" not in anomaly["features"]
    assert "is_anomaly" not in clustering["features"]
    assert "customer_id" not in clustering["features"]
    assert run_anomaly_detection(df, contamination=0)["ok"] is False


def _sales_regression_data(rows=120):
    values = range(rows)
    sales = [250 + value * 11.5 + (value % 7) * 2.25 for value in values]
    return pd.DataFrame({
        "sales": sales,
        "cost": [value * 0.58 for value in sales],
        "profit": [value * 0.42 for value in sales],
        "profit_margin": [0.42] * rows,
        "advertising_spend": [value * 0.03 + 5.5 for value in range(rows)],
        "row_id": list(values),
        "units": pd.Series(pd.array([value % 17 if value % 19 else pd.NA for value in values], dtype="Int64")),
        "store": pd.Series(["North", "South", pd.NA, "West"] * (rows // 4), dtype="string"),
    })


def test_dav_model_paths_support_sales_regression_without_post_sale_leakage():
    df = _sales_regression_data()

    assert infer_task(df["sales"]) == "regression"

    supervised = run_supervised(df, "sales")
    assert supervised["ok"] is True
    assert {"MAE", "RMSE", "R2"}.issubset(supervised["results"].columns)
    assert {"cost", "profit", "profit_margin"}.issubset(supervised["dropped_features"])
    assert "advertising_spend" not in supervised["dropped_features"]
    assert "row_id" in supervised["dropped_features"]

    comparison = compare_models(df, target="sales")
    assert comparison["ok"] is True
    assert comparison["task"] == "regression"
    assert {"MAE", "RMSE", "R2"}.issubset(comparison["models"].columns)
    assert "Proposed Hybrid" in comparison["models"]["Model"].values

    hybrid = build_hybrid_model(df, target="sales")
    assert hybrid["ok"] is True
    assert hybrid["task"] == "regression"
    assert {"MAE", "RMSE", "R2"}.issubset(hybrid["metrics"])
    assert "confusion_matrix" not in hybrid
    assert {"cost", "profit", "profit_margin"}.issubset(hybrid["dropped_features"])

    advanced = run_advanced_predictive(df, target="sales", task="regression")
    assert advanced["ok"] is True
    assert {"MAE", "RMSE", "R2"}.issubset(advanced["metrics"])
    assert {"cost", "profit", "profit_margin"}.issubset(advanced["dropped_features"])

    preprocessing = build_preprocessing_decisions(df, "sales")
    assert not {"cost", "profit", "profit_margin"}.intersection(preprocessing["numeric_columns"])
    assert any(decision["step"] == "Target leakage" for decision in preprocessing["decisions"])


def test_nullable_feature_values_do_not_raise_ambiguous_na_errors():
    result = run_supervised(_sales_regression_data(), "sales")

    assert result["ok"] is True
    assert all("Error" not in row for row in result["results"].to_dict("records"))


def test_rare_class_split_falls_back_to_unstratified_holdout():
    X = pd.DataFrame({"feature": range(100)})
    y = pd.Series([0] * 99 + [1])

    X_train, X_test, y_train, y_test = _safe_split(X, y, "classification")

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)


def test_explainability_supports_regression_models():
    df = _sales_regression_data(80).drop(columns=["cost", "profit", "profit_margin"])
    fitted = fit_explanation_model(df, "sales")

    assert fitted["ok"] is True
    assert fitted["task"] == "regression"
    assert fitted["model"].predict(fitted["X_test"]).shape == (len(fitted["X_test"]),)

    shap_result = generate_shap_explanation(
        fitted["model"], fitted["X_train"], fitted["X_test"]
    )
    lime_result = generate_lime_explanation(
        fitted["model"], fitted["X_train"], fitted["X_test"], target_name="sales"
    )

    assert shap_result["ok"] is True
    assert shap_result["task"] == "regression"
    assert lime_result["ok"] is True
    assert lime_result["task"] == "regression"


def test_raw_and_cleaned_duplicate_counts_are_calculated_from_each_frame():
    raw = pd.DataFrame({
        "Store Name": ["North", "North", "South", "South"],
        "Sales": [100, 100, 200, 200],
    })

    cleaned, _ = clean_dataset(raw, build_cleaning_plan(raw))

    assert len(raw) == 4
    assert int(raw.duplicated().sum()) == 2
    assert len(cleaned) == 2
    assert int(cleaned.duplicated().sum()) == 0


def test_target_distribution_data_keeps_continuous_values_for_histograms():
    target = pd.Series(
        pd.array([31_900.89 + value * 100 for value in range(367)], dtype="Float64"),
        name="sales",
    )
    target.iloc[10] = pd.NA

    task, distribution = build_target_distribution_data(target)

    assert task == "regression"
    assert list(distribution.columns) == ["Target value"]
    assert len(distribution) == 366
    assert distribution["Target value"].min() == 31_900.89
    assert distribution["Target value"].max() == 68_500.89


def test_target_distribution_data_counts_categorical_classes():
    target = pd.Series(["No", "Yes", "No", "Yes", pd.NA] * 73, dtype="string")

    task, distribution = build_target_distribution_data(target)

    assert task == "classification"
    assert distribution.set_index("Target class")["Rows"].to_dict() == {"No": 146, "Yes": 146}


def test_risk_positive_class_is_not_guessed_from_unknown_labels():
    assert positive_risk_class_index(["No", "Yes"]) == 1
    assert positive_risk_class_index(["No Churn", "Churn"]) == 1
    assert positive_risk_class_index(["Class A", "Class B"]) is None
