# RetailIQ V0.7 — Machine Learning Methodology

## Purpose

RetailIQ extends its adaptive analytics workflow with transparent machine-learning baselines. V0.7 is designed to avoid treating every numeric column as a prediction target and to provide reproducible, interpretable benchmark results.

## Adaptive routing

1. Profile the uploaded tabular dataset.
2. Search for high-confidence target semantics such as `churn`, `sales`, `revenue`, `profit`, `demand`, `label`, or `target`.
3. Automatically select a target only when the semantic evidence is strong.
4. Otherwise, leave the target unset and recommend unsupervised analysis.
5. Allow a user to manually select a target when domain knowledge identifies one that cannot be inferred from the column name.

Weak names such as `value`, `amount`, `score`, and `quantity` are displayed only as suggestions and are never auto-selected. This prevents a legitimate feature such as `monetary_value` from being incorrectly interpreted as a regression target in a customer-segmentation dataset.

## Supervised learning

For a selected target, RetailIQ infers:

- Classification for categorical/binary targets and low-cardinality discrete targets.
- Regression for continuous numerical targets.

Preprocessing includes median imputation for numerical features, most-frequent imputation and one-hot encoding for categorical features, and calendar expansion for date-like columns. Identifier-like, constant, and near-unique fields are excluded to reduce noise and leakage risk.

### Baseline models

Classification:
- Logistic Regression
- Decision Tree
- Random Forest

Regression:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Evaluation

A reproducible hold-out split is used with random seed 42. Classification reports Accuracy, weighted Precision, weighted Recall, and weighted F1. Regression reports MAE, RMSE, and R².

The best baseline is selected by weighted F1 for classification and RMSE for regression. This is a benchmark convention, not a claim of global optimality.

## Explainability

V0.7 uses **permutation importance on the original input columns**. Each feature is shuffled repeatedly and the resulting performance change is measured. This avoids directly comparing raw coefficients whose magnitudes can be distorted by different feature scales and avoids presenting one-hot dummy variables as if they were independent business variables.

## Unsupervised learning

When no reliable target is selected, RetailIQ supports:

- K-Means clustering with standardized numeric features and Silhouette Score.
- Isolation Forest anomaly detection with a configurable contamination proportion.

## Reproducibility

The ML random seed is fixed at 42 for benchmark runs. Supervised benchmark metadata and model results can be appended to `results/ml_experiments.csv`, including timestamp, task, target, dataset row counts, feature counts, test size, dropped fields, seed, and model metrics.

## Suitability diagnostics

RetailIQ warns when datasets are small, have substantial missingness, or exhibit class imbalance. These warnings are intended to prevent overclaiming benchmark results in the research paper.

## Research interpretation

The ML module should be evaluated as one component of the broader adaptive analytics framework. The research evaluation should compare reproducibility, routing correctness, model performance, preprocessing behavior, and recommendation quality across multiple benchmark datasets rather than relying on one accuracy value.
