# RetailIQ Research Evaluation Methodology

## Purpose
Evaluate the reproducibility and adaptive routing behavior of RetailIQ rather than claim that one model is universally optimal.

## Datasets
The benchmark uses fixed scikit-learn datasets: Breast Cancer and Wine for classification, Diabetes for regression, and Iris with its target removed for unsupervised analysis.

## Supervised evaluation
Five-fold (or user-selected ten-fold) cross-validation is used. Preprocessing is fitted separately within each training fold to reduce leakage. Baseline models are Logistic Regression, Decision Tree, and Random Forest for classification; Linear Regression, Decision Tree, and Random Forest for regression.

## Metrics
Classification uses Accuracy, weighted Precision, weighted Recall, and weighted F1. Regression uses MAE, RMSE, and R2. Model selection is based on weighted F1 for classification and lowest RMSE for regression.

## Unsupervised evaluation
K-Means is evaluated for K=2 through K=5 using standardized numeric features. Silhouette score is the primary cluster-quality metric; inertia is reported as a secondary diagnostic.

## Statistical comparison
When sufficient fold-level results exist, a Friedman test is used across the model set. Pairwise Wilcoxon signed-rank tests are only generated after a significant Friedman result.

## Reproducibility
Random state is fixed at 42. All benchmark artifacts are written to `results/` as CSV files.
