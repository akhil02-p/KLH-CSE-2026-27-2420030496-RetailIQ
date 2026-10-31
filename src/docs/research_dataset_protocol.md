# RetailIQ Expanded Research Dataset Protocol

## Purpose
This suite is the next-stage evaluation set for the research paper. It is deliberately separated into public benchmark datasets and clearly labeled synthetic retail-domain datasets.

## Dataset groups
- **Classification:** Breast Cancer, Wine, Iris, Digits.
- **Regression:** Diabetes.
- **Unsupervised:** Iris, Wine, Digits with reference targets removed.
- **Synthetic retail-domain validation:** retail sales regression and customer churn classification generated with seed 42.

The public datasets are sourced from scikit-learn's packaged benchmark datasets; the UCI provenance of several datasets is documented in the project references. The synthetic retail datasets are not presented as external real-world evidence.

## Evaluation design
- Automatic routing is evaluated against the known task label in the manifest.
- Supervised learning uses **5 repeats × 5 folds** with shuffled stratified folds for classification and shuffled K-fold for regression.
- All imputation, encoding, and scaling are fitted inside each training fold through scikit-learn pipelines to reduce leakage.
- Baselines: Logistic Regression / Decision Tree / Random Forest for classification; Linear Regression / Decision Tree / Random Forest for regression.
- Primary classification metric for model comparison: weighted F1.
- Primary regression metric for model comparison: RMSE.
- Clustering reports Silhouette, Davies–Bouldin, Calinski–Harabasz, inertia, and runtime across K=2..6.
- Friedman tests are used for repeated fold comparisons; Wilcoxon signed-rank tests are only added after a significant Friedman test, with Holm correction for multiple pairwise comparisons.

## Important interpretation rule
The suite measures whether RetailIQ can route and benchmark unfamiliar tabular datasets reproducibly. It does **not** prove that RetailIQ's models are globally optimal, nor does it establish superiority over every AutoML or analytics system.
