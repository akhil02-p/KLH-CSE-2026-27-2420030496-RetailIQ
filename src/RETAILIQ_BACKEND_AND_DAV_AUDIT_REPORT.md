# RetailIQ Backend and DAV Audit

**Review scope:** `RetailIQ_v08_work` only  
**Review type:** source, test, dataset-manifest, and interactive Streamlit audit  
**Conclusion:** The project implements substantial DAV-aligned workflow features, but the available evidence does not establish formal compliance with an external course or institutional rubric. Several execution/data-integrity issues found during the audit were fixed and verified.

## Executive summary

The application is a dataset-driven Streamlit analytics platform with 26 navigable pages, supervised and unsupervised ML workflows, task-aware model comparison and hybrid prediction, SHAP/LIME explanations, and research-evaluation artifacts. The codebase includes a documented churn use case, but the UI also supports general tabular datasets.

Reliability fixes cover upload/removal state, target-derived features, train/test imputation isolation, unsupervised label/identifier leakage, nullable data, regression-versus-classification routing, post-sale predictor leakage for sales, raw-versus-clean duplicate reporting, categorical preprocessing, and chart/logo presentation. Sales workflows retain legitimate continuous numeric predictors such as advertising spend. A continuous sales target is explicitly not treated as a categorical customer-risk outcome.

Final validation: **16 tests passed**, changed Python modules compiled, all **26/26** navigation destinations remain present, and browser upload/replacement/removal and logo navigation were verified. Browser screenshots showed a readable regression histogram and class-count chart in dark and light themes. On the available churn benchmark, Predictive Analytics, Model Comparison, Hybrid Model, Explainable AI, and Customer Risk Analysis rendered successfully without a traceback. The available retail sales and churn benchmarks completed supervised, advanced, comparison, hybrid, SHAP, and LIME checks. The exact requested files `RetailIQ_Master_Test_Dataset(1).csv`, `RetailIQ_Bar_Chart_Test_Dataset.csv`, and `RetailIQ_Customer_Churn_Demo.csv` were not available, and no official DAV rubric was found; therefore their exact-file behavior and formal DAV compliance are not established. The final test run emitted five non-failing warnings from date parsing and SHAP colormap deprecations.

## Issue-by-issue root causes and verification

| Issue | Root cause observed | Correction and evidence | Status |
|---|---|---|---|
| Regression target distribution showed dense vertical lines and an `index` axis | `value_counts()` treated a continuous target as hundreds of categories; `reset_index()` then exposed its target values under a generic `index` label. The available sales benchmark has 300 unique values between 31,900.89 and 183,109.72, so the prior 40k–180k ticks were sales amounts, not row counts. The unavailable master file's exact values cannot be confirmed. | Continuous targets now plot a histogram of the non-missing values from the cleaned target column, with the target value on X and row frequency on Y. Categorical targets retain class-count bars. Captured live screenshots show the labeled histogram in dark mode and class bars in light mode; tests check raw numeric values and nullable targets. | **Fixed and live-verified on available regression/classification datasets; exact master dataset unavailable.** |
| `boolean value of NA is ambiguous` in prediction | Nullable pandas extension dtypes can carry `pd.NA` into scikit-learn as an object value. Model preparation now converts nullable numeric features to `float64`, categorical values to object with `np.nan`, and fits imputers inside the training pipeline. The exact named failing CSV was not available to reproduce its traceback. | Regression test trains with nullable integer/string features and confirms no model error; the mixed-type advanced pipeline test confirms numeric and categorical imputers fit on training rows only. Available sales/churn datasets also complete predictive workflows (they contain no missing cells). | **Fix verified by regression tests; original-file reproduction unavailable.** |
| Sales-target leakage | For future sales prediction, cost/profit/margin are post-sale fields, while the feature builder could derive predictors from target-like sales values. | Target-safe feature engineering and sales-only exclusions remove known post-outcome fields without globally deleting them. Tests assert those exclusions and retain advertising spend; the available sales benchmark drops only `campaign_id`. | **Fixed and test-verified; master-file columns unavailable.** |
| Duplicate counts were confusing or static | A cleaned-only profile cannot report what was removed during cleaning; counts must use the preserved raw and cleaned frames independently. | Dataset Understanding calculates each frame's row and duplicate counts at render time. A test verifies 4 raw rows/2 duplicates → 2 cleaned rows/0 duplicates; live replacement shows its own 300/300 and 0/0 values. | **Fixed and verified; exact 367/365, 2/0 file unavailable.** |
| Continuous target caused a stratified split error in DAV pages | The split strategy must depend on target task; stratification is invalid for continuous regression targets and unsafe for rare classes. | Shared task inference and `_safe_split` stratify only suitable classification data. Real sales routes as regression; real churn routes as classification. Model Comparison and Hybrid use the same deterministic 25% holdout; regression XAI uses a regressor and task-appropriate SHAP/LIME. | **Fixed and verified on available sales/churn benchmarks.** |
| Customer risk could be misleading for sales or unknown labels | A continuous sales target is not a categorical risk label; choosing the last class as positive when labels are unfamiliar can invert risk probabilities. | Continuous targets and non-risk target names show an explicit limitation. Positive risk class is selected only from recognizable positive labels; unknown labels now stop with an explanation instead of guessing. Live churn uses `Yes`; live sales reports risk analysis as not applicable. | **Fixed and live-verified on available datasets.** |
| Sidebar logo was generic and poorly proportioned | The active logo was a generic Material `monitoring` icon, and broad sidebar paragraph CSS forced the wordmark to uppercase with excessive tracking. | The clickable Streamlit button now uses an orange bar-mark SVG via scoped CSS, a theme-aware surface, and explicit RetailIQ wordmark typography. Live light/dark screenshots show the branding; clicking from Dataset Understanding returns to Executive Overview while retaining the active dataset. | **Fixed and live-verified.** |

## Fifteen-area findings

Status definitions: **Verified** means behavior is supported by code and/or tests; **Fixed** means a defect was corrected and the correction was tested; **Partial** means the capability exists but evidence or safeguards are incomplete.

| # | Audit area | Status | Evidence and remaining limits |
|---|---|---|---|
| 1 | Backend structure and application routing | **Verified / Partial** | `app.py` provides a single Streamlit entry point and dispatches among 26 pages. Analytics logic is organized under `src/`. This audit verified the current local app, not a deployed service or production architecture. |
| 2 | Dataset ingestion and provenance | **Partial** | The uploader accepts CSV/XLS/XLSX and rejects empty/unreadable data with a visible error. `research_datasets/dataset_manifest.csv` lists 10 benchmark datasets. No `online_retail_customer_churn.csv` or `Target_Churn` field was found; therefore an expected external churn dataset cannot be claimed as present. |
| 3 | Upload identity, session integrity, and removal | **Fixed** | Upload identity now hashes the filename and file bytes, so changed contents under the same name are not mistaken for the prior upload. Clearing increments the uploader widget key and clears the active session state. Browser verification confirmed the landing screen returns with no active dataset, retained filename, or removal control. |
| 4 | Dataset profiling and understanding | **Verified / Partial** | Profiling reports row/column counts, types, missing values, duplicates, date-like fields, and target distribution when identified. Dataset Understanding reports raw and cleaned row/duplicate counts independently; regression coverage verifies 4 raw rows/2 duplicates versus 2 cleaned rows/0 duplicates. Profiling heuristics still depend on the input schema and are not a substitute for domain review. |
| 5 | Cleaning and preprocessing | **Verified / Partial** | Cleaning plans/logs and type-aware model preprocessing are implemented; supervised pipelines fit imputation, encoding, and scaling within model training. The audit did not exhaustively validate every cleaning transformation across every supported file format. |
| 6 | Problem framing and business objective | **Partial** | `build_project_definition()` documents churn, retention, KPIs, and research/business questions. This is a fixed project framing inside an otherwise general analytics app; no external stakeholder approval or course marking rubric was available. |
| 7 | Target identification and task routing | **Partial** | Strong target semantics drive inference, and the app supports explicit target selection. Churn, sales, revenue, profit, demand, prediction, and related names are recognized by the project’s target heuristics. DAV comparison, hybrid prediction, and XAI support classification and regression; arbitrary target names still require confirmation. |
| 8 | Feature engineering and leakage control | **Fixed / Partial** | Feature creation removes an explicit or strongly inferred target before deriving predictors, then restores its original position. Sales prediction additionally excludes known post-sale fields such as cost, profit, and profit margin, while retaining legitimate continuous numeric features. Tests cover the synthetic leakage case; the available retail sales benchmark confirms advertising spend is retained. Remaining engineered features are heuristic proxies and need business validation before interpretation. |
| 9 | EDA and statistical analysis | **Partial** | The app exposes EDA, visualization, and hypothesis-testing workflows. All navigation routes rendered during the smoke test, but every statistical test and edge case was not independently benchmarked in this audit. |
| 10 | Supervised baselines and evaluation | **Verified / Partial** | `run_supervised()` evaluates baseline classifiers/regressors with preprocessing in a training pipeline and reports task-appropriate metrics. DAV model comparison and hybrid paths return classification metrics or regression MAE/RMSE/R², and use safe holdout splitting for rare classes. Both checked-in retail benchmarks completed all three workflows. The app’s default holdout results are not a substitute for external validation or temporal evaluation. |
| 11 | Advanced predictive model preprocessing | **Fixed** | Advanced gradient boosting now uses a pipeline with median imputation fitted only on training rows. A regression test spies on imputer fitting and confirms 64 training rows—not all 80 rows—are used for fitting. |
| 12 | Clustering and anomaly detection | **Fixed / Partial** | Unsupervised workflows now exclude detected targets, common reference-label fields (including `is_anomaly`), and identifier-like columns. Anomaly contamination is validated. Regression tests confirm these exclusions. The workflow remains numeric-only, and a label with an arbitrary name may not be recognized automatically. |
| 13 | Hybrid model and explainability | **Verified / Partial** | Hybrid prediction and SHAP/LIME explanation paths support regression and classification, including encoded categorical features; regression SHAP uses permutation explanations. Classification uses a safe voting fallback when a rare class cannot support stacking folds. Explanations describe model behavior; they do not establish causal drivers or fairness. |
| 14 | Recommendations and Responsible AI | **Partial** | Risk thresholds produce retention-oriented recommendations, and the UI/docs describe human oversight, privacy, fairness, and limitations. These are guidance statements, not implemented subgroup fairness measurements, privacy/access controls, causal uplift analysis, or operational monitoring. |
| 15 | Reproducibility, documentation, and compliance evidence | **Partial** | The project documents seed 42, benchmark design, and a 5-repeats × 5-fold research protocol; the manifest lists 10 datasets. Existing research summaries were inspected but **not rerun or regenerated** to preserve user results. The old DAV summary’s test count and unconditional conclusion were corrected. No official external DAV rubric/evaluation brief was found, so formal compliance remains unverified. |

## Changes made

- **Upload/session lifecycle — `app.py`:** content-based upload fingerprinting; versioned uploader key on clear; immediate rerun after successful ingestion so dataset-dependent controls appear without another click.
- **Target-safe features and metadata — `src/dav_support.py`:** strong target inference expanded to sales/revenue/profit/demand/prediction/forecast names; feature engineering excludes the selected/inferred target while deriving features; preprocessing predictor lists omit it.
- **Train-only imputation — `src/advanced_ml.py`:** moved median imputation into the fitted advanced-model pipeline, after the train/test split.
- **Unsupervised leakage safeguards — `src/ml_analyzer.py`:** detected targets, common reference-label names, and identifier-like columns are filtered from clustering/anomaly inputs; invalid contamination values are rejected.
- **Task-aware prediction — `src/dav_support.py`, `src/ml_analyzer.py`, `src/advanced_ml.py`:** classification/regression metrics, nullable-value normalization, safe rare-class splitting, mixed numeric/categorical preprocessing, sales-specific post-outcome leakage filtering, logged/sanitized model failures, and regression explainability.
- **Dataset reporting and interface — `app.py`:** separate raw/clean duplicate counts, regression histograms versus categorical class counts, stronger shared chart styling, and a theme-aware clickable RetailIQ bar-mark button that preserves the active dataset.
- **Regression coverage — `tests/test_dav_pipeline.py`:** 16 tests cover target-safe feature generation, train-only imputation for numeric and categorical fields, unsupervised label/identifier exclusion, nullable inputs, rare-class splitting, sales leakage, regression XAI, risk-class selection, target distributions, and raw/clean duplicate counts.
- **Documentation:** this report records the full audit; `docs/DAV_COMPLIANCE_REPORT.md` now describes the evidence as a self-assessment rather than asserting external compliance.

## Validation evidence

| Check | Result |
|---|---|
| Full test suite (`.venv\Scripts\python.exe -m pytest -q`) | **16 passed**, 0 failed; 5 warnings |
| Python compilation (`py_compile` on changed Python files) | **Passed** |
| Streamlit health endpoint | **HTTP 200** |
| Navigation destinations | **26/26 present**; live navigation exercised on all changed DAV pages |
| Browser replacement upload | Switched between `retail_customer_churn_classification.csv` and `retail_sales_regression.csv`; each showed 300 raw → 300 cleaned rows |
| Browser removal state | Returned to landing screen; active dataset, removal control, and selected filename absent |
| Browser logo navigation | Returned from Dataset Understanding to Executive Overview while preserving the active churn upload |
| Target distribution | Regression histogram and classification count bar visually checked in dark and light themes; axes show target value/class and row frequency |
| DAV browser pages | Prediction, comparison, hybrid, XAI, and customer-risk pages rendered on the available churn dataset without a traceback; sales risk correctly reported not applicable |
| Retail benchmark execution | Both available benchmarks completed supervised, advanced, comparison, hybrid, SHAP, and LIME checks; sales retained legitimate spend predictors and dropped its campaign ID |
| Duplicate and missing-value checks | Available datasets: dynamic raw/clean 300/300 and duplicate 0/0; synthetic test: raw duplicate 2 → clean duplicate 0; nullable numeric/string features pass model tests |
| Requested external CSVs | Master, bar-chart, and churn-demo files were not present; their exact behavior could not be tested |
| Research result regeneration | **Not run**; existing dataset/result files were preserved |

Warnings were non-failing: pandas date-format inference and SHAP colormap deprecations.

## Priority follow-up

1. Obtain the official DAV rubric and map its exact criteria to project evidence before making a formal compliance claim.
2. Obtain `RetailIQ_Master_Test_Dataset(1).csv`, `RetailIQ_Bar_Chart_Test_Dataset.csv`, and `RetailIQ_Customer_Churn_Demo.csv` if exact validation is required; none is in the workspace.
3. Confirm the required churn dataset and target schema with the project owner/instructor; the checked-in manifest does not contain the named `Target_Churn` dataset.
4. Validate engineered retail features with domain owners and evaluate on representative, externally sourced data using temporal or external holdouts.
5. Add measured subgroup fairness, probability calibration, and operational monitoring before using customer-risk recommendations for live decisions.
6. Exercise XLS/XLSX upload/error cases and additional statistical/EDA edge conditions.

## Scope and limitations

This audit covers the nested `RetailIQ_v08_work` project, not the parent repository’s pre-existing deletions. It is not a formal external DAV certification, a production deployment test, or a full security assessment. Existing research artifacts were read only and left unchanged.
