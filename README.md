# RetailIQ – Explainable Machine Learning-Based Retail Analytics Platform

**Project motto:** *Built for smarter, data-driven decisions.*

RetailIQ is an interactive retail analytics platform designed to help users move from raw retail data to meaningful, explainable business insights. It brings dataset understanding, data preparation, exploratory analysis, machine learning, model evaluation, explainability, and decision-support outputs together in one Streamlit application.

Rather than presenting predictions as unexplained numbers, RetailIQ aims to help users understand the data, evaluate analytical models, and interpret the factors influencing their results.

---

## Project Team

| S. No. | University ID | Name |
|---|---|---|
| 1 | 2420030246 | V. Maneesh Kumar Reddy |
| 2 | 2420030202 | K. Kumar Satya Sai |
| 3 | 2420030234 | Y. Guna Veera Sri Amith |
| 4 | 2420030496 | P. Akhil Kumar Reddy |

## Project Supervisor

**Dr. K Swapnika**

## Academic Year

**2026–2027**

---

## 1. Problem Statement

Retail businesses generate data about transactions, sales, products, customers, and purchasing behaviour. However, raw datasets are not immediately useful for decision-making. They may contain missing values, duplicate records, inconsistent data types, and patterns that are difficult to identify by manually inspecting tables.

Businesses also need more than historical reports. Depending on the available data, they may need to understand customer behaviour, identify unusual patterns, compare predictive models, estimate future outcomes, and understand why a model produced a particular prediction.

RetailIQ addresses this challenge through a unified, interactive workflow for retail data preparation, analysis, machine learning, and explainability.

## 2. Project Objectives

- Allow users to upload and work with retail datasets.
- Understand dataset structure, column types, missing values, and duplicate records.
- Support data cleaning and preparation.
- Perform exploratory data analysis and statistical analysis.
- Explore feature relationships and support feature engineering.
- Apply suitable machine learning methods based on the dataset and analytical task.
- Compare model performance using appropriate evaluation metrics.
- Support explainability using SHAP and LIME where compatible.
- Present visual analytics and business-oriented insights.
- Support customer-risk analysis and recommendations when the dataset contains the required information.
- Provide a single interface for the analytical workflow.

## 3. How RetailIQ Works

1. **Upload data** – The user supplies a supported dataset, such as a CSV file.
2. **Understand the dataset** – The platform inspects its dimensions, columns, data types, missing values, and other quality indicators.
3. **Prepare the data** – Data-cleaning and feature-preparation operations make the dataset suitable for analysis.
4. **Explore and analyse** – Statistical summaries, visualizations, distributions, and relationships help users investigate the data.
5. **Apply machine learning** – Suitable analytical models are selected according to the target and task, such as regression or classification, where supported.
6. **Evaluate and explain** – Model results are evaluated, compared, and interpreted using explainability methods where applicable.
7. **Use the insights** – Visual analytics and recommendations help translate findings into possible business actions.

The exact outputs depend on the dataset, its columns, data quality, target variable, and the suitability of the selected analysis.

---

## 4. Main Application Areas

RetailIQ contains a set of connected analytical pages. The available pages include:

| Application area | Purpose |
|---|---|
| Executive Overview | Presents the project context and high-level dataset or business information. |
| Dataset Understanding | Summarizes dataset dimensions, column details, distributions, and quality indicators. |
| Data Cleaning | Supports inspection and preparation of incomplete or duplicated data. |
| EDA & Statistics | Explores distributions, descriptive statistics, and relationships in the data. |
| Feature Engineering | Supports preparation and transformation of features for analysis and modelling. |
| Predictive Analytics | Runs suitable predictive tasks, including regression or classification where supported. |
| Model Comparison | Compares candidate models using task-appropriate evaluation metrics. |
| Hybrid Model | Provides the project's proposed hybrid modelling workflow. |
| Explainable AI | Uses SHAP and LIME, where compatible, to help interpret model behaviour and predictions. |
| Visual Analytics | Presents analytical results through visualizations. |
| Customer Risk Analysis | Supports risk-focused analysis when the uploaded dataset contains suitable information. |
| Recommendations | Presents data-driven recommendations where supported by the available analysis. |
| Data Story | Helps communicate analytical findings in a business-oriented format. |
| Research Evaluation | Provides access to saved research and evaluation artifacts. |

The application may include additional workflow or compliance pages. Existing navigation destinations should be retained when the project is updated.

## 5. Technologies Used

| Technology | Role in the project |
|---|---|
| Python | Main programming language and application logic. |
| Streamlit | Interactive web application and dashboard interface. |
| Pandas | Dataset loading, manipulation, and analysis. |
| NumPy | Numerical operations. |
| Matplotlib | Data visualization. |
| Seaborn | Statistical visualization. |
| Scikit-learn | Machine learning, preprocessing, model selection, and evaluation. |
| SHAP | Model explanation, where supported. |
| LIME | Local model explanation, where supported. |
| Git | Version control. |
| GitHub | Source-code hosting and team collaboration. |
| Custom CSS | Dashboard styling and visual presentation. |

> The application uses the existing Python and Streamlit architecture. It is not a React-based frontend.

---

## 6. Repository Structure

The project is organized around the Streamlit entry point and supporting Python modules. The main folders include:

```text
RetailIQ/
├── app.py
├── requirements.txt
├── RUN_RETAILIQ.bat
├── src/
│   └── ... application and analytical modules
├── data/
│   └── ... permitted datasets and dataset instructions
├── docs/
│   └── ... project and technical documentation
├── reports/
│   └── ... review, audit, and project reports
├── results/
│   └── ... generated analytical outputs and artifacts
├── tests/
│   └── ... automated tests
└── README.md
```

The exact contents may evolve as the project develops. Do not commit virtual environments, credentials, or restricted datasets.

---

## 7. Setup Instructions

### Prerequisites

- Python installed on the computer.
- Git, if cloning the repository.
- A terminal such as PowerShell, Command Prompt, or a VS Code terminal.

### 1. Clone the repository

Use the repository URL recorded for the project:

```bash
git clone https://github.com/maneeshreddy79/KLH-CSE-2026-27-2420030246-RetailIQ.git
cd KLH-CSE-2026-27-13-RetailIQ
```

If the repository's submitted URL or folder name differs, use the exact URL and directory shown in GitHub.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or, in Command Prompt:

```bat
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the project's `requirements.txt` as the source of truth for dependencies. If it is unavailable, check the project documentation before installing packages individually.

### 4. Run RetailIQ

From the project root:

```bash
streamlit run app.py
```

If the project-provided Windows launcher `RUN_RETAILIQ.bat` is available and configured, it can also be used to start the application.

Streamlit will print a local URL in the terminal. Open that URL in a web browser.

---

## 8. Using the Application

1. Start the Streamlit application.
2. Upload a supported dataset using the dataset uploader.
3. Confirm the active dataset shown by the application.
4. Begin with Executive Overview or Dataset Understanding.
5. Inspect data quality before relying on analytical outputs.
6. Use the relevant analysis page for the question you want to investigate.
7. Select an appropriate target and modelling task when required.
8. Review model metrics and explanations rather than relying on predictions alone.
9. Use recommendations only when they are supported by the dataset and analysis.

### Dataset guidance

- Use CSV datasets with clearly named columns and a documented meaning for important fields.
- Choose a target column that matches the question being investigated.
- For regression, the target should represent a meaningful continuous quantity.
- For classification, the target should represent meaningful classes.
- Avoid using information that would not be available at prediction time; this can cause target leakage.
- Check whether the dataset contains enough examples for the selected task.
- Do not upload confidential, personal, or unauthorized data.

Dataset-specific outputs are not guaranteed to be available for every page. A module should explain when a dataset does not contain the fields or observations required for a valid analysis.

---

## 9. Testing and Validation

The project includes automated tests and has undergone application-level checks during development. Validation should be repeated after code changes.

Recommended checks include:

- Application startup and dependency loading.
- Dataset upload, replacement, and removal.
- Correct raw and cleaned row counts.
- Missing-value and duplicate reporting.
- Regression and classification task handling.
- Appropriate train/test splitting and evaluation metrics.
- Prevention of target leakage.
- Model comparison and hybrid-model execution.
- SHAP/LIME compatibility with supported models.
- Navigation across application pages.
- Preservation of the active dataset during navigation.
- Light- and dark-theme readability.
- User-friendly error handling.

**Project status:** Development and validation are ongoing. A successful test of one dataset or module does not establish that every dataset and every analytical page is fully supported.

Formal academic/DAV compliance should only be claimed after checking the implementation against the official university rubric.

---

## 10. Current Project Status

RetailIQ is an evolving academic project. Its interface, analytical modules, tests, and documentation are updated as development progresses.

For the latest implementation status:

- Review the current source code.
- Run the automated test suite.
- Start the application and test the relevant workflows.
- Review the latest audit and project reports in `reports/` and `docs/`.

Do not treat a planned, partially implemented, or unverified capability as completed.

---

## 11. GitHub Contribution Guidelines

All team members should contribute through their own GitHub accounts so that individual contributions can be tracked.

The team should:

- Make progressive commits throughout development.
- Use meaningful commit messages.
- Keep changes focused and reviewable.
- Avoid uploading the complete project through a single account as one bulk commit.
- Tag major project deliverables appropriately.

Suggested milestone tags:

```text
review-1
review-2
final
```

---

## 12. Security and Data Privacy

Do not upload the following to the repository:

- API keys, passwords, access tokens, or credentials.
- Private keys or environment files containing secrets.
- Confidential institutional or customer information.
- Personal data without appropriate authorization.
- Licensed or third-party datasets that cannot legally be redistributed.

Use `.gitignore` to exclude virtual environments, local configuration, secrets, temporary files, and other non-project artifacts.

---

## 13. Repository Information

**Repository name:** `KLH-CSE-2026-27-13-RetailIQ`

**Repository URL recorded in the project README:**

```text
https://github.com/maneeshreddy79/KLH-CSE-2026-27-2420030246-RetailIQ.git
```

Confirm the canonical repository URL in GitHub before sharing or changing repository details. The earlier README contained two different clone/repository URLs.

The repository should remain accessible to the project supervisor and other authorized academic reviewers.

---

## 14. Project Motto

> **Built for smarter, data-driven decisions.**

RetailIQ aims to connect data preparation, analysis, machine learning, and explainability in one interactive platform—helping users turn retail data into insights they can understand and use.
