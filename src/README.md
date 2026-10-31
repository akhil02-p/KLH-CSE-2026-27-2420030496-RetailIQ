# RetailIQ

### Adaptive Data Analytics & Business Insight System

RetailIQ is an interactive data analytics application that transforms an unfamiliar CSV or Excel dataset into a structured analytical workspace.

Instead of manually inspecting, cleaning, profiling, visualizing, and deciding what to analyze, RetailIQ automates the initial stages of the data analyst workflow and provides data-driven recommendations for the next analytical steps.

This version also includes a customer churn and retention decision-support workflow aligned with common DAV project practices. It preserves the original adaptive analytics experience while adding project definition, dataset understanding, preprocessing transparency, feature engineering, task-aware model comparison, explainable AI, customer-risk recommendations, and responsible AI framing. This is a project self-assessment, not a claim of formal compliance; no official course or institutional rubric was available for verification.

> **Upload → Profile → Clean → Analyze → Visualize → Recommend**

## DAV Workflow Coverage (Self-Assessment)

- Project definition and business objective framing for churn reduction
- Dataset understanding and target detection for customer churn
- Transparent preprocessing and feature engineering decisions
- Classification and regression baselines with task-appropriate holdout metrics
- Hybrid model comparison, with SHAP and LIME explainability for supported tasks
- Data story, methodology, and Responsible AI sections
- Decision-support recommendations for retention prioritization

The predictive workflows normalize nullable values, preserve numeric and categorical predictors, avoid known post-sale fields when predicting `sales`, and handle classification targets that are too rare for stratified holdout splitting. Dataset Understanding reports raw and cleaned row/duplicate counts separately; categorical targets use class counts, while continuous targets use a binned histogram with explicit value and row axes. Charts use a higher-contrast palette in light and dark themes, and the clickable sidebar brand mark returns to Executive Overview without clearing the active dataset. Exact behavior with `RetailIQ_Master_Test_Dataset(1).csv`, `RetailIQ_Bar_Chart_Test_Dataset.csv`, and `RetailIQ_Customer_Churn_Demo.csv` has not been validated because those files were not present in the project.

---

## 📌 Problem Statement

When a data analyst receives a new dataset, the first challenge is understanding the data before performing deeper analysis.

A typical initial workflow involves:

- Understanding the dataset structure
- Identifying data types and column roles
- Detecting missing values and duplicates
- Cleaning inconsistent data
- Identifying useful dimensions and measures
- Creating relevant KPIs
- Exploring trends and category-level patterns
- Studying relationships between variables
- Deciding which analyses or dashboards would be useful

Performing these steps manually for every new dataset can be repetitive and time-consuming.

---

## 💡 Proposed Solution

RetailIQ provides an adaptive analytics workspace that automatically examines an uploaded dataset and determines how it can be analyzed.

The application:

1. Accepts a CSV or Excel dataset
2. Profiles the dataset automatically
3. Performs conservative data cleaning and preprocessing
4. Detects numeric, categorical, and date-like fields
5. Generates relevant KPIs
6. Creates interactive visualizations
7. Provides data-quality insights
8. Recommends suitable analytical next steps
9. Allows users to filter and export the processed data

The system is designed to work with **unfamiliar datasets rather than a single fixed schema**.

## 🚀 Local and Docker Quick Start

### Local Python environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal, typically http://localhost:8501.

### Docker

```bash
docker build -t retailiq .
docker run --rm -p 8501:8501 retailiq
```

This starts the RetailIQ dashboard on port 8501 and keeps the app configuration aligned with the project’s Streamlit entrypoint.

---

## ✨ Key Features

### 📂 Multi-Format Dataset Upload

Supports common tabular formats:

- CSV
- XLSX
- XLS

Users can upload their own dataset and begin analysis without manually configuring the application for a specific schema.

---

### 🔍 Automatic Dataset Profiling

RetailIQ automatically examines the uploaded dataset and identifies:

- Number of rows
- Number of columns
- Numeric columns
- Categorical columns
- Date-like columns
- Missing values
- Unique values
- Potential column roles
- Dataset-level quality information

---

### 🧹 Automated Data Cleaning

The preprocessing pipeline performs conservative cleaning operations such as:

- Standardizing column names
- Removing completely empty columns
- Removing duplicate rows
- Trimming unnecessary whitespace
- Detecting date-like columns
- Converting suitable fields to datetime format

Cleaning actions are tracked so users can understand the transformations performed on the dataset.

---

### 📊 Adaptive KPI Generation

RetailIQ dynamically identifies suitable numeric measures and generates relevant KPIs.

Examples include:

- Row count
- Total values
- Average values
- Number of dimensions
- Data-quality percentage

The KPI layer adapts to the structure of the uploaded dataset instead of relying entirely on hard-coded business metrics.

---

### 📈 Interactive Analytics Dashboard

The application provides interactive analytical views including:

- KPI cards
- Dataset filters
- Trend analysis
- Category breakdowns
- Contribution/share analysis
- Numeric relationships
- Data distributions

Visualizations adapt according to the available columns in the uploaded dataset.

---

### 🎯 Business Recommendations

RetailIQ analyzes the characteristics of the dataset and recommends appropriate next analytical steps.

Examples include:

- Exploratory Data Analysis
- Dashboard development
- Time-series analysis
- Data segmentation
- Data-quality improvements

The recommendations explain **why** a particular analytical approach may be useful.

---

### 🛡️ Data Quality Analysis

RetailIQ provides visibility into potential data-quality problems including:

- Missing values
- Duplicate records
- Empty columns
- Constant columns
- Column-level issues
- Cleaning actions

This helps users understand the reliability of the dataset before making analytical conclusions.

---

### 📥 Filtered Data Export

Users can apply filters to the dataset and export the resulting data as a CSV file for further analysis.

---

## 🖥️ Application Workflow

```text
┌─────────────────────┐
│   Upload Dataset    │
│    CSV / Excel      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Dataset Profiling   │
│ Structure & Quality │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Data Cleaning &     │
│ Preprocessing       │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Analytical Structure│
│ Detection            │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Adaptive KPIs &     │
│ Visualizations      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Data Quality &      │
│ Relationship Analysis│
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Business             │
│ Recommendations      │
└──────────┬──────────┘
           ↓
┌─────────────────────┐
│ Filter & Export Data │
└─────────────────────┘
```

---

## 🖥️ Dashboard Sections

| Section | Purpose |
|---|---|
| **Overview** | KPIs, dataset summary and trend analysis |
| **Breakdown** | Category-level comparisons and contribution analysis |
| **Relationships** | Relationships and distributions between numeric variables |
| **Data Quality** | Missing values, duplicates and preprocessing information |
| **Data** | Filtered dataset preview and CSV export |

---

## 🏗️ Project Architecture

```text
RetailIQ/
│
├── app.py
│
├── src/
│   ├── analyzer.py
│   ├── cleaner.py
│   └── recommender.py
│
├── data/
│
├── docs/
│
├── results/
│
├── reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Core Components

#### `app.py`

Main Streamlit application responsible for:

- Dataset upload
- Application layout
- Dashboard navigation
- Filtering
- KPI generation
- Visualization rendering
- Data export

#### `src/analyzer.py`

Responsible for dataset profiling and analytical structure detection.

It identifies:

- Numeric fields
- Categorical fields
- Date-like fields
- Missing values
- Duplicate rows
- Potential column roles
- Data-quality information

#### `src/cleaner.py`

Responsible for dataset preprocessing and cleaning operations.

#### `src/recommender.py`

Responsible for generating analytical recommendations based on the characteristics of the uploaded dataset.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data manipulation and preprocessing |
| **NumPy** | Numerical operations |
| **Streamlit** | Interactive web application |
| **Plotly** | Interactive data visualization |
| **OpenPyXL** | XLSX file processing |
| **XLRD** | XLS file processing |

---

## 📊 Analytical Capabilities

RetailIQ currently focuses on the following data analytics tasks:

### Data Preparation

- Data ingestion
- Data profiling
- Data cleaning
- Duplicate detection
- Missing-value analysis
- Data-type detection
- Date parsing

### Exploratory Data Analysis

- Descriptive statistics
- Numeric distributions
- Category-level analysis
- Trend analysis
- Contribution analysis
- Relationship analysis

### Business Intelligence

- Adaptive KPIs
- Interactive filtering
- Dashboard-style reporting
- Business recommendations
- Filtered data export

---

## 🧠 Recommendation Engine

RetailIQ currently uses a **rule-based recommendation approach**.

The recommendation system examines the structure and characteristics of the uploaded dataset and determines which analytical actions are appropriate.

For example:

```text
Dataset contains date + numeric measure
                ↓
        Time-series analysis
        may be recommended
```

or:

```text
Dataset contains categorical
and numeric variables
                ↓
       Segmentation /
       breakdown analysis
       may be recommended
```

This approach allows the application to provide recommendations without requiring a machine-learning model.

---

## 🤖 Does RetailIQ Use Machine Learning?

### Current Version: No

The current V0.7 implementation does **not use machine-learning models for prediction**.

Instead, it focuses on:

- Automated data profiling
- Data preprocessing
- Exploratory analytics
- Visualization
- Rule-based recommendations

Machine-learning based capabilities are part of the future roadmap rather than the current implementation.

---

## 🚀 Getting Started

### Prerequisites

Make sure you have:

- Python 3.10+
- pip
- Git

---

### 1. Clone the Repository

```bash
git clone https://github.com/maneeshreddy79/RetailIQ.git
cd RetailIQ
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Supported Input Formats

RetailIQ currently supports:

```text
CSV
XLSX
XLS
```

The application is designed to work with datasets having different structures and column names.

---

## 📌 Example Use Cases

RetailIQ can be used as an initial analytical workspace for datasets such as:

- Retail sales
- Customer transactions
- E-commerce orders
- Marketing campaigns
- Financial records
- Business operations
- Product performance
- Customer analytics

The application does not require the uploaded dataset to follow one predefined business schema.

---

## 🔬 Current Project Scope

The current version focuses on the **initial data analyst workflow**:

```text
Raw Dataset
     ↓
Understand
     ↓
Clean
     ↓
Explore
     ↓
Visualize
     ↓
Identify Opportunities
```

RetailIQ is currently an analytics and decision-support application rather than a predictive machine-learning platform.

---

## ⚠️ Current Limitations

The current version has several limitations:

- Recommendations are rule-based
- No predictive ML models
- No direct SQL/database connectivity
- No automated statistical hypothesis testing
- No advanced anomaly detection
- No production-grade multi-user authentication
- No automated natural-language querying of datasets

These limitations define areas for future development.

---

## 🔮 Future Roadmap

### Analytics

- [ ] Advanced outlier detection
- [ ] Automated anomaly detection
- [ ] Correlation analysis
- [ ] Statistical hypothesis testing
- [ ] Advanced segmentation
- [ ] Automated statistical insights

### Business Intelligence

- [ ] Automated report generation
- [ ] More advanced dashboard recommendations
- [ ] Executive summary generation
- [ ] Custom KPI configuration

### Data Sources

- [ ] SQL database connectivity
- [ ] Cloud data-source integration
- [ ] API-based data ingestion

### Machine Learning

- [ ] Predictive analytics
- [ ] Automated model selection
- [ ] ML-based recommendations
- [ ] Forecasting
- [ ] Anomaly detection models

### AI-Assisted Analytics

- [ ] Natural-language dataset queries
- [ ] Automated insight generation
- [ ] AI-generated business summaries
- [ ] Conversational analytics

---

## 📈 Development Status

**Current Version:** V0.7

**Status:** Active Development

RetailIQ is being developed as an adaptive analytics platform focused on automating the initial stages of the data analyst workflow.

The current version establishes the core data ingestion, profiling, preprocessing, visualization and recommendation pipeline.

---

## 🎓 Project Objective

The long-term objective of RetailIQ is to reduce the repetitive work involved in the initial stages of data analysis.

The vision is to build an adaptive analytics assistant that can:

```text
Understand unfamiliar data
          ↓
Prepare the data
          ↓
Discover analytical patterns
          ↓
Recommend useful analyses
          ↓
Generate insights
          ↓
Support business decisions
```

---

## 👨‍💻 Author

### Veluru Maneesh Kumar Reddy

**Data Analyst | Business Intelligence**

B.Tech Computer Science & Engineering (AI & ML)

Interested in:

- Data Analytics
- Business Intelligence
- Business Analytics
- Product Analytics

### Links

**GitHub:**  
https://github.com/maneeshreddy79

**LinkedIn:**  
https://www.linkedin.com/in/maneesh-kumar-reddy-2b66bb325/

---

## 📄 License

This project is intended for educational, portfolio and demonstration purposes.

---

⭐ If you find RetailIQ interesting, consider exploring the repository and the project evolution.
## ML Analysis (new)

RetailIQ now includes an adaptive ML analysis tab with:

- automatic target/task suggestions;
- classification baselines: Logistic Regression, Decision Tree and Random Forest;
- regression baselines: Linear Regression, Decision Tree and Random Forest;
- hold-out evaluation with task-appropriate metrics;
- feature-importance/coefficient views where supported;
- K-Means clustering with Silhouette Score;
- Isolation Forest anomaly detection.

See `docs/ml_methodology.md` for the research methodology and evaluation design.


## V0.8 Research Evaluation
V0.8 adds a reproducible research benchmark with automatic-routing evaluation, leakage-safe 5/10-fold cross-validation, K-Means K-range evaluation, runtime logging, and statistical comparison artifacts. Results are saved under `results/`.
