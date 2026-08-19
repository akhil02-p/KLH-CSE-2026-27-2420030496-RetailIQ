# Data-Driven Customer Behaviour Analytics System

## Team Members

| S. No. | University ID | Name                    |
| ------ | ------------- | ----------------------- |
| 1      | 2420030246    | V. Maneesh Kumar Reddy  |
| 2      | 2420030202    | K. Kumar Satya Sai      |
| 3      | 2420030234    | Y. Guna Veera Sri Amith |
| 4      | 2420030496    | P. Akhil Kumar Reddy    |

## Supervisor

**Dr. K Swapnika**

## Academic Year

**2026–2027**

## Abstract

The Data-Driven Customer Behaviour Analytics System is developed to analyze retail customer transaction data and generate meaningful business insights.

The system performs data preprocessing, exploratory data analysis, customer segmentation, anomaly detection, association rule mining, and business intelligence analysis. Retail datasets containing customer transactions, product information, sales records, invoice details, and purchasing behaviour are used for analysis.

Machine learning techniques such as **K-Means Clustering, DBSCAN, and Isolation Forest**, along with **Apriori Association Rule Mining**, are used to identify customer segments, detect unusual purchasing behaviour, and discover product purchasing patterns.

The results are presented through an interactive **Power BI dashboard** containing customer segmentation, sales performance, product analysis, regional analysis, and key business performance indicators. The system also aims to generate actionable recommendations for customer retention, marketing strategies, inventory planning, and overall business performance.

## Project Objectives

* Analyze retail customer transaction data.
* Perform data cleaning and preprocessing.
* Perform Exploratory Data Analysis (EDA).
* Identify customer purchasing behaviour.
* Perform customer segmentation using machine learning.
* Detect abnormal purchasing behaviour.
* Identify product purchasing patterns using association rule mining.
* Analyze sales, products, customers, and regional performance.
* Develop an interactive Power BI dashboard.
* Generate useful business insights and recommendations.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Apriori
* Power BI
* Git
* GitHub

## Dataset

The project uses retail datasets obtained from benchmark sources such as:

* UCI Machine Learning Repository
* Kaggle

The datasets contain information related to customer transactions, products, sales, invoices, and purchasing behaviour.

If a dataset cannot be legally uploaded to GitHub, the dataset source and instructions for obtaining it will be documented in the `/data` folder.

## Repository Structure

```text
/
├── src/
├── docs/
├── data/
├── results/
├── reports/
└── README.md
```

### `src/`

Contains the source code for data preprocessing, analysis, machine learning, and other project components.

### `docs/`

Contains project-related documentation and supporting documents.

### `data/`

Contains project datasets where permitted. If the dataset cannot be uploaded, this folder contains documentation describing the original data source.

### `results/`

Contains analysis results, visualizations, model outputs, and other generated results.

### `reports/`

Contains project review and final reports.

### `README.md`

Contains project information, team details, setup instructions, execution instructions, and project status.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/maneeshreddy79/KLH-CSE-2026-27-13-RetailIQ.git
cd KLH-CSE-2026-27-13-RetailIQ
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, the required Python libraries can be installed using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

## Execution Instructions

1. Clone the repository.
2. Install the required dependencies.
3. Place the permitted dataset in the `/data` folder or follow the dataset source instructions provided there.
4. Run the data preprocessing code.
5. Perform Exploratory Data Analysis.
6. Run the customer segmentation models.
7. Run anomaly detection.
8. Run association rule mining.
9. Generate and save the analysis results in `/results`.
10. Use the generated results for the Power BI dashboard.

Execution commands will be updated as the project development progresses.

## Current Phase Status

**Current Phase:** Project Development

**Status:** In Progress

| Project Phase                        | Status      |
| ------------------------------------ | ----------- |
| Project Abstract                     | Completed   |
| Data Collection                      | In Progress |
| Data Preprocessing                   | In Progress |
| Exploratory Data Analysis            | In Progress |
| Customer Segmentation                | Planned     |
| Anomaly Detection                    | Planned     |
| Association Rule Mining              | Planned     |
| Power BI Dashboard                   | Planned     |
| Automated Customer Behaviour Verdict | Planned     |
| Final Integration                    | Planned     |

> This section will be updated whenever the project moves to a new phase.

## GitHub Contribution Guidelines

All team members will contribute to the repository using their **own GitHub accounts** so that individual contributions can be verified.

The team will:

* Maintain progressive commits throughout the project.
* Make at least one meaningful commit per week.
* Use meaningful commit messages.
* Avoid bulk uploading the complete project through a single team member's account.
* Tag major phase deliverables appropriately.

### Phase Tags

The major project deliverables will be tagged as:

```text
review-1
review-2
final
```

## Security and Data Privacy

The following must not be uploaded to this repository:

* API keys
* Passwords
* Access tokens
* Credentials
* Private keys
* Confidential institutional data
* Unauthorized licensed datasets

Sensitive files and credentials will be excluded using `.gitignore`.

## Repository Access

The repository will be made accessible to:

* **Project Supervisor:** Dr. K Swapnika

The repository will remain accessible until the final project evaluation is completed.

## Repository Information

**Repository Name:**

```text
KLH-CSE-2026-27-13-RetailIQ
```

**Repository URL:**

```text
https://github.com/maneeshreddy79/KLH-CSE-2026-27-2420030246-RetailIQ.git
```

Once the repository URL is recorded/submitted, the repository will not be renamed or transferred without written approval from the Course Coordinator.

## Project Compliance Checklist

* [ ] Repository follows the required naming convention.
* [ ] Mandatory folder structure is maintained.
* [ ] README contains all required project and team information.
* [ ] Every team member contributes using their own GitHub account.
* [ ] Progressive commits are maintained throughout the project.
* [ ] At least one meaningful commit is made per week.
* [ ] Phase deliverables are properly tagged.
* [ ] Supervisor has repository access.
* [ ] Course Coordinator has repository access.
* [ ] No credentials or API keys are uploaded.
* [ ] No confidential institutional data is uploaded.
* [ ] No unauthorized licensed datasets are uploaded.
* [ ] Repository will not be renamed or transferred after submission without written approval.
