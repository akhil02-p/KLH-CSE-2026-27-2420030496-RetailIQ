def generate_recommendations(profile, df):
    recs = []
    if profile["missing_cells"] or profile["duplicate_rows"]:
        recs.append({
            "priority":"⚠️","title":"Resolve Data-Quality Issues",
            "description":"Review missing values, duplicates and inconsistent fields before relying on downstream KPIs.",
            "why":"Poor-quality inputs can distort analysis.",
            "business_value":"Improves reliability and reproducibility.",
            "tools":"Python, Pandas, SQL",
            "question":"Which fields require cleaning or validation before reporting?"
        })
    if profile["numeric_columns"]:
        recs.append({
            "priority":"1️⃣","title":"Perform Exploratory Data Analysis",
            "description":"Inspect distributions, summary statistics, outliers and relationships among measurable fields.",
            "why":"EDA is a safe first analytical step for an unfamiliar dataset.",
            "business_value":"Reveals patterns and anomalies.",
            "tools":"Python, Pandas, NumPy, Plotly",
            "question":"What are the main distributions, trends and unusual observations?"
        })
    if profile["categorical_columns"] and profile["numeric_columns"]:
        recs.append({
            "priority":"2️⃣","title":"Build an Interactive Dashboard",
            "description":"Compare key measures across categorical dimensions using filters and visual summaries.",
            "why":"The dataset has measures and dimensions suitable for reporting.",
            "business_value":"Supports recurring monitoring and decision-making.",
            "tools":"Power BI, Python, Plotly",
            "question":"Which segments or categories contribute most to the key measure?"
        })
    if profile["date_columns"] and profile["numeric_columns"]:
        recs.append({
            "priority":"3️⃣","title":"Perform Time-Series Analysis",
            "description":"Aggregate measurable fields over time to identify trends, peaks, dips and seasonality.",
            "why":"A date-like field and numeric measure are available.",
            "business_value":"Helps identify growth patterns and periods requiring attention.",
            "tools":"Python, Pandas, Plotly, Power BI",
            "question":"How does the key metric change over time?"
        })
    if profile["categorical_columns"]:
        recs.append({
            "priority":"4️⃣","title":"Segment the Data",
            "description":"Analyze performance by important categorical fields such as region, product or customer group.",
            "why":"Categorical dimensions allow meaningful group comparisons.",
            "business_value":"Supports targeted actions.",
            "tools":"SQL, Pandas, Power BI",
            "question":"Which groups perform best and which require attention?"
        })
    if not recs:
        recs.append({
            "priority":"ℹ️","title":"Inspect Dataset Structure",
            "description":"The dataset contains limited analytical signals. Review its business meaning before advanced analytics.",
            "why":"Recommendations should not overstate what the data can support.",
            "business_value":"Prevents misleading analysis.",
            "tools":"Python, Pandas",
            "question":"What business decision is this dataset intended to support?"
        })
    return recs
