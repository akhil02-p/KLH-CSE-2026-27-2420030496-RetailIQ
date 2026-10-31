import pandas as pd
import plotly.express as px


def _safe_cols(df, cols):
    return [c for c in cols if c is not None and c in df.columns]


def build_charts(df, profile):
    """Build a small set of safe generic Plotly charts.

    V0.5's main dashboard is rendered in app.py, but this helper is retained
    for backwards compatibility with V0.4 code.
    """
    charts = {}
    numeric = [c for c in profile.get("numeric_columns", []) if c in df.columns]
    categorical = [c for c in profile.get("categorical_columns", []) if c in df.columns]
    dates = [c for c in profile.get("date_columns", []) if c in df.columns]

    if dates and numeric:
        d, v = dates[0], numeric[0]
        temp = df[_safe_cols(df, [d, v])].dropna().copy()
        temp[d] = pd.to_datetime(temp[d], errors="coerce")
        temp = temp.dropna(subset=[d]).groupby(d, as_index=False)[v].sum()
        if not temp.empty:
            charts["📈 Trend Over Time"] = px.line(temp, x=d, y=v, markers=True, title=f"{v} Over Time")

    if categorical and numeric:
        c, v = categorical[0], numeric[0]
        temp = df[_safe_cols(df, [c, v])].dropna().groupby(c, as_index=False)[v].sum().sort_values(v, ascending=False).head(15)
        if not temp.empty:
            charts["📊 Measure by Category"] = px.bar(temp, x=c, y=v, title=f"{v} by {c}", text_auto=".2s")

    if len(numeric) >= 2:
        x, y = numeric[:2]
        temp = df[_safe_cols(df, [x, y])].dropna()
        if not temp.empty:
            charts["🔵 Numeric Relationship"] = px.scatter(temp, x=x, y=y, title=f"{x} vs {y}")

    if numeric:
        v = numeric[0]
        temp = df.dropna(subset=[v])
        if not temp.empty:
            charts["📦 Distribution"] = px.histogram(temp, x=v, nbins=30, title=f"Distribution of {v}")
    return charts
