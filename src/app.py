from pathlib import Path
import hashlib
import io
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, f_oneway, ttest_ind

from src.analyzer import analyze_dataset
from src.cleaner import build_cleaning_plan, clean_dataset
from src.dav_support import (
    build_customer_recommendation,
    build_data_story,
    build_hybrid_model,
    build_target_distribution_data,
    build_preprocessing_decisions,
    build_project_definition,
    compare_models,
    engineer_features,
    fit_explanation_model,
    generate_lime_explanation,
    generate_shap_explanation,
    positive_risk_class_index,
    summarize_dataset,
)
from src.recommender import generate_recommendations
from src.ml_analyzer import (
    detect_target, infer_task, target_candidates,
    run_supervised, run_clustering, run_anomaly_detection,
)
from src.advanced_ml import (
    run_advanced_predictive, run_forecast,
    ml_guided_recommendations, generate_natural_language_insights,
)

logger = logging.getLogger(__name__)

# ============================================================
# RetailIQ — Premium Adaptive Analytics Frontend
# Redesigned UI/UX: warm cream/orange/brown/green palette.
# All analytics/ML engines in src/ are UNCHANGED.
# ============================================================

st.set_page_config(
    page_title="RetailIQ — Adaptive Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────── State Init ─────────────────────────

def init_state():
    defaults = {
        "theme": "light",
        "page_search": "Overview",
        "confirm_clear_dataset": False,
        "raw_df": None,
        "cleaned_df": None,
        "profile": None,
        "clean_profile": None,
        "cleaning_log": [],
        "uploaded_name": None,
        "uploaded_fingerprint": None,
        "upload_widget_version": 0,
        "selected_page": "Overview",
        "filters": {},
        "ml_results": {},
        "anomaly_result": None,
        "cluster_result": None,
        "forecast_result": None,
        "advanced_result": None,
        "recommendations": [],
        "project_definition": build_project_definition(),
        "dataset_summary": {},
        "comparison_result": {},
        "hybrid_model": {},
        "selected_customer_index": 0,
        "customer_explanation": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────── Color Palettes ──────────────────────

LIGHT = {
    "bg": "#f2e4d4", "surface": "#ffffff", "surface2": "#fafafa",
    "sidebar_bg": "#fafafa", "muted_bg": "#e4dacd",
    "text": "#1a1a1a", "muted": "#6b5e52", "sidebar_fg": "#3f3f46",
    "primary": "#da6f0b", "primary_light": "#f0a05a",
    "secondary": "#593c26", "accent": "#4cae4f", "accent_dark": "#3d9040",
    "border": "#dbcebd", "ring": "#da6f0b",
    "destructive": "#ef4444", "success": "#4cae4f",
    "warning": "#da6f0b", "info": "#593c26",
    "chart1": "#8b4513", "chart2": "#2d7d30", "chart3": "#593c26",
    "chart4": "#a64b00", "chart5": "#7a5230",
}

DARK = {
    "bg": "#1a1a1a", "surface": "#262626", "surface2": "#303030",
    "sidebar_bg": "#1f1f1f", "muted_bg": "#2e2e2e",
    "text": "#f2e4d4", "muted": "#c9c1b7", "sidebar_fg": "#c9c1b7",
    "primary": "#da6f0b", "primary_light": "#e8935a",
    "secondary": "#a07050", "accent": "#4cae4f", "accent_dark": "#3d9040",
    "border": "rgba(218,175,155,0.18)", "ring": "#da6f0b",
    "destructive": "#ef4444", "success": "#4cae4f",
    "warning": "#da6f0b", "info": "#a07050",
    "chart1": "#f0a05a", "chart2": "#8bd68d", "chart3": "#f2e4d4",
    "chart4": "#e8a060", "chart5": "#d6b98c",
}

C = DARK if st.session_state.theme == "dark" else LIGHT

# ───────────────────────── CSS Injection ─────────────────────

def inject_css():
    is_dark = st.session_state.theme == "dark"
    sidebar_active_bg = "rgba(218,111,11,0.14)" if not is_dark else "rgba(218,111,11,0.20)"
    sidebar_hover_bg  = "rgba(218,111,11,0.07)" if not is_dark else "rgba(218,111,11,0.12)"
    footer_bg = "rgba(218,111,11,0.06)" if not is_dark else "rgba(218,111,11,0.12)"

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
  --bg: {C['bg']}; --surface: {C['surface']}; --surface2: {C['surface2']};
  --sidebar-bg: {C['sidebar_bg']}; --muted-bg: {C['muted_bg']};
  --text: {C['text']}; --muted: {C['muted']}; --sidebar-fg: {C['sidebar_fg']};
  --primary: {C['primary']}; --secondary: {C['secondary']}; --accent: {C['accent']};
  --border: {C['border']}; --destructive: {C['destructive']};
  --shadow-sm: 0 1px 3px rgba(89,60,38,0.10), 0 1px 2px rgba(89,60,38,0.06);
  --shadow-md: 0 4px 16px rgba(89,60,38,0.12), 0 2px 8px rgba(89,60,38,0.08);
  --radius: 14px; --radius-sm: 8px;
}}

html, body, [data-testid="stAppViewContainer"] {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg); color: var(--text);
}}
.stApp {{ background: var(--bg); color: var(--text); }}
.block-container {{ max-width: 1520px; padding: 0.75rem 1.75rem 3rem; }}
[data-testid="stHeader"] {{ background: transparent; }}

[data-testid="stSidebar"] {{
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--border);
  box-shadow: 2px 0 12px rgba(89,60,38,0.06);
}}
[data-testid="stSidebar"] * {{ color: var(--sidebar-fg) !important; }}
[data-testid="stSidebar"] .stButton button {{
  border: 1px solid transparent !important; background: transparent !important;
  color: var(--sidebar-fg) !important; text-align: left !important;
  border-radius: 10px !important; min-height: 42px !important;
  font-size: 0.85rem !important; font-weight: 500 !important;
  padding: 0 12px !important; width: 100% !important;
  transition: all 0.18s cubic-bezier(0.4,0,0.2,1) !important;
}}
[data-testid="stSidebar"] .stButton button:hover {{
  background: {sidebar_hover_bg} !important; color: var(--primary) !important;
  border-color: rgba(218,111,11,0.20) !important; transform: translateX(2px) !important;
}}
[data-testid="stSidebar"] .stButton button[kind="primary"] {{
  background: {sidebar_active_bg} !important; color: var(--primary) !important;
  border-color: rgba(218,111,11,0.30) !important; font-weight: 700 !important;
}}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p {{
  color: var(--muted) !important; font-size: 0.72rem !important; font-weight: 700 !important;
  letter-spacing: 0.10em !important; text-transform: uppercase !important; padding: 12px 12px 4px !important;
}}

h1,h2,h3,h4,h5,h6 {{ color: var(--text); font-family: 'Inter', sans-serif; letter-spacing: -0.025em; font-weight: 700; }}
p, label, .stCaption {{ color: var(--muted); }}

div[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px 20px 16px; box-shadow: var(--shadow-sm);
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}}
div[data-testid="stMetric"]:hover {{ box-shadow: var(--shadow-md); transform: translateY(-1px); }}
div[data-testid="stMetricLabel"] {{ color: var(--muted) !important; font-size: 0.82rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }}
div[data-testid="stMetricValue"] {{ color: var(--text) !important; font-weight: 800 !important; font-size: 1.65rem !important; }}

.stButton button, .stDownloadButton button {{
  border-radius: 10px !important; border: 1px solid var(--border) !important;
  background: var(--surface) !important; color: var(--text) !important;
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  font-size: 0.84rem !important; padding: 0.45rem 1.1rem !important;
  transition: all 0.18s ease !important;
}}
.stButton button:hover, .stDownloadButton button:hover {{
  background: var(--muted-bg) !important; border-color: var(--primary) !important;
}}
.stButton button[kind="primary"] {{
  background: var(--primary) !important; border-color: var(--primary) !important; color: white !important;
}}
.stButton button[kind="primary"]:hover {{
  background: #c26009 !important; border-color: #c26009 !important; color: white !important;
  box-shadow: 0 4px 12px rgba(218,111,11,0.35) !important;
}}
.stButton button:focus-visible, .stDownloadButton button:focus-visible,
div[data-baseweb="select"] > div:focus-visible, input:focus-visible, textarea:focus-visible {{
  outline: 3px solid rgba(218,111,11,0.48) !important;
  outline-offset: 2px !important;
}}
[data-testid="stSidebar"] .stButton button {{
  position: relative; overflow: hidden;
}}
[data-testid="stSidebar"] .stButton button[kind="primary"]::before {{
  content: ''; position: absolute; left: 0; top: 50%; width: 3px; height: 18px;
  border-radius: 0 3px 3px 0; background: var(--primary); transform: translateY(-50%);
}}
[data-testid="stSidebar"] .stButton button:hover > div {{
  transform: translateX(1px);
}}
[data-testid="stIconMaterial"] {{
  font-family: "Material Symbols Rounded" !important;
  font-feature-settings: "liga";
}}
.st-key-retailiq_logo .stButton button {{
  display: flex !important; align-items: center !important; justify-content: flex-start !important;
  gap: 12px !important; min-height: 58px !important; padding: 0 16px !important;
  border: 1px solid rgba(218,111,11,0.42) !important; border-radius: 12px !important;
  background: linear-gradient(135deg, {C['surface']}, {C['muted_bg']}) !important;
  color: {C['text']} !important; box-shadow: var(--shadow-sm) !important;
}}
.st-key-retailiq_logo .stButton button:hover {{
  background: {C['muted_bg']} !important; border-color: {C['primary']} !important;
  transform: translateY(-1px) !important;
}}
.st-key-retailiq_logo .stButton button::before {{
  content: ""; display: block; flex: 0 0 30px; width: 30px; height: 30px;
  background: center / contain no-repeat url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect x='3' y='17' width='6' height='12' rx='2' fill='%23DA6F0B'/%3E%3Crect x='13' y='10' width='6' height='19' rx='2' fill='%23E88939'/%3E%3Crect x='23' y='3' width='6' height='26' rx='2' fill='%23F0A05A'/%3E%3C/svg%3E");
}}
.st-key-retailiq_logo .stButton button p {{
  margin: 0 !important; padding: 0 !important; color: {C['text']} !important;
  font-size: 1.16rem !important; font-weight: 800 !important;
  letter-spacing: -0.035em !important; line-height: 1.15 !important;
  text-align: left !important; text-transform: none !important;
}}

div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input, .stTextArea textarea {{
  background: var(--surface) !important; border-color: var(--border) !important;
  color: var(--text) !important; border-radius: var(--radius-sm) !important; font-family: 'Inter', sans-serif !important;
}}
[data-testid="stSelectbox"] [role="group"],
[data-testid="stMultiSelect"] [role="group"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  color: var(--text) !important; border-radius: var(--radius-sm) !important;
}}
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] {{
  color: var(--text) !important;
}}
div[data-baseweb="select"] > div:focus-within, .stTextInput input:focus {{
  border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(218,111,11,0.12) !important;
}}
[data-baseweb="popover"] > div {{
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
}}

[data-testid="stFileUploader"] {{
  background: var(--surface) !important; border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
}}
[data-testid="stFileUploaderDropzone"] {{
  background: var(--surface) !important; border-color: var(--border) !important;
  color: var(--muted) !important;
}}
[data-testid="stFileUploaderDropzone"] [data-testid="stBaseButton-secondary"] {{
  background: var(--surface) !important; border-color: var(--border) !important;
  color: var(--text) !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: var(--primary) !important; }}

.stTabs [data-baseweb="tab-list"] {{
  background: var(--muted-bg) !important; border-radius: 10px !important;
  padding: 4px !important; gap: 2px !important; border-bottom: none !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important; border-radius: 8px !important; color: var(--muted) !important;
  font-weight: 600 !important; font-size: 0.83rem !important; padding: 6px 16px !important;
  border: none !important; transition: all 0.18s ease !important;
}}
.stTabs [aria-selected="true"] {{
  background: var(--surface) !important; color: var(--primary) !important; box-shadow: var(--shadow-sm) !important;
}}

details[data-testid="stExpander"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; overflow: hidden !important;
}}
details[data-testid="stExpander"] summary {{ font-weight: 600 !important; color: var(--text) !important; padding: 12px 16px !important; }}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  overflow: hidden !important; background: var(--surface) !important;
}}

.stAlert {{ border-radius: var(--radius-sm) !important; border-left: 4px solid !important; }}
[data-testid="stSuccess"] {{ background: rgba(76,174,79,0.08) !important; border-color: var(--accent) !important; }}
[data-testid="stWarning"] {{ background: rgba(218,111,11,0.08) !important; border-color: var(--primary) !important; }}
[data-testid="stError"] {{ background: rgba(239,68,68,0.08) !important; border-color: var(--destructive) !important; }}
[data-testid="stInfo"] {{ background: rgba(89,60,38,0.08) !important; border-color: var(--secondary) !important; }}

hr {{ border-color: var(--border) !important; margin: 1rem 0 !important; }}

.page-kicker {{
  color: var(--primary); font-size: 0.72rem; font-weight: 800;
  text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 4px;
  font-family: 'Inter', sans-serif;
}}
.page-title {{
  font-size: 1.85rem; font-weight: 800; color: var(--text);
  letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 4px; font-family: 'Inter', sans-serif;
}}
.page-subtitle {{
  color: var(--muted); font-size: 0.88rem; font-weight: 400;
  margin-bottom: 1.25rem; line-height: 1.5;
}}
.kpi-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px 20px 16px; box-shadow: var(--shadow-sm); transition: all 0.2s ease;
  position: relative; overflow: hidden;
}}
.kpi-card:hover {{ box-shadow: var(--shadow-md); transform: translateY(-2px); }}
.kpi-card::before {{
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--accent-color, var(--primary));
}}
.kpi-icon {{ font-size: 1.4rem; margin-bottom: 8px; display: block; }}
.kpi-value {{ font-size: 1.65rem; font-weight: 800; color: var(--text); line-height: 1.1; font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
.kpi-label {{ font-size: 0.75rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }}
.kpi-subtext {{ font-size: 0.78rem; color: var(--muted); margin-top: 4px; }}
.dashboard-card {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; box-shadow: var(--shadow-sm); margin-bottom: 14px; transition: box-shadow 0.2s ease;
}}
.dashboard-card:hover {{ box-shadow: var(--shadow-md); }}
.card-title {{
  font-size: 0.92rem; font-weight: 700; color: var(--text); margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px; font-family: 'Inter', sans-serif;
}}
.card-title-icon {{
  width: 28px; height: 28px; border-radius: 7px; display: inline-flex;
  align-items: center; justify-content: center; font-size: 0.85rem; flex-shrink: 0;
}}
.insight {{
  border: 1px solid var(--border); background: var(--surface2); border-radius: var(--radius-sm);
  padding: 12px 15px; margin: 6px 0; font-size: 0.85rem; line-height: 1.5;
  color: var(--muted); transition: border-color 0.18s ease;
}}
.insight:hover {{ border-color: rgba(218,111,11,0.35); }}
.insight strong {{ color: var(--text); font-weight: 600; }}
.rec {{
  border: 1px solid var(--border); border-radius: var(--radius); padding: 18px;
  background: var(--surface); transition: all 0.2s ease; min-height: 140px;
  display: flex; flex-direction: column; margin-bottom: 12px;
}}
.rec:hover {{ border-color: rgba(218,111,11,0.45); box-shadow: var(--shadow-md); transform: translateY(-1px); }}
.rec-priority {{
  display: inline-block; font-size: 0.70rem; font-weight: 700; padding: 2px 8px;
  border-radius: 20px; background: rgba(218,111,11,0.12); color: var(--primary);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;
}}
.rec-title {{ font-size: 0.92rem; font-weight: 700; color: var(--text); margin-bottom: 8px; line-height: 1.35; }}
.rec-desc {{ font-size: 0.82rem; color: var(--muted); flex: 1; line-height: 1.5; }}
.rec-value {{ font-size: 0.78rem; color: var(--accent); font-weight: 600; margin-top: 10px; }}
.topbar {{
  border: 1px solid var(--border); background: var(--surface); border-radius: var(--radius);
  padding: 10px 16px; margin-bottom: 20px; box-shadow: var(--shadow-sm);
  display: flex; align-items: center; font-size: 0.85rem; color: var(--muted);
}}
.logo-wrap {{ padding: 20px 16px 16px; border-bottom: 1px solid var(--border); margin-bottom: 8px; }}
.logo-word {{ font-size: 1.3rem; font-weight: 850; color: var(--text); letter-spacing: -0.03em; font-family: 'Inter', sans-serif; }}
.logo-sub {{ font-size: 0.7rem; color: var(--muted); font-weight: 500; letter-spacing: 0.04em; }}
.logo-badge {{
  display: inline-block; background: var(--primary); color: white; font-size: 0.6rem;
  font-weight: 800; padding: 1px 6px; border-radius: 10px; letter-spacing: 0.08em;
  text-transform: uppercase; margin-left: 6px; vertical-align: middle;
}}
.sidebar-footer {{
  border: 1px solid var(--border); background: {footer_bg};
  border-radius: var(--radius-sm); padding: 14px; margin: 12px 0 8px; font-size: 0.82rem;
}}
.small {{ font-size: 0.8rem; color: var(--muted); }}
.badge {{ display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; }}
.badge-success {{ background: rgba(76,174,79,0.12); color: var(--accent); }}
.badge-warning {{ background: rgba(218,111,11,0.12); color: var(--primary); }}
.badge-info {{ background: rgba(89,60,38,0.10); color: var(--secondary); }}
.section-title {{ font-size: 1rem; font-weight: 700; color: var(--text); margin-bottom: 10px; font-family: 'Inter', sans-serif; letter-spacing: -0.01em; }}
.footer {{ border-top: 1px solid var(--border); margin-top: 32px; padding-top: 16px; color: var(--muted); font-size: 0.78rem; display: flex; justify-content: space-between; align-items: center; }}
*, *::before, *::after {{ transition: background-color 0.2s ease, border-color 0.2s ease; }}

@media (max-width: 760px) {{
  .block-container {{ padding: 0.75rem 0.75rem 2rem; }}
  .page-title {{ font-size: 1.55rem; }}
  .dashboard-card, .kpi-card {{ padding: 15px; }}
  .topbar {{ padding: 9px 12px; margin-bottom: 12px; }}
  .footer {{ align-items: flex-start; flex-direction: column; gap: 6px; }}
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; gap: 0.65rem; }}
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
    min-width: min(100%, 280px) !important; flex: 1 1 100% !important;
  }}
}}

@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    scroll-behavior: auto !important; animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important; transition-duration: 0.01ms !important;
  }}
}}
</style>
""", unsafe_allow_html=True)

inject_css()

# ─────────────────────── Helpers ─────────────────────────────

RETAILIQ_COLORSCALE = [
    [0.0, "#f2e4d4"], [0.25, "#e8935a"], [0.5, "#da6f0b"],
    [0.75, "#593c26"], [1.0, "#1a1a1a"],
]

def fmt(x, money=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    x = float(x); sign = "-" if x < 0 else ""; ax = abs(x)
    if money:
        if ax >= 1e9: return f"{sign}₹{ax/1e9:.2f}B"
        if ax >= 1e6: return f"{sign}₹{ax/1e6:.2f}M"
        if ax >= 1e3: return f"{sign}₹{ax/1e3:.1f}K"
        return f"{sign}₹{ax:,.0f}"
    if ax >= 1e9: return f"{sign}{ax/1e9:.2f}B"
    if ax >= 1e6: return f"{sign}{ax/1e6:.2f}M"
    if ax >= 1e3: return f"{sign}{ax/1e3:.1f}K"
    return f"{sign}{ax:,.0f}" if float(ax).is_integer() else f"{sign}{ax:,.2f}"

def label(x): return str(x).replace("_", " ").title()

def business_metric(df):
    priorities = ["sales","revenue","profit","amount","value","net_sales","demand","quantity"]
    for p in priorities:
        for c in df.columns:
            if str(c).lower() == p and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>1:
                return c
    return None

def numeric_cols(df):
    return [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique(dropna=True)>1]

def category_cols(df):
    out = []
    for c in df.select_dtypes(include=["object","string","category","bool"]).columns:
        n = df[c].nunique(dropna=True)
        if 2 <= n <= min(100, max(2, int(len(df)*.8))): out.append(c)
    return out

def date_cols(profile): return profile.get("date_columns", [])

def clear_dataset_state():
    keys = ["raw_df","cleaned_df","profile","clean_profile","cleaning_log","uploaded_name","filters",
            "ml_results","anomaly_result","cluster_result","forecast_result","advanced_result",
            "recommendations","dataset_summary","comparison_result","hybrid_model",
            "selected_customer_index","customer_explanation","uploaded_fingerprint"]
    for k in keys: st.session_state[k] = [] if k in ["cleaning_log","recommendations"] else ({} if k in ["filters","ml_results","dataset_summary","comparison_result","hybrid_model","customer_explanation"] else (0 if k=="selected_customer_index" else None))
    st.session_state.upload_widget_version += 1
    st.session_state.project_definition = build_project_definition()
    st.session_state.selected_page = "Overview"
    st.session_state.page_search = "Overview"
    st.session_state.confirm_clear_dataset = False

def _safe_numeric_series(df, column):
    if column not in df.columns: return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()

def _safe_label_series(df, column):
    if column not in df.columns: return pd.Series(dtype=str)
    return df[column].astype(str).fillna("Missing")

def run_statistical_analysis(df, test_name, numeric_col=None, group_col=None, category_col=None):
    alpha = 0.05
    if test_name == "T-test":
        if not numeric_col or not group_col or numeric_col not in df.columns or group_col not in df.columns:
            return {"ok": False, "message": "Select both a numeric variable and a binary grouping variable to run the t-test."}
        grouped = df[[group_col, numeric_col]].dropna().copy()
        if grouped.empty: return {"ok": False, "message": "No valid observations remain after removing missing values."}
        labels = pd.Series(grouped[group_col].astype(str)).dropna().unique()
        if len(labels) != 2: return {"ok": False, "message": "The selected grouping variable must contain exactly two groups for an independent two-sample t-test."}
        g1 = grouped[grouped[group_col].astype(str) == labels[0]][numeric_col]
        g2 = grouped[grouped[group_col].astype(str) == labels[1]][numeric_col]
        if len(g1) < 2 or len(g2) < 2: return {"ok": False, "message": "Each group needs at least two non-missing values for a valid t-test."}
        stat, p_value = ttest_ind(g1, g2, equal_var=False)
        return {"ok": True, "test_name": "Independent two-sample t-test", "variables": [numeric_col, group_col],
                "statistic": float(stat), "p_value": float(p_value), "significance_level": alpha,
                "significant": bool(p_value < alpha),
                "interpretation": "p < 0.05 indicates a statistically significant difference between the two groups; p >= 0.05 indicates no statistically significant difference.",
                "group_1_mean": float(g1.mean()), "group_2_mean": float(g2.mean()),
                "group_1_label": str(labels[0]), "group_2_label": str(labels[1])}
    if test_name == "Chi-square":
        if not category_col or not group_col or category_col not in df.columns or group_col not in df.columns:
            return {"ok": False, "message": "Select two categorical variables to run the chi-square test."}
        col1 = _safe_label_series(df, category_col); col2 = _safe_label_series(df, group_col)
        valid = pd.DataFrame({category_col: col1, group_col: col2}).dropna()
        if valid.empty: return {"ok": False, "message": "No valid categorical observations remain after removing missing values."}
        if valid[category_col].nunique() < 2 or valid[group_col].nunique() < 2:
            return {"ok": False, "message": "The selected categorical variables must each contain at least two non-empty groups for a chi-square test."}
        if valid[category_col].nunique() > 20 or valid[group_col].nunique() > 20:
            return {"ok": False, "message": "One or both categorical variables are too high-cardinality for a reliable chi-square test on the current dataset."}
        table = pd.crosstab(valid[category_col], valid[group_col])
        if table.shape[0] < 2 or table.shape[1] < 2:
            return {"ok": False, "message": "Not enough valid category combinations are present to perform chi-square testing."}
        chi2, p_value, dof, expected = chi2_contingency(table)
        return {"ok": True, "test_name": "Chi-square test of independence", "variables": [category_col, group_col],
                "contingency_table": table, "statistic": float(chi2), "degrees_of_freedom": int(dof),
                "p_value": float(p_value), "significance_level": alpha, "significant": bool(p_value < alpha),
                "interpretation": "p < 0.05 indicates a statistically significant association between the two categorical variables; p >= 0.05 indicates no statistically significant association."}
    if test_name == "ANOVA":
        if not numeric_col or not group_col or numeric_col not in df.columns or group_col not in df.columns:
            return {"ok": False, "message": "Select a numeric variable and a grouping variable with at least three groups to run ANOVA."}
        groups_df = df[[group_col, numeric_col]].dropna().copy()
        if groups_df.empty: return {"ok": False, "message": "No valid observations remain after removing missing values."}
        if groups_df[group_col].nunique() < 3: return {"ok": False, "message": "ANOVA requires at least three groups in the selected grouping variable."}
        group_values = [g for _, g in groups_df.groupby(group_col)[numeric_col]]
        if len(group_values) < 3: return {"ok": False, "message": "Not enough valid groups are available for ANOVA."}
        if any(len(g) < 2 for g in group_values): return {"ok": False, "message": "Each ANOVA group needs at least two non-missing observations."}
        f_value, p_value = f_oneway(*group_values)
        return {"ok": True, "test_name": "One-way ANOVA", "variables": [numeric_col, group_col],
                "statistic": float(f_value), "p_value": float(p_value), "significance_level": alpha,
                "significant": bool(p_value < alpha),
                "interpretation": "p < 0.05 indicates that the numeric variable differs significantly across the groups; p >= 0.05 indicates no statistically significant group difference.",
                "group_count": int(groups_df[group_col].nunique())}
    return {"ok": False, "message": "The selected statistical test is not available for the current dataset."}

def chart_style(fig, height=330):
    is_dark = st.session_state.theme == "dark"
    grid_color = "rgba(242,228,212,0.24)" if is_dark else "#d2c2af"
    fig.update_layout(
        template="plotly_dark" if is_dark else "plotly_white",
        height=height, margin=dict(l=8,r=8,t=44,b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=C["text"], size=12),
        legend_title_text="", legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=C["text"])),
        hoverlabel=dict(
            bgcolor=C["surface"], bordercolor=C["border"],
            font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        ),
        xaxis=dict(
            gridcolor=grid_color, gridwidth=1, zerolinecolor=C["muted"],
            linecolor=C["muted"], linewidth=1, showline=True, mirror=True,
            tickfont=dict(size=11, color=C["text"]),
            title_font=dict(size=11, color=C["muted"]), ticks="outside",
        ),
        yaxis=dict(
            gridcolor=grid_color, gridwidth=1, zerolinecolor=C["muted"],
            linecolor=C["muted"], linewidth=1, showline=True, mirror=True,
            tickfont=dict(size=11, color=C["text"]),
            title_font=dict(size=11, color=C["muted"]), ticks="outside",
        ),
        title_font=dict(size=14, color=C["text"], family="Inter, sans-serif"),
        colorway=[C["chart1"],C["chart2"],C["chart3"],C["chart4"],C["chart5"]],
    )
    return fig

def card(title, body, icon="◈", accent=None):
    accent = accent or C["primary"]
    st.markdown(
        f"<div class='dashboard-card' style='border-top:3px solid {accent}'>"
        f"<div class='card-title'><span class='card-title-icon' style='background:rgba(218,111,11,0.10);color:{accent}'>{icon}</span>{title}</div>"
        f"<div style='font-size:0.86rem;color:{C['muted']};line-height:1.55'>{body}</div></div>",
        unsafe_allow_html=True,
    )

def kpi_card(label_text, value, icon, accent, subtext=None):
    sub = f"<div class='kpi-subtext'>{subtext}</div>" if subtext else ""
    st.markdown(
        f"<div class='kpi-card' style='--accent-color:{accent}'>"
        f"<span class='kpi-icon' style='color:{accent}'>{icon}</span>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-label'>{label_text}</div>{sub}</div>",
        unsafe_allow_html=True,
    )

def set_page(page):
    st.session_state.selected_page = page
    st.session_state.page_search = page

def handle_page_search():
    st.session_state.selected_page = st.session_state.page_search

def load_uploaded(uploaded):
    if uploaded is None: return
    content = uploaded.getvalue()
    fingerprint = hashlib.sha256(uploaded.name.lower().encode("utf-8") + b"\0" + content).hexdigest()
    if st.session_state.uploaded_fingerprint == fingerprint and st.session_state.raw_df is not None: return
    try:
        source = io.BytesIO(content)
        raw = pd.read_csv(source) if uploaded.name.lower().endswith(".csv") else pd.read_excel(source)
    except Exception:
        logger.exception("Could not read uploaded dataset %s", uploaded.name)
        st.error("Could not read the dataset. Check that the file is a valid CSV or Excel workbook.")
        return
    if raw.empty: st.error("The uploaded dataset is empty."); return
    with st.spinner("Profiling and preparing the dataset..."):
        profile = analyze_dataset(raw)
        plan = build_cleaning_plan(raw)
        cleaned, log = clean_dataset(raw, plan)
        clean_profile = analyze_dataset(cleaned)
        st.session_state.raw_df = raw; st.session_state.cleaned_df = cleaned
        st.session_state.profile = profile; st.session_state.clean_profile = clean_profile
        st.session_state.cleaning_log = log; st.session_state.uploaded_name = uploaded.name
        st.session_state.uploaded_fingerprint = fingerprint
        st.session_state.filters = {}; st.session_state.ml_results = {}
        st.session_state.anomaly_result = None; st.session_state.cluster_result = None
        st.session_state.forecast_result = None; st.session_state.advanced_result = None
        st.session_state.selected_customer_index = 0
        st.session_state.customer_explanation = {}
        st.session_state.comparison_result = {}
        st.session_state.hybrid_model = {}
        st.session_state.recommendations = generate_recommendations(clean_profile, cleaned)
        update_dataset_context(cleaned)
        st.rerun()

def filtered_df():
    df = st.session_state.cleaned_df.copy()
    if df is None: return None
    prof = st.session_state.clean_profile or {}
    for c, vals in st.session_state.filters.items():
        if c in df.columns and vals: df = df[df[c].astype(str).isin(vals)]
    for d in prof.get("date_columns", []):
        if d in df.columns: df[d] = pd.to_datetime(df[d], errors="coerce")
    return df

def top_dimension(df, exclude=None):
    exclude = set(exclude or []); cats = category_cols(df)
    cats = [c for c in cats if c not in exclude]
    if not cats: return None
    cats.sort(key=lambda c: (df[c].nunique(dropna=True) > 20, df[c].nunique(dropna=True)))
    return cats[0]

def safe_group(df, dim, measure, n=10):
    if not dim or not measure or dim not in df.columns or measure not in df.columns: return pd.DataFrame()
    x = df[[dim,measure]].dropna().groupby(dim,as_index=False)[measure].sum()
    return x.sort_values(measure,ascending=False).head(n)

def update_dataset_context(df):
    if df is None or df.empty: return
    st.session_state.dataset_summary = summarize_dataset(df)
    st.session_state.project_definition = build_project_definition()
    target = st.session_state.dataset_summary.get("target_column")
    if target:
        try:
            st.session_state.comparison_result = compare_models(df, target=target)
        except Exception:
            logger.exception("Model comparison failed during dataset load")
            st.session_state.comparison_result = {
                "ok": False,
                "error": "Model comparison could not run. Review application logs for details.",
            }
        try:
            st.session_state.hybrid_model = build_hybrid_model(df, target=target)
        except Exception:
            logger.exception("Hybrid model failed during dataset load")
            st.session_state.hybrid_model = {
                "ok": False,
                "error": "Hybrid model could not run. Review application logs for details.",
            }
    else:
        st.session_state.comparison_result = {"ok": False, "error": "No target column was detected for supervised analysis."}
        st.session_state.hybrid_model = {"ok": False, "error": "No target column was detected for hybrid modeling."}

# ─────────────────── DAV Compliance Renderers ─────────────────

def render_project_definition():
    definition = st.session_state.get("project_definition") or build_project_definition()
    st.markdown("<div class='page-kicker'>Project Definition</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>RetailIQ Customer Churn Decision Support</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>A data-grounded, explainable, hybrid-model analytics workflow for retention-focused decision support.</div>", unsafe_allow_html=True)
    for title, body in [("Problem statement", definition["problem_statement"]),("Business objective", definition["business_objective"]),("Analytics objective", definition["analytics_objective"]),("Research objective", definition["research_objective"]),("Target variable", definition["target_variable"]),("Prediction task", definition["prediction_task"])]:
        st.markdown(f"<div class='insight'><strong>{title}:</strong> {body}</div>", unsafe_allow_html=True)
    with st.expander("Key performance indicators", expanded=True): st.write(definition["kpis"])
    with st.expander("Research questions", expanded=True): st.write(definition["research_questions"])
    with st.expander("Business questions", expanded=True): st.write(definition["business_questions"])
    with st.expander("Constraints and limitations", expanded=True): st.write(definition["constraints_and_limitations"])

def render_dataset_understanding(df):
    st.markdown("<div class='page-kicker'>Dataset Understanding</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Data Profiling & Understanding</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Review how the dataset is structured, which fields are relevant, and what the target looks like before model training.</div>", unsafe_allow_html=True)
    raw_df = st.session_state.get("raw_df")
    cleaned_df = st.session_state.get("cleaned_df")
    raw_df = raw_df if isinstance(raw_df, pd.DataFrame) else df
    cleaned_df = cleaned_df if isinstance(cleaned_df, pd.DataFrame) else df
    summary = summarize_dataset(cleaned_df)
    raw_duplicates = int(raw_df.duplicated().sum())
    cleaned_duplicates = int(cleaned_df.duplicated().sum())
    metric_specs = [
        ("Original rows", len(raw_df)),
        ("Cleaned rows", len(cleaned_df)),
        ("Duplicates before cleaning", raw_duplicates),
        ("Duplicates remaining", cleaned_duplicates),
        ("Columns", summary["columns"]),
        ("Numeric", len(summary["numeric_columns"])),
        ("Categorical", len(summary["categorical_columns"])),
        ("Missing values", summary["missing_total"]),
    ]
    for start in range(0, len(metric_specs), 4):
        metric_cols = st.columns(4)
        for col, (metric_label, value) in zip(metric_cols, metric_specs[start:start + 4]):
            col.metric(metric_label, value)
    tab1, tab2 = st.tabs(["Overview", "Column details"])
    with tab1:
        if summary.get("target_column"):
            target_c = summary["target_column"]
            target_task, distribution = build_target_distribution_data(df[target_c])
            if target_task == "regression" and not distribution.empty:
                bins = min(30, max(10, int(np.ceil(np.sqrt(len(distribution))))))
                fig = px.histogram(
                    distribution,
                    x="Target value",
                    nbins=bins,
                    title=f"Target Distribution: {target_c}",
                    labels={"Target value": f"{target_c} value", "count": "Rows"},
                )
                fig.update_layout(xaxis_title=f"{target_c} value", yaxis_title="Rows", bargap=0.08)
                fig.update_traces(
                    marker_color=C["chart1"],
                    marker_line_color=C["surface"],
                    marker_line_width=1,
                    opacity=0.95,
                    hovertemplate=f"{target_c} range: %{{x}}<br>Rows: %{{y}}<extra></extra>",
                )
                st.plotly_chart(chart_style(fig, 330), use_container_width=True)
                st.caption(
                    f"Continuous target values across {len(distribution):,} non-missing analysis rows; "
                    "the histogram bins actual target values."
                )
            elif not distribution.empty:
                fig = px.bar(
                    distribution,
                    x="Target class",
                    y="Rows",
                    title=f"Target Distribution: {target_c}",
                    text="Rows",
                    labels={"Target class": target_c, "Rows": "Rows"},
                )
                fig.update_traces(
                    marker_color=C["chart1"],
                    marker_line_color=C["surface"],
                    marker_line_width=1,
                    textposition="outside",
                    textfont_color=C["text"],
                    cliponaxis=False,
                    hovertemplate=f"{target_c}: %{{x}}<br>Rows: %{{y}}<extra></extra>",
                )
                st.plotly_chart(chart_style(fig, 330), use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Metric": [
                "Duplicate rows detected before cleaning",
                "Duplicate rows remaining after cleaning",
                "Target column",
                "Task-relevant target balance",
            ],
            "Value": [
                raw_duplicates,
                cleaned_duplicates,
                summary["target_column"],
                summary["class_imbalance"],
            ],
        }), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(pd.DataFrame([{"Column":col,"Data Type":summary["data_types"].get(col),"Missing":summary["missing_values"].get(col,0),"Unique Values":summary["unique_values"].get(col,0)} for col in df.columns]), use_container_width=True, hide_index=True)

def render_data_cleaning(df):
    st.markdown("<div class='page-kicker'>Data Cleaning</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Cleaning & Preprocessing Decisions</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>The cleaning pipeline is preserved and decisions are surfaced in a transparent, dataset-aware manner.</div>", unsafe_allow_html=True)
    plan = build_cleaning_plan(df); cleaned, log = clean_dataset(df, plan); decisions = build_preprocessing_decisions(cleaned)
    st.success(f"{len(log)} cleaning actions were recorded during the active preprocessing pipeline.")
    if log: st.dataframe(pd.DataFrame(log), use_container_width=True, hide_index=True)
    with st.expander("Selected preprocessing decisions", expanded=True):
        for decision in decisions["decisions"]:
            st.markdown(f"<div class='insight'><strong>{decision['step']}:</strong> {decision['method']} — {decision['reason']}</div>", unsafe_allow_html=True)

def render_eda_statistics(df):
    st.markdown("<div class='page-kicker'>EDA & Statistics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Univariate, bivariate, multivariate, and statistical views remain data-type aware and only show what is meaningful for the active dataset.</div>", unsafe_allow_html=True)
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if numeric:
        col1, col2 = st.columns(2)
        with col1:
            visual_col = st.selectbox("Distribution", numeric[:5])
            fig = px.histogram(df, x=visual_col, nbins=25, title=f"Distribution of {visual_col}")
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig, 300), use_container_width=True)
        with col2:
            if len(numeric) >= 2:
                x_col = st.selectbox("X variable", numeric, index=0)
                y_col = st.selectbox("Y variable", numeric, index=1 if len(numeric) > 1 else 0)
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
                fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig, 300), use_container_width=True)
    if numeric and len(numeric) >= 2:
        corr = df[numeric].corr().round(3)
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", color_continuous_scale=RETAILIQ_COLORSCALE)
        st.plotly_chart(chart_style(fig, 380), use_container_width=True)
    if st.checkbox("Show descriptive statistics", value=False): st.dataframe(df.describe().T, use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("<div class='page-kicker'>Statistical Significance Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Hypothesis Testing</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Use the appropriate significance test only when the data supports it. p < 0.05 is treated as statistically significant; p >= 0.05 is not significant.</div>", unsafe_allow_html=True)
    stat_test = st.selectbox("Statistical test", ["T-test", "Chi-square", "ANOVA"], index=0)
    if stat_test == "T-test":
        binary_candidates = [col for col in df.columns if df[col].nunique(dropna=True) == 2 or pd.api.types.is_bool_dtype(df[col])]
        if not numeric or not binary_candidates:
            st.info("A valid numeric variable and a binary grouping variable are required for an independent two-sample t-test on this dataset.")
        else:
            num_col = st.selectbox("Numeric variable", numeric, index=0)
            group_col = st.selectbox("Binary grouping column", binary_candidates, index=0)
            result = run_statistical_analysis(df, "T-test", numeric_col=num_col, group_col=group_col)
            if result.get("ok"):
                r1, r2 = st.columns(2)
                r1.metric("Group 1 mean", f"{result['group_1_mean']:.4f}", help=result['group_1_label'])
                r2.metric("Group 2 mean", f"{result['group_2_mean']:.4f}", help=result['group_2_label'])
                st.markdown(f"<div class='insight'><strong>{result['test_name']}</strong><br>t-statistic: <strong>{result['statistic']:.4f}</strong> &nbsp; p-value: <strong>{result['p_value']:.6f}</strong> &nbsp; Significant: <strong>{'Yes ✓' if result['significant'] else 'No'}</strong><br><span style='margin-top:6px;display:block'>{result['interpretation']}</span></div>", unsafe_allow_html=True)
                fig = px.box(df[[group_col, num_col]].dropna(), x=group_col, y=num_col, title=f"{num_col} by {group_col}")
                fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig, 300), use_container_width=True)
            else: st.warning(result.get("message"))
    elif stat_test == "Chi-square":
        cat_candidates = [col for col in df.columns if str(df[col].dtype).startswith("object") or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])]
        if len(cat_candidates) < 2: st.info("Two suitable categorical variables are required for a chi-square test of independence.")
        else:
            cat_a = st.selectbox("Categorical variable A", cat_candidates, index=0)
            cat_b = st.selectbox("Categorical variable B", [c for c in cat_candidates if c != cat_a], index=0)
            result = run_statistical_analysis(df, "Chi-square", category_col=cat_a, group_col=cat_b)
            if result.get("ok"):
                st.dataframe(result['contingency_table'], use_container_width=True, hide_index=True)
                st.markdown(f"<div class='insight'><strong>{result['test_name']}</strong><br>Variables: {result['variables'][0]} x {result['variables'][1]}<br>chi2 = <strong>{result['statistic']:.4f}</strong> &nbsp; df = {result['degrees_of_freedom']} &nbsp; p-value = <strong>{result['p_value']:.6f}</strong> &nbsp; Significant: <strong>{'Yes' if result['significant'] else 'No'}</strong><br><span style='margin-top:6px;display:block'>{result['interpretation']}</span></div>", unsafe_allow_html=True)
                table = result['contingency_table'].reset_index()
                fig = px.bar(table, x=table.columns[0], y=[c for c in table.columns[1:]], barmode='group', title=f"Contingency: {cat_a} x {cat_b}")
                st.plotly_chart(chart_style(fig, 300), use_container_width=True)
            else: st.warning(result.get("message"))
    else:
        anova_cands = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique(dropna=True) > 1]
        group_cands = [col for col in df.columns if df[col].nunique(dropna=True) >= 3 and (str(df[col].dtype).startswith("object") or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))]
        if not anova_cands or not group_cands: st.info("ANOVA requires a numeric variable and a grouping variable with at least three groups.")
        else:
            num_col = st.selectbox("Numeric variable", anova_cands, index=0)
            group_col = st.selectbox("Grouping variable", group_cands, index=0)
            result = run_statistical_analysis(df, "ANOVA", numeric_col=num_col, group_col=group_col)
            if result.get("ok"):
                st.markdown(f"<div class='insight'><strong>{result['test_name']}</strong><br>Numeric: {num_col} | Groups: {group_col} ({result['group_count']} groups)<br>F = <strong>{result['statistic']:.4f}</strong> &nbsp; p-value = <strong>{result['p_value']:.6f}</strong> &nbsp; Significant: <strong>{'Yes' if result['significant'] else 'No'}</strong><br><span style='margin-top:6px;display:block'>{result['interpretation']}</span></div>", unsafe_allow_html=True)
                fig = px.box(df[[group_col, num_col]].dropna(), x=group_col, y=num_col, title=f"{num_col} by {group_col}")
                fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig, 300), use_container_width=True)
            else: st.warning(result.get("message"))

def render_feature_engineering(df):
    st.markdown("<div class='page-kicker'>Feature Engineering</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Feature Creation & Selection</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Only features supported by the uploaded data are created, and each feature is documented from actual source columns.</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    engineered = engineer_features(df, target_column=target); feature_descriptions = engineered.attrs.get("feature_descriptions", {})
    new_features = [col for col in engineered.columns if col not in df.columns]
    if not new_features: st.info("No additional features were created because the dataset does not contain the required source columns.")
    else: st.dataframe(pd.DataFrame({"Feature": new_features, "Description": [feature_descriptions.get(col, "Derived from available business data.") for col in new_features]}), use_container_width=True, hide_index=True)
    st.dataframe(engineered.head(10), use_container_width=True, hide_index=True)

def render_predictive_analytics(df):
    st.markdown("<div class='page-kicker'>Predictive Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Supervised Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>RetailIQ detects whether the selected target is categorical or continuous and reports matching holdout metrics.</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    if not target: st.warning("No target column is currently detected for supervised prediction."); return
    st.write(f"Detected target: **{target}**")
    task = infer_task(df[target]); st.caption(f"Detected task: **{task}**")
    result = run_supervised(df, target, task)
    if result.get("ok"):
        st.dataframe(result["results"], use_container_width=True, hide_index=True)
        if target.strip().lower() == "sales" and {"cost", "profit", "profit_margin"}.intersection(result.get("dropped_features", [])):
            st.info(
                "Cost, profit, and profit margin are excluded from future sales predictors because they are typically known only after a sale and can leak the outcome."
            )
        if result.get("dropped_features"):
            st.caption(f"Excluded unusable or leakage-prone predictors: {', '.join(result['dropped_features'])}")
    else:
        st.error(result.get("error", "Prediction did not run."))

def render_model_comparison(df):
    st.markdown("<div class='page-kicker'>Model Comparison</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Baseline vs Existing vs Proposed Hybrid</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    if not target: st.warning("A target must be detected before the comparison view is meaningful."); return
    comparison = compare_models(df, target=target)
    if not comparison.get("ok"): st.error(comparison.get("error", "Model comparison failed.")); return
    models = comparison["models"].copy(); st.dataframe(models, use_container_width=True, hide_index=True)
    task = comparison.get("task")
    metrics = (
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
        if task == "classification"
        else ["MAE", "RMSE", "R2"]
    )
    for metric_n in metrics:
        if metric_n in models.columns:
            ascending = task == "regression" and metric_n in {"MAE", "RMSE"}
            plot_data = models.dropna(subset=[metric_n]).sort_values(metric_n, ascending=ascending)
            if not plot_data.empty:
                fig = px.bar(plot_data, x="Model", y=metric_n, title=f"{metric_n} by Model", text_auto=".3f")
                fig.update_traces(marker_color=C["chart1"], marker_line_color=C["surface"], marker_line_width=1)
                st.plotly_chart(chart_style(fig, 260), use_container_width=True)
    if comparison.get("best_overall"):
        best = comparison["best_overall"]
        if task == "regression":
            st.success(f"Lowest holdout RMSE: **{best['Model']}** ({best['RMSE']:.4f}).")
        else:
            st.success(f"Best weighted F1: **{best['Model']}** ({best['F1']:.4f}).")

def render_hybrid_model(df):
    st.markdown("<div class='page-kicker'>Hybrid Model</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>RetailIQ Proposed Hybrid Ensemble</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    if not target: st.warning("The hybrid model requires a target column."); return
    hybrid = build_hybrid_model(df, target=target)
    if not hybrid.get("ok"): st.error(hybrid.get("error", "Hybrid model training failed.")); return
    metrics = hybrid["metrics"]; st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)
    st.write(f"Model name: **{hybrid['model_name']}**")
    if hybrid["task"] == "classification":
        cm = np.array(hybrid["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix", color_continuous_scale=RETAILIQ_COLORSCALE)
        st.plotly_chart(chart_style(fig, 300), use_container_width=True)
        st.success("The hybrid classifier combines fitted base estimators using stacking (or soft voting when rare classes make cross-validation unsafe).")
    else:
        actual_vs_predicted = pd.DataFrame({
            "Actual": hybrid["y_test"].to_numpy(),
            "Predicted": hybrid["predictions"],
        })
        fig = px.scatter(actual_vs_predicted, x="Actual", y="Predicted", title="Holdout: Actual vs Predicted")
        fig.update_traces(marker_color=C["chart1"], marker_line_color=C["surface"], marker_line_width=1)
        st.plotly_chart(chart_style(fig, 300), use_container_width=True)
        if hybrid.get("dropped_features"):
            st.caption(f"Excluded unusable or leakage-prone predictors: {', '.join(hybrid['dropped_features'])}")

def render_explainable_ai(df):
    st.markdown("<div class='page-kicker'>Explainable AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>SHAP & LIME</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    if not target: st.warning("Explainability requires a valid target column."); return
    fitted = fit_explanation_model(df, target)
    if not fitted.get("ok"):
        st.warning(fitted.get("error", "The target is not suitable for explainability."))
        return
    st.caption(f"Detected task: **{fitted['task']}**")
    if target.strip().lower() == "sales" and {"cost", "profit", "profit_margin"}.intersection(fitted.get("dropped_features", [])):
        st.info("Cost, profit, and profit margin are excluded because they are typically post-sale values unavailable when predicting future sales.")
    idx = st.slider('Evaluation row index', min_value=0, max_value=max(0, len(fitted["X_test"]) - 1), value=0, step=1)
    shap_result = generate_shap_explanation(fitted["model"], fitted["X_train"], fitted["X_test"], customer_index=idx)
    lime_result = generate_lime_explanation(fitted["model"], fitted["X_train"], fitted["X_test"], customer_index=idx, target_name=target)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('SHAP')
        if shap_result.get('ok'):
            label = "Predicted class" if fitted["task"] == "classification" else "Predicted value"
            st.metric(label, str(shap_result['prediction']) if fitted["task"] == "classification" else f"{shap_result['prediction']:.4f}")
            st.dataframe(pd.DataFrame(shap_result['top_positive']), use_container_width=True, hide_index=True)
        else: st.warning(shap_result.get('error'))
    with col2:
        st.subheader('LIME')
        if lime_result.get('ok'):
            label = "Predicted class" if fitted["task"] == "classification" else "Predicted value"
            st.metric(label, str(lime_result['prediction']) if fitted["task"] == "classification" else f"{lime_result['prediction']:.4f}")
            st.dataframe(pd.DataFrame(lime_result['top_features'], columns=['Feature', 'Contribution']), use_container_width=True, hide_index=True)
        else: st.warning(lime_result.get('error'))
    st.markdown("<div class='insight'><strong>SHAP:</strong> feature contributions within the fitted model. <strong>LIME:</strong> a local surrogate around one observation. Neither establishes causality.</div>", unsafe_allow_html=True)

def render_visual_analytics(df):
    st.markdown("<div class='page-kicker'>Visual Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Multi-dimensional Decision Support</div>", unsafe_allow_html=True)
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    if not numeric: st.info("No numeric fields are available for visual analysis."); return
    x_col = st.selectbox("X-axis", numeric, index=0)
    y_col = st.selectbox("Y-axis", numeric, index=1 if len(numeric) > 1 else 0)
    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
    fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig, 380), use_container_width=True)
    if len(numeric) >= 2:
        corr = df[numeric].corr().round(3)
        heatmap = px.imshow(corr, labels=dict(color="Correlation"), title="Correlation Heatmap", color_continuous_scale=RETAILIQ_COLORSCALE)
        st.plotly_chart(chart_style(heatmap, 340), use_container_width=True)

def render_customer_risk_analysis(df):
    st.markdown("<div class='page-kicker'>Customer Risk Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Risk Review & Decision Support</div>", unsafe_allow_html=True)
    target = st.session_state.dataset_summary.get("target_column") or detect_target(df)
    if not target: st.warning("Customer risk analysis requires a target column such as churn."); return
    if infer_task(df[target]) == "regression":
        st.info("Customer risk analysis requires a categorical risk outcome; the selected target is continuous and is treated as regression.")
        return
    target_tokens = ("churn", "attrition", "risk", "default", "fraud", "delinquen")
    if not any(token in target.lower() for token in target_tokens):
        st.info(f"'{target}' is not an explicitly labeled customer-risk outcome. Risk analysis requires a churn, attrition, default, fraud, or similar label.")
        return
    fitted = fit_explanation_model(df, target)
    if not fitted.get("ok"):
        st.warning(fitted.get("error", "The risk model could not be trained for this dataset."))
        return
    if fitted["task"] != "classification":
        st.info("Customer risk analysis requires a categorical risk outcome such as churn or default.")
        return
    model = fitted["model"]
    classes = list(model.named_steps["model"].classes_)
    if len(classes) != 2:
        st.info("Customer risk analysis currently requires a binary labeled outcome.")
        return
    positive_index = positive_risk_class_index(classes)
    if positive_index is None:
        st.info(
            "The positive risk class could not be identified from the target labels. "
            f"Review these labels and use a recognizable positive value: {classes}."
        )
        return
    X_train, X_test = fitted["X_train"], fitted["X_test"]
    idx = st.slider('Select evaluation record', min_value=0, max_value=max(0, len(X_test) - 1), value=0, step=1)
    prob = model.predict_proba(X_test.iloc[[idx]])[0, positive_index]
    risk_label = 'High Risk' if prob >= 0.6 else 'Medium Risk' if prob >= 0.35 else 'Low Risk'
    explanation = generate_shap_explanation(
        model, X_train, X_test, customer_index=idx, class_index=positive_index
    )
    drivers = explanation.get('top_positive', []) if explanation.get('ok') else []
    recommendation = build_customer_recommendation(probability=float(prob), risk_label=risk_label, drivers=drivers)
    col1, col2 = st.columns(2)
    col1.metric('Probability of churn', f'{prob:.2%}'); col2.metric('Risk level', risk_label)
    st.caption(f"Positive risk class: {classes[positive_index]}")
    st.markdown(f"<div class='insight'><strong>Recommended action:</strong> {recommendation['action']}</div>", unsafe_allow_html=True)
    if explanation.get('ok'): st.dataframe(pd.DataFrame(explanation['top_positive']), use_container_width=True, hide_index=True)
    else: st.warning(explanation.get("error", "Feature explanation is not available for this model."))

def render_data_story(df):
    story = build_data_story(df, st.session_state.get("comparison_result"))
    st.markdown("<div class='page-kicker'>Data Story</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Narrative Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>A concise, evidence-based story built from the actual project results.</div>", unsafe_allow_html=True)
    for title, body in [("What Happened?", story["what_happened"]),("Why Did It Happen?", story["why_it_happened"]),("What Is Likely to Happen?", story["what_is_likely_to_happen"]),("What Should We Do?", story["what_should_we_do"])]:
        st.markdown(f"<div class='insight'><strong>{title}</strong><br><span style='margin-top:6px;display:block'>{body}</span></div>", unsafe_allow_html=True)

def render_methodology():
    st.markdown("<div class='page-kicker'>Methodology</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Analytical Workflow</div>", unsafe_allow_html=True)
    for step, detail in [("Problem formulation and target definition","Establish what the business wants to predict or understand."),("Dataset understanding and profiling","Analyze column types, distributions, missing values, and cardinality."),("Data cleaning and preprocessing","Apply a transparent, reproducible cleaning pipeline."),("EDA and statistical analysis","Univariate, bivariate, multivariate, and hypothesis testing."),("Feature engineering and selection","Create domain-relevant features; select meaningful predictors."),("Model benchmarking and comparison","Evaluate baseline and standard models on the same hold-out set."),("Hybrid ensemble construction","Stack base learners for improved generalization."),("Explainable AI with SHAP and LIME","Interpret model decisions at global and local levels."),("Decision support and responsible AI review","Ensure fairness, transparency, and human oversight.")]:
        st.markdown(f"<div class='insight'><strong>✓ {step}</strong><br><span style='font-size:0.82rem'>{detail}</span></div>", unsafe_allow_html=True)

def render_responsible_ai(df):
    st.markdown("<div class='page-kicker'>Responsible AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Fairness, Privacy, and Transparency</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Responsible analytics requires transparency about limitations, fairness checks, and human oversight.</div>", unsafe_allow_html=True)
    for icon_title, detail in [("Bias & Fairness","Review performance across major customer segments and demographic groups when available."),("Privacy","Avoid exposing personal data and restrict any identifying fields from the dashboard unless required."),("Transparency","Explain predictions using SHAP and LIME rather than relying on a black-box output alone."),("Limitations","Models can fail when data quality, class imbalance, or label quality are weak."),("Ethical Analytics","Recommendations should be treated as decision-support and reviewed with business context.")]:
        st.markdown(f"<div class='insight'><strong>{icon_title}</strong><br><span style='font-size:0.82rem'>{detail}</span></div>", unsafe_allow_html=True)

# ─────────────────────────── Sidebar ─────────────────────────

nav_groups = [
    ("DAV COMPLIANCE", [
        ("Executive Overview","📋"),("Dataset Understanding","🗂"),("Data Cleaning","🧹"),
        ("EDA & Statistics","📊"),("Feature Engineering","⚙️"),("Predictive Analytics","🤖"),
        ("Model Comparison","🏆"),("Hybrid Model","🔀"),("Explainable AI","🔍"),
        ("Visual Analytics","👁"),("Customer Risk Analysis","⚠️"),("Recommendations","💡"),
        ("Data Story","📖"),("Methodology","📐"),("Responsible AI","🛡️"),
    ]),
    ("WORKSPACE", [
        ("Overview","🏠"),("Data","📁"),("Data Quality","✅"),("Breakdown","📉"),("Relationships","🔗"),
    ]),
    ("ML ANALYSIS", [
        ("Classification","🎯"),("Regression","📈"),("Clustering","🫧"),("Forecasting","🔮"),("Anomaly Detection","🚨"),
    ]),
    ("ADVANCED", [("Research Evaluation","🔬")]),
]
nav_pages = [page_name for _, items in nav_groups for page_name, _ in items]

with st.sidebar:
    with st.container(key="retailiq_logo"):
        st.button(
            "RetailIQ",
            key="retailiq_home",
            use_container_width=True,
            type="secondary",
            help="Return to Executive Overview without changing the active dataset.",
            on_click=set_page,
            args=("Executive Overview",),
        )

    for group, items in nav_groups:
        st.caption(group)
        for page_name, icon in items:
            active = st.session_state.selected_page == page_name
            st.button(
                f"{icon}  {page_name}", key=f"nav_{page_name}",
                use_container_width=True, type="primary" if active else "secondary",
                on_click=set_page, args=(page_name,),
            )

    st.divider()
    is_dark = st.session_state.theme == "dark"
    if st.button(f"{'☀️' if is_dark else '🌙'}  {'Light Mode' if is_dark else 'Dark Mode'}", use_container_width=True, key="theme_toggle"):
        st.session_state.theme = "light" if is_dark else "dark"; st.rerun()
    if st.session_state.uploaded_name:
        if not st.session_state.confirm_clear_dataset:
            if st.button("🗑️  Remove Dataset", use_container_width=True, type="secondary", key="clear_ds"):
                st.session_state.confirm_clear_dataset = True
                st.rerun()
        else:
            st.warning("Remove the active dataset and its session results?")
            confirm_col, cancel_col = st.columns(2)
            with confirm_col:
                if st.button("Remove", use_container_width=True, type="primary", key="confirm_clear_ds"):
                    clear_dataset_state(); st.rerun()
            with cancel_col:
                if st.button("Keep Dataset", use_container_width=True, key="cancel_clear_ds"):
                    st.session_state.confirm_clear_dataset = False
                    st.rerun()
    st.markdown(
        "<div class='sidebar-footer'><b style='color:var(--text)'>Turn data into decisions.</b><br>"
        "<span class='small'>Adaptive insights. Richer exploration. Data-driven action.</span></div>",
        unsafe_allow_html=True,
    )

# ────────────────────── Top Bar ──────────────────────────────

top1, top2 = st.columns([1.4, 1.0])
with top1:
    st.selectbox(
        "Search pages", nav_pages,
        key="page_search", on_change=handle_page_search,
        help="Type to search and navigate to an existing workspace.",
    )
    if st.session_state.uploaded_name:
        raw_rows = len(st.session_state.raw_df); clean_rows = len(st.session_state.cleaned_df) if st.session_state.cleaned_df is not None else 0
        st.caption(f"Active dataset: {st.session_state.uploaded_name} · {raw_rows:,} raw → {clean_rows:,} clean rows")
with top2:
    uploaded = st.file_uploader(
        "Upload Dataset", type=["csv","xlsx","xls"], label_visibility="collapsed",
        key=f"global_upload_{st.session_state.upload_widget_version}",
        help="Upload a CSV, XLSX or XLS dataset to begin analysis",
    )
load_uploaded(uploaded)

# ─────────────────────── Landing Screen ──────────────────────

if st.session_state.cleaned_df is None:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='page-kicker'>Adaptive Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Turn raw data into business decisions.</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Analyze. Discover. Act. RetailIQ automatically adapts its dashboard to the structure of your dataset.</div>", unsafe_allow_html=True)
    feat1, feat2, feat3, feat4 = st.columns(4)
    for col, (icon, title, desc, accent) in zip([feat1,feat2,feat3,feat4],[
        ("📊","Smart Analytics","Auto-detects your data type and recommends the right analysis",C["primary"]),
        ("🤖","ML-Powered","Classification, regression, clustering, forecasting & anomaly detection",C["accent"]),
        ("🔍","Explainable AI","SHAP & LIME explanations for every model prediction",C["secondary"]),
        ("⚡","No-Code","Upload a CSV and get instant insights — no coding required",C["primary"]),
    ]):
        with col:
            st.markdown(f"<div class='dashboard-card' style='border-top:3px solid {accent};text-align:center;padding:24px 16px'><div style='font-size:2rem;margin-bottom:10px'>{icon}</div><div style='font-weight:700;color:var(--text);margin-bottom:6px;font-size:0.92rem'>{title}</div><div class='small'>{desc}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='dashboard-card' style='text-align:center;padding:60px 40px;border:2px dashed {C['border']}'><div style='font-size:3.5rem;margin-bottom:16px'>⬆️</div><div style='font-size:1.35rem;font-weight:800;color:var(--text);margin-bottom:8px'>Drop your dataset to get started</div><div style='color:var(--muted);margin-bottom:16px'>Supports CSV, XLSX, and XLS files</div><div class='badge badge-info' style='margin:0 auto'>Your dashboard will be generated from actual data — no fabricated metrics</div></div>", unsafe_allow_html=True)
    st.stop()

df = filtered_df()
profile = st.session_state.clean_profile
metric = business_metric(df)
nums = numeric_cols(df)
cats = category_cols(df)
dates = profile.get("date_columns", [])
page = st.session_state.selected_page

# ─────────────── DAV Compliance Pages ────────────────────────

if page == "Executive Overview":
    render_project_definition(); render_dataset_understanding(df)
    if st.session_state.get("comparison_result") and st.session_state["comparison_result"].get("ok"):
        st.markdown("### Model comparison snapshot")
        st.dataframe(st.session_state["comparison_result"]["models"], use_container_width=True, hide_index=True)
elif page == "Dataset Understanding": render_dataset_understanding(df)
elif page == "Data Cleaning": render_data_cleaning(df)
elif page == "EDA & Statistics": render_eda_statistics(df)
elif page == "Feature Engineering": render_feature_engineering(df)
elif page == "Predictive Analytics": render_predictive_analytics(df)
elif page == "Model Comparison": render_model_comparison(df)
elif page == "Hybrid Model": render_hybrid_model(df)
elif page == "Explainable AI": render_explainable_ai(df)
elif page == "Visual Analytics": render_visual_analytics(df)
elif page == "Customer Risk Analysis": render_customer_risk_analysis(df)
elif page == "Data Story": render_data_story(df)
elif page == "Methodology": render_methodology()
elif page == "Responsible AI": render_responsible_ai(df)

# ─────────────────────── Overview ────────────────────────────

elif page == "Overview":
    st.markdown("<div class='page-kicker'>Executive Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Business Overview</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Key performance indicators, trends, and insights from your dataset at a glance.</div>", unsafe_allow_html=True)
    if len(df) == 0: st.warning("No rows match the current filters."); st.stop()
    kpi_accents = [C["primary"],C["accent"],C["secondary"],"#e8935a",C["primary"]]
    kpi_icons = ["🗃️","🔢","💰","✅","📌"]
    kpi_specs = []
    kpi_specs.append(("Total Rows",fmt(len(df)),kpi_icons[0],kpi_accents[0]))
    kpi_specs.append(("Total Columns",fmt(df.shape[1]),kpi_icons[1],kpi_accents[1]))
    if metric:
        money = any(x in metric.lower() for x in ["sales","revenue","profit","amount","price","value"])
        kpi_specs.append((f"Total {label(metric)}",fmt(df[metric].sum(),money),kpi_icons[2],kpi_accents[2]))
    else: kpi_specs.append(("Numeric Features",fmt(len(nums)),kpi_icons[2],kpi_accents[2]))
    quality = max(0,100-(df.isna().sum().sum()/max(df.size,1)*100))
    kpi_specs.append(("Data Quality",f"{quality:.1f}%",kpi_icons[3],kpi_accents[3]))
    if cats: kpi_specs.append(("Categories",fmt(len(cats)),kpi_icons[4],kpi_accents[4]))
    kpi_cols = st.columns(len(kpi_specs))
    for col, (lbl, val, icon, accent) in zip(kpi_cols, kpi_specs):
        with col: kpi_card(lbl, val, icon, accent)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    a, b = st.columns([1.8, 1])
    with a:
        if dates and nums:
            d = dates[0]; v = metric or nums[0]
            t = df[[d,v]].dropna().copy(); t[d] = pd.to_datetime(t[d],errors="coerce"); t = t.dropna(subset=[d])
            if not t.empty:
                span = max((t[d].max()-t[d].min()).days,1)
                freq = "D" if span<=60 else ("W" if span<=365 else "ME")
                t = t.set_index(d)[v].resample(freq).sum().reset_index()
                fig = px.area(t,x=d,y=v,title=f"{label(v)} Trend Over Time")
                fig.update_traces(line=dict(color=C["primary"],width=2.5),fillcolor="rgba(218,111,11,0.12)")
                st.plotly_chart(chart_style(fig,380),use_container_width=True)
        elif metric and top_dimension(df):
            dim=top_dimension(df); t=safe_group(df,dim,metric,12).sort_values(metric)
            fig=px.bar(t,x=metric,y=dim,orientation="h",title=f"{label(metric)} by {label(dim)}",text_auto=".2s")
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,380),use_container_width=True)
        elif len(nums)>=2:
            fig=px.scatter(df,x=nums[0],y=nums[1],title=f"{label(nums[0])} vs {label(nums[1])}")
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,380),use_container_width=True)
        else: card("Primary analytical view","The dataset does not contain enough compatible fields for a headline trend. Use Breakdown and Data for deeper exploration.","📊",C["primary"])
    with b:
        if metric and top_dimension(df):
            dim=top_dimension(df); t=safe_group(df,dim,metric,8)
            fig=px.pie(t,names=dim,values=metric,hole=.58,title=f"{label(metric)} by {label(dim)}",color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a","#2d7d30","#f0a05a","#7a5230","#a8d5a9"])
            fig.update_traces(textfont_size=11); st.plotly_chart(chart_style(fig,380),use_container_width=True)
        elif cats:
            dim=top_dimension(df); counts=df[dim].value_counts().head(8).reset_index(); counts.columns=[dim,"Count"]
            fig=px.pie(counts,names=dim,values="Count",hole=.58,title=f"{label(dim)} Distribution")
            st.plotly_chart(chart_style(fig,380),use_container_width=True)
        else: card("Composition","No categorical dimension is available for a composition view.","◌",C["primary"])
    c1,c2,c3=st.columns([1,1.15,1])
    with c1:
        if metric and top_dimension(df):
            dim=top_dimension(df); t=safe_group(df,dim,metric,5).sort_values(metric)
            fig=px.bar(t,x=metric,y=dim,orientation="h",title=f"Top 5 {label(dim)}",text_auto=".2s")
            fig.update_traces(marker_color=C["accent"]); st.plotly_chart(chart_style(fig,280),use_container_width=True)
        elif nums:
            fig=px.histogram(df,x=nums[0],nbins=25,title=f"{label(nums[0])} Distribution")
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,280),use_container_width=True)
    with c2:
        if metric and len(cats)>=2:
            dim=cats[1]; t=safe_group(df,dim,metric,8)
            fig=px.bar(t,x=dim,y=metric,title=f"{label(metric)} by {label(dim)}",text_auto=".2s")
            fig.update_traces(marker_color=C["secondary"]); st.plotly_chart(chart_style(fig,280),use_container_width=True)
        elif len(nums)>=2:
            fig=px.scatter(df,x=nums[0],y=nums[1],title="Numeric Relationship")
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,280),use_container_width=True)
        else: card("Performance","More categorical or numeric dimensions are required.","📈",C["primary"])
    with c3:
        st.markdown("<div class='section-title'>💡 Key Insights</div>",unsafe_allow_html=True)
        insights=generate_natural_language_insights(df,profile,st.session_state.anomaly_result,st.session_state.forecast_result)
        if metric and top_dimension(df):
            dim=top_dimension(df); t=safe_group(df,dim,metric,1)
            if not t.empty:
                r=t.iloc[0]; insights.insert(0,f"{label(r[dim])} contributes the highest {label(metric)} in the current filtered dataset.")
        for ins in insights[:4]: st.markdown(f"<div class='insight'>◈ &nbsp; {ins}</div>",unsafe_allow_html=True)
    st.markdown("<div style='height:4px'></div>",unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🎯 Recommended Next Steps</div>",unsafe_allow_html=True)
    recs=st.session_state.recommendations or generate_recommendations(profile,df)
    rcols=st.columns(min(4,max(1,len(recs))))
    page_map={"Time-Series":"Forecasting","Forecasting":"Forecasting","Classification":"Classification","Regression":"Regression","Cluster":"Clustering","Relationship":"Relationships","Categorical":"Breakdown","Anomaly":"Anomaly Detection","Data-Quality":"Data Quality"}
    for i,rec in enumerate(recs[:4]):
        with rcols[i]:
            st.markdown(f"<div class='rec'><span class='rec-priority'>{rec.get('priority','')}</span><div class='rec-title'>{rec['title']}</div><div class='rec-desc'>{rec['description']}</div><div class='rec-value'>▲ {rec['business_value']}</div></div>",unsafe_allow_html=True)
            dest=next((v for k,v in page_map.items() if k.lower() in rec["title"].lower()),None)
            if dest:
                st.button(
                    "Explore →", key=f"ovrec_{i}", use_container_width=True,
                    on_click=set_page, args=(dest,),
                )

# ─────────────────────── Data ────────────────────────────────

elif page == "Data":
    st.markdown("<div class='page-kicker'>Dataset Workspace</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Data</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Explore, filter, and download the cleaned dataset from the active dashboard context.</div>",unsafe_allow_html=True)
    m1,m2,m3,m4=st.columns(4); mem_mb=df.memory_usage(deep=True).sum()/1024**2
    with m1: kpi_card("Total Rows",f"{len(df):,}","📊",C["primary"])
    with m2: kpi_card("Total Columns",f"{df.shape[1]:,}","🔢",C["accent"])
    with m3: kpi_card("Memory Usage",f"{mem_mb:.1f} MB","💾",C["secondary"])
    with m4: kpi_card("Numeric Columns",f"{len(nums):,}","🔢",C["primary"])
    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
    cats_for_filter=category_cols(df)
    if cats_for_filter:
        with st.expander("🔽  Active Filters",expanded=False):
            fcols=st.columns(min(4,len(cats_for_filter)))
            for i,cat in enumerate(cats_for_filter[:4]):
                with fcols[i]:
                    unique_vals=sorted(df[cat].dropna().astype(str).unique())
                    sel=st.multiselect(label(cat),unique_vals,default=st.session_state.filters.get(cat,[]),key=f"filter_{cat}")
                    st.session_state.filters[cat]=sel
    st.dataframe(df.head(500),use_container_width=True,height=500,hide_index=True)
    col_dl,col_info=st.columns([1,3])
    with col_dl:
        st.download_button("⬇️  Download Filtered CSV",df.to_csv(index=False).encode("utf-8"),file_name="retailiq_filtered.csv",mime="text/csv",type="primary")
    with col_info:
        st.markdown(f"<div class='insight' style='margin-top:0'>Dataset: <strong>{st.session_state.uploaded_name}</strong> &nbsp;·&nbsp; Raw rows: <strong>{len(st.session_state.raw_df):,}</strong> &nbsp;·&nbsp; Cleaned rows: <strong>{len(df):,}</strong> &nbsp;·&nbsp; Columns: <strong>{df.shape[1]}</strong></div>",unsafe_allow_html=True)

# ─────────────────────── Data Quality ────────────────────────

elif page == "Data Quality":
    st.markdown("<div class='page-kicker'>Trust & Reliability</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Data Quality</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Understand what changed during cleaning and what remains in the working dataset.</div>",unsafe_allow_html=True)
    missing=int(df.isna().sum().sum()); dup=int(df.duplicated().sum())
    q=max(0,100-(missing/max(df.size,1)*100))
    m1,m2,m3,m4=st.columns(4)
    with m1: kpi_card("Quality Score",f"{q:.1f}%","✅",C["accent"])
    with m2: kpi_card("Missing Cells",f"{missing:,}","⚠️",C["primary"])
    with m3: kpi_card("Duplicate Rows",f"{dup:,}","🔁",C["secondary"])
    with m4: kpi_card("Total Columns",f"{df.shape[1]:,}","🔢",C["primary"])
    st.markdown("<div style='height:12px'></div>",unsafe_allow_html=True)
    c1,c2=st.columns([1.3,1])
    with c1:
        card("Column Quality Report","Review missing values and data types for each field in the cleaned dataset.","✓",C["accent"])
        qt=profile.get("quality_table",pd.DataFrame())
        st.dataframe(qt if not qt.empty else pd.DataFrame({"Status":["No major quality issues detected"]}),use_container_width=True,hide_index=True)
        miss_series=df.isna().sum(); miss_series=miss_series[miss_series>0]
        if not miss_series.empty:
            miss_df=miss_series.reset_index(); miss_df.columns=["Column","Missing Count"]
            fig=px.bar(miss_df.sort_values("Missing Count",ascending=True),x="Missing Count",y="Column",orientation="h",title="Missing Values by Column",text_auto=True)
            fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,280),use_container_width=True)
        else: st.markdown("<div class='insight'><span style='color:var(--accent)'>✓</span> No missing values in the cleaned dataset.</div>",unsafe_allow_html=True)
    with c2:
        raw_rows=len(st.session_state.raw_df); clean_rows=len(st.session_state.cleaned_df); removed=raw_rows-clean_rows
        card("Cleaning Summary",f"Raw: <b>{raw_rows:,}</b> rows x {st.session_state.raw_df.shape[1]} columns<br>Cleaned: <b>{clean_rows:,}</b> rows x {st.session_state.cleaned_df.shape[1]} columns<br>Rows removed: <b>{removed:,}</b>","🧹",C["primary"])
        if st.session_state.cleaning_log: st.dataframe(pd.DataFrame(st.session_state.cleaning_log),use_container_width=True,hide_index=True)
        else: st.success("No cleaning actions were required — the dataset was already clean.")
        dtype_map={}
        for col in df.columns:
            dtype=str(df[col].dtype)
            if "int" in dtype or "float" in dtype: dtype_map[col]="Numeric"
            elif "datetime" in dtype: dtype_map[col]="Datetime"
            elif "bool" in dtype: dtype_map[col]="Boolean"
            else: dtype_map[col]="Categorical"
        dtype_counts=pd.Series(dtype_map).value_counts().reset_index(); dtype_counts.columns=["Type","Count"]
        fig=px.pie(dtype_counts,names="Type",values="Count",hole=0.55,title="Column Types",color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a"])
        st.plotly_chart(chart_style(fig,280),use_container_width=True)

# ─────────────────────── Breakdown ───────────────────────────

elif page == "Breakdown":
    st.markdown("<div class='page-kicker'>Comparative Analysis</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Breakdown</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Compare measures across meaningful dimensions. RetailIQ recommends a chart type, but you stay in control.</div>",unsafe_allow_html=True)
    if not cats or not nums: st.info("A categorical dimension and numeric measure are required for breakdown analysis.")
    else:
        ctrl1,ctrl2,ctrl3=st.columns(3)
        dim=ctrl1.selectbox("Dimension",cats,index=0,key="bd_dim")
        met=ctrl2.selectbox("Measure",nums,index=nums.index(metric) if metric in nums else 0,key="bd_met")
        unique=int(df[dim].nunique(dropna=True))
        recommended="Donut Chart" if unique<=6 else "Bar Chart"
        options=["⭐ Recommended: "+recommended,"Bar Chart","Horizontal Bar Chart","Line Chart","Area Chart","Pie Chart","Donut Chart"]
        choice=ctrl3.selectbox("Chart Type",options,key="bd_graph")
        graph=recommended if choice.startswith("⭐") else choice
        data=safe_group(df,dim,met,15); title_txt=f"{label(met)} by {label(dim)}"
        if graph=="Bar Chart":
            fig=px.bar(data.sort_values(met,ascending=False),x=dim,y=met,title=title_txt,text_auto=".2s"); fig.update_traces(marker_color=C["primary"])
        elif graph=="Horizontal Bar Chart":
            fig=px.bar(data.sort_values(met),x=met,y=dim,orientation="h",title=title_txt,text_auto=".2s"); fig.update_traces(marker_color=C["primary"])
        elif graph=="Line Chart":
            fig=px.line(data,x=dim,y=met,markers=True,title=title_txt); fig.update_traces(line=dict(color=C["primary"],width=2.5))
        elif graph=="Area Chart":
            fig=px.area(data,x=dim,y=met,title=title_txt); fig.update_traces(line=dict(color=C["primary"]),fillcolor="rgba(218,111,11,0.15)")
        else:
            fig=px.pie(data,names=dim,values=met,hole=.55 if graph=="Donut Chart" else 0,title=title_txt,color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a","#2d7d30","#f0a05a","#7a5230","#a8d5a9"])
        st.plotly_chart(chart_style(fig,440),use_container_width=True)
        col_tbl,col_ins=st.columns([2,1])
        with col_tbl: st.dataframe(data,use_container_width=True,hide_index=True)
        with col_ins:
            if not data.empty:
                top=data.iloc[0]; bot=data.iloc[-1]; avg=data[met].mean()
                st.markdown(f"<div class='insight'><strong>Top performer:</strong><br>{label(top[dim])} — {fmt(top[met])}</div>",unsafe_allow_html=True)
                st.markdown(f"<div class='insight'><strong>Bottom performer:</strong><br>{label(bot[dim])} — {fmt(bot[met])}</div>",unsafe_allow_html=True)
                st.markdown(f"<div class='insight'><strong>Average {label(met)}:</strong><br>{fmt(avg)}</div>",unsafe_allow_html=True)

# ─────────────────────── Relationships ───────────────────────

elif page == "Relationships":
    st.markdown("<div class='page-kicker'>Exploration</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Relationships</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Explore associations, distributions, and numeric relationships between variables in your dataset.</div>",unsafe_allow_html=True)
    if len(nums)>=2:
        tab_scatter,tab_corr=st.tabs(["Scatter Plot","Correlation Matrix"])
        with tab_scatter:
            a,b,c=st.columns(3)
            x=a.selectbox("X-axis",nums,index=0); y=b.selectbox("Y-axis",nums,index=1)
            color=c.selectbox("Color by",["None"]+cats)
            data=df[[x,y]+([] if color=="None" else [color])].dropna()
            fig=px.scatter(data,x=x,y=y,color=None if color=="None" else color,title=f"{label(x)} vs {label(y)}",color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a"])
            if color=="None": fig.update_traces(marker=dict(color=C["primary"],size=5,opacity=0.65))
            st.plotly_chart(chart_style(fig,500),use_container_width=True)
            st.dataframe(data.head(500),use_container_width=True,hide_index=True)
        with tab_corr:
            corr=df[nums].corr().round(3)
            fig=px.imshow(corr,text_auto=True,aspect="auto",title="Correlation Heatmap",color_continuous_scale=RETAILIQ_COLORSCALE)
            st.plotly_chart(chart_style(fig,480),use_container_width=True)
    elif metric:
        fig=px.histogram(df,x=metric,nbins=30,title=f"Distribution of {label(metric)}")
        fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,430),use_container_width=True)
    else: st.info("Not enough numeric measures for relationship analysis.")

# ─────────────── Classification / Regression ──────────────────

elif page in ["Classification","Regression"]:
    task=page.lower(); task_icon="🎯" if page=="Classification" else "📈"
    st.markdown(f"<div class='page-kicker'>ML Analysis · {page}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{task_icon} {page}</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Run the existing RetailIQ supervised-learning engine and explore its results through a dedicated analytics workspace.</div>",unsafe_allow_html=True)
    target_options=list(df.columns); auto=detect_target(df); default=target_options.index(auto) if auto in target_options else 0
    ctrl1,ctrl2=st.columns([2,1])
    with ctrl1: target=st.selectbox("Target column",target_options,index=default,key=f"{task}_target")
    with ctrl2:
        inferred=infer_task(df[target]); st.markdown(f"<div style='margin-top:26px'><span class='badge badge-info'>Detected: {inferred}</span></div>",unsafe_allow_html=True)
    if st.button(f"🚀  Run {page} Analysis",type="primary",key=f"run_{task}"):
        with st.spinner(f"Running {page.lower()} models..."):
            res=run_supervised(df,target,task)
        if res.get("ok"): st.session_state.ml_results[page]=res; st.success(f"{page} analysis completed successfully.")
        else: st.error(res.get("error","Analysis failed."))
    res=st.session_state.ml_results.get(page)
    if res:
        results=res["results"].copy()
        card("Model Benchmark Results",f"Target: <b>{label(target)}</b> &nbsp;·&nbsp; Validation: {res.get('validation_strategy','holdout')}","🏆",C["primary"])
        st.dataframe(results,use_container_width=True,hide_index=True)
        metric_name="F1" if task=="classification" else "RMSE"
        if metric_name in results.columns:
            vals=pd.to_numeric(results[metric_name],errors="coerce")
            if vals.notna().any():
                fig=px.bar(results[results[metric_name].notna()].sort_values(metric_name,ascending=(task!="classification")),x="Model",y=metric_name,title=f"{metric_name} by Model",text_auto=".3f")
                fig.update_traces(marker_color=C["primary"]); st.plotly_chart(chart_style(fig,320),use_container_width=True)
        imps=res.get("importances",{})
        if imps:
            model_name=list(imps.keys())[0]; imp=imps[model_name]
            if isinstance(imp,pd.DataFrame) and not imp.empty:
                fig=px.bar(imp.head(12).sort_values("Importance"),x="Importance",y="Feature",orientation="h",title=f"Feature Importance — {model_name}",text_auto=".3f")
                fig.update_traces(marker_color=C["accent"]); st.plotly_chart(chart_style(fig,380),use_container_width=True)
        st.markdown("---")
        if st.button(f"🚀  Run Advanced Gradient Boosting",key=f"adv_{task}"):
            with st.spinner("Evaluating Gradient Boosting..."):
                adv=run_advanced_predictive(df,target,task)
            if adv.get("ok"): st.session_state.advanced_result=adv; st.success("Advanced model evaluation completed.")
            else: st.error(adv.get("error","Advanced model failed."))
        adv=st.session_state.advanced_result
        if adv and adv.get("target")==target and adv.get("task")==task:
            st.markdown("### Advanced Predictive Model")
            st.metric("Primary metric",fmt(adv["primary_metric"]))
            st.dataframe(pd.DataFrame([adv["metrics"]]),use_container_width=True,hide_index=True)

# ─────────────────────── Clustering ──────────────────────────

elif page == "Clustering":
    st.markdown("<div class='page-kicker'>ML Analysis · Clustering</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>🫧 Clustering</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Discover latent groups within the dataset without requiring a target label. Adjust the number of clusters and explore cluster profiles.</div>",unsafe_allow_html=True)
    if len(nums)<2: st.info("At least two varying numeric features are required for clustering.")
    else:
        ctrl1,ctrl2=st.columns([1,2])
        with ctrl1: k=st.slider("Number of clusters",2,min(8,max(3,len(df)-1)),3)
        with ctrl2:
            st.markdown("<div style='margin-top:28px'></div>",unsafe_allow_html=True)
            run_clust=st.button("🚀  Run Clustering",type="primary")
        if run_clust:
            with st.spinner("Computing clusters using K-Means..."):
                st.session_state.cluster_result=run_clustering(df,n_clusters=k)
            if st.session_state.cluster_result and st.session_state.cluster_result.get("ok"): st.success(f"Clustering completed — {k} clusters identified.")
        res=st.session_state.cluster_result
        if res and res.get("ok"):
            m1,m2,m3=st.columns(3)
            with m1: kpi_card("Silhouette Score",f"{res['silhouette']:.3f}","📏",C["primary"])
            with m2: kpi_card("Clusters",str(k),"🫧",C["accent"])
            with m3: kpi_card("Samples",f"{len(res['assignments']):,}","🗃️",C["secondary"])
            assign=res["assignments"]
            if len(res["features"])>=2:
                fig=px.scatter(assign,x=res["features"][0],y=res["features"][1],color="Cluster",title="Cluster Map — Primary Two Features",hover_data=res["features"][:6],color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a","#2d7d30","#f0a05a","#7a5230","#a8d5a9"])
                fig.update_traces(marker_size=6,marker_opacity=0.75); st.plotly_chart(chart_style(fig,460),use_container_width=True)
            cluster_counts=assign["Cluster"].value_counts().reset_index(); cluster_counts.columns=["Cluster","Count"]
            fig2=px.pie(cluster_counts,names="Cluster",values="Count",hole=0.52,title="Cluster Size Distribution",color_discrete_sequence=[C["primary"],C["accent"],C["secondary"],"#e8935a","#2d7d30","#f0a05a","#7a5230"])
            st.plotly_chart(chart_style(fig2,320),use_container_width=True)
            st.dataframe(assign.head(500),use_container_width=True,hide_index=True)

# ─────────────────────── Forecasting ─────────────────────────

elif page == "Forecasting":
    st.markdown("<div class='page-kicker'>ML Analysis · Forecasting</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>🔮 Forecasting</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Use detected time structure to estimate the next periods without overstating certainty. Historical and forecast values are clearly distinguished.</div>",unsafe_allow_html=True)
    if not dates: st.info("No date-like field was detected in your dataset. Forecasting requires a column with dates or timestamps.")
    elif not nums: st.info("No numeric forecasting measure was detected.")
    else:
        ctrl1,ctrl2,ctrl3=st.columns(3)
        dc=ctrl1.selectbox("Date column",dates)
        target=ctrl2.selectbox("Forecast measure",nums,index=nums.index(metric) if metric in nums else 0)
        horizon=ctrl3.slider("Forecast horizon (periods)",3,30,7)
        if st.button("🚀  Run Forecast",type="primary"):
            with st.spinner("Evaluating forecasting models..."):
                st.session_state.forecast_result=run_forecast(df,dc,target,horizon)
            if st.session_state.forecast_result and st.session_state.forecast_result.get("ok"):
                st.success(f"Forecast generated — {horizon} periods ahead using {st.session_state.forecast_result['best_model']}.")
        res=st.session_state.forecast_result
        if res and res.get("ok"):
            st.metric("Selected model",res["best_model"])
            hist=res["aggregated"][["date","value"]].rename(columns={"value":target})
            fc=res["forecast"].rename(columns={"forecast":target})
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"],y=hist[target],name="Historical",mode="lines",line=dict(color=C["secondary"],width=2.5)))
            fig.add_trace(go.Scatter(x=fc["date"],y=fc[target],name="Forecast",mode="lines+markers",line=dict(color=C["primary"],width=2.5,dash="dash"),marker=dict(color=C["primary"],size=8)))
            fig.update_layout(title=f"{label(target)} — Historical & Forecast",xaxis_title="",yaxis_title=label(target))
            st.plotly_chart(chart_style(fig,470),use_container_width=True)
            eval_tab,fc_tab=st.tabs(["Model Evaluation","Forecast Values"])
            with eval_tab: st.dataframe(res["evaluation"],use_container_width=True,hide_index=True)
            with fc_tab: st.dataframe(res["forecast"],use_container_width=True,hide_index=True)

# ─────────────────────── Anomaly Detection ───────────────────

elif page == "Anomaly Detection":
    st.markdown("<div class='page-kicker'>ML Analysis · Anomaly Detection</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>🚨 Anomaly Detection</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Identify unusual observations using Isolation Forest. Adjust the expected anomaly fraction and compare with reference labels if available.</div>",unsafe_allow_html=True)
    ctrl1,ctrl2=st.columns([1,2])
    with ctrl1: contamination=st.slider("Expected anomaly fraction",0.01,0.20,0.05,0.01)
    with ctrl2:
        st.markdown("<div style='margin-top:28px'></div>",unsafe_allow_html=True)
        run_anom=st.button("🚀  Detect Anomalies",type="primary")
    if run_anom:
        with st.spinner("Running Isolation Forest..."):
            st.session_state.anomaly_result=run_anomaly_detection(df,contamination=contamination)
        if st.session_state.anomaly_result and st.session_state.anomaly_result.get("ok"): st.success("Anomaly detection completed.")
    res=st.session_state.anomaly_result
    if res and res.get("ok"):
        m1,m2,m3=st.columns(3)
        with m1: kpi_card("Flagged Anomalies",f"{res['anomalies']:,}","🚨",C["destructive"])
        with m2: kpi_card("Contamination",f"{res['contamination']:.0%}","⚠️",C["primary"])
        with m3: kpi_card("Features Used",f"{len(res['features']):,}","🔢",C["secondary"])
        if res.get("ground_truth_available"):
            gm=res.get("ground_truth_metrics",{}); st.success(f"Reference label: {res.get('ground_truth_column','reference')} · F1 {gm.get('F1',0):.3f}")
        data=res["data"]
        if len(res["features"])>=2:
            fig=px.scatter(data,x=res["features"][0],y=res["features"][1],color="ML_Anomaly",title="Normal vs Flagged Observations",hover_data=["Anomaly_Score"],color_discrete_map={"Normal":C["accent"],"Anomaly":C["destructive"]})
            fig.update_traces(marker_size=5,marker_opacity=0.70); st.plotly_chart(chart_style(fig,460),use_container_width=True)
        if "Anomaly_Score" in data.columns:
            fig2=px.histogram(data,x="Anomaly_Score",color="ML_Anomaly",nbins=40,title="Anomaly Score Distribution",barmode="overlay",color_discrete_map={"Normal":C["accent"],"Anomaly":C["destructive"]})
            st.plotly_chart(chart_style(fig2,300),use_container_width=True)
        st.dataframe(data.sort_values("Anomaly_Score").head(100),use_container_width=True,hide_index=True)

# ─────────────────────── Recommendations ─────────────────────

elif page == "Recommendations":
    st.markdown("<div class='page-kicker'>Decision Support</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>💡 Recommendations</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Evidence-linked next steps generated from the current dataset and all completed analyses.</div>",unsafe_allow_html=True)
    recs=generate_recommendations(profile,df)
    extra=ml_guided_recommendations(st.session_state.advanced_result or {},st.session_state.anomaly_result,st.session_state.forecast_result)
    all_recs=recs+extra; seen=set(); valid_recs=[]
    for r in all_recs:
        title_key=r.get("title") or r.get("Recommendation","Recommendation")
        if title_key in seen: continue
        seen.add(title_key); valid_recs.append(r)
    if not valid_recs: st.info("No recommendations available. Run ML analyses first to generate data-driven recommendations.")
    else:
        for i in range(0,len(valid_recs),2):
            cols=st.columns(2)
            for j,col in enumerate(cols):
                if i+j<len(valid_recs):
                    r=valid_recs[i+j]
                    title_text=r.get("title") or r.get("Recommendation","Recommendation")
                    description=r.get("description") or r.get("Recommendation","")
                    why=r.get("why") or r.get("Evidence","")
                    business=r.get("business_value") or r.get("Evidence","")
                    priority=r.get("priority","")
                    with col:
                        st.markdown(f"<div class='rec'><span class='rec-priority'>{priority}</span><div class='rec-title'>{title_text}</div><div class='rec-desc'>{description}</div><div class='small' style='margin-top:8px'>Why: {why}</div><div class='rec-value'>▲ Business value: {business}</div></div>",unsafe_allow_html=True)

# ─────────────────────── Research Evaluation ─────────────────

elif page == "Research Evaluation":
    st.markdown("<div class='page-kicker'>Advanced</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-title'>🔬 Research Evaluation</div>",unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Reproducible evaluation artifacts are available here without running expensive benchmarks automatically.</div>",unsafe_allow_html=True)
    out=Path("results/research_suite")
    col_run,col_status=st.columns([1,2])
    with col_run: run_research=st.button("🚀  Run Research Evaluation",type="primary")
    with col_status:
        if out.exists():
            files=list(out.glob("*.csv")); st.markdown(f"<div style='margin-top:10px'><span class='badge badge-success'>✓ {len(files)} artifact(s) available</span></div>",unsafe_allow_html=True)
    if run_research:
        try:
            from src.research_suite import run_suite
            with st.spinner("Running research suite — this may take several minutes..."):
                result=run_suite(root="research_datasets",results_dir="results/research_suite",repeats=5,folds=5)
            st.session_state.research_result=result; st.success("Research evaluation completed successfully.")
        except Exception as exc: st.error(f"Research evaluation could not be completed: {exc}")
    if hasattr(st.session_state,"research_result"):
        st.dataframe(st.session_state.research_result.get("run_summary",pd.DataFrame()).head(50),use_container_width=True,hide_index=True)
    if out.exists():
        files=list(out.glob("*.csv"))
        if files:
            st.markdown("### Saved Research Artifacts")
            for f in files[:8]:
                with st.expander(f"📄 {f.name}",expanded=False):
                    try: st.dataframe(pd.read_csv(f).head(50),use_container_width=True,hide_index=True)
                    except Exception: st.warning(f"Could not read {f.name}")
    else:
        card("No Research Artifacts","No saved research artifacts are available in this working copy. Use the Run Research Evaluation button to generate them. The evaluation uses datasets from the <code>research_datasets/</code> directory.","🔬",C["primary"])

# ─────────────────────── Footer ──────────────────────────────

st.markdown(
    "<div class='footer'><span>◈ &nbsp; RetailIQ v1.0 &nbsp; · &nbsp; Built for smarter, data-driven decisions</span>"
    "<span>Adaptive Analytics &nbsp; · &nbsp; All ML calculations performed by existing backend engines</span></div>",
    unsafe_allow_html=True,
)
