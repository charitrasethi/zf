import streamlit as st

st.set_page_config(
    page_title="Zaria Fashion Analytics",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background:#f4f6f8; }
.kpi-card  { background:#fff; border:1px solid #dee2e6; border-radius:10px;
             padding:18px 14px; text-align:center; }
.kpi-val   { font-size:30px; font-weight:700; }
.kpi-lbl   { font-size:12px; color:#6c757d; margin-top:4px; }
.sec       { font-size:16px; font-weight:600; border-left:4px solid #1D9E75;
             padding-left:10px; margin:22px 0 10px; }
.ibox      { background:#e8f5f0; border-left:4px solid #1D9E75;
             padding:10px 14px; border-radius:6px; font-size:13px; margin:6px 0; }
.wbox      { background:#fff8e1; border-left:4px solid #ffc107;
             padding:10px 14px; border-radius:6px; font-size:13px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = {
    "🏠  Overview":              "pg_overview",
    "📊  Descriptive":           "pg_descriptive",
    "🔍  Diagnostic":            "pg_diagnostic",
    "👥  Clustering":            "pg_clustering",
    "🔗  Association Rules":     "pg_arm",
    "🎯  Classification":        "pg_classification",
    "💰  Regression & CLV":      "pg_regression",
    "🆕  New Customer Predictor":"pg_predictor",
}

st.sidebar.markdown("## 👗 Zaria Fashion")
st.sidebar.markdown("*Data-Driven Intelligence Platform*")
st.sidebar.markdown("---")
choice = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Dataset · 1,200 respondents · 25 survey columns · Pan-India")
st.sidebar.caption("Algorithms · K-Means · Apriori · RF · XGBoost · Ridge")

import importlib
importlib.import_module(PAGES[choice]).render()
