import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "ml/reports"

st.title("SHAP Explainability")

summary = REPORTS_DIR / "shap_summary.png"
dependence = REPORTS_DIR / "shap_dependence.png"

if summary.exists():
    st.image(
        str(summary),
        caption="SHAP Summary Plot"
    )

if dependence.exists():
    st.image(
        str(dependence),
        caption="SHAP Dependence Plot"
    )