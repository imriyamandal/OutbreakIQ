import streamlit as st

st.set_page_config(
    page_title="OutbreakIQ",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 OutbreakIQ")

st.markdown(
"""
AI-Powered Disease Outbreak Forecasting Platform
"""
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Forecast Model",
    "XGBoost"
)

col2.metric(
    "Backend",
    "FastAPI"
)

col3.metric(
    "Status",
    "Live"
)

st.success("Dashboard running successfully.")