import streamlit as st

def render_alert_panel():

    st.error(
        "High Dengue Risk detected in Maharashtra"
    )

    st.warning(
        "Moderate Malaria Risk in West Bengal"
    )

    st.success(
        "No major alerts in Tamil Nadu"
    )