import streamlit as st
from pathlib import Path

def render_sidebar():

    logo = Path("app/assets/logo.jpg")

    st.sidebar.image(
        str(logo),
        use_container_width=True
    )

    st.sidebar.markdown(
        """
        ## OutbreakIQ
        Public Health Intelligence
        """
    )
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Prediction",
            "Analytics",
            "Model Insights",
            "History",
            "Heatmap",
            "Alerts"
        ], label_visibility="collapsed"
    )

    st.sidebar.divider()

    st.sidebar.info(
        "AI-Powered Disease Outbreak Prediction"
    )

    return page