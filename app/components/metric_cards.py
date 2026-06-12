import streamlit as st

def render_metrics():

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Total Cases",
        "12,548",
        "+12%"
    )

    c2.metric(
        "Active Alerts",
        "18",
        "+3"
    )

    c3.metric(
        "High Risk States",
        "8"
    )

    c4.metric(
        "Model Accuracy",
        "82%"
    )