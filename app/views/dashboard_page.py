import streamlit as st

from components.metric_cards import render_metrics
from components.forecast_chart import render_forecast_chart
from components.risk_table import render_risk_table
from components.alert_panel import render_alert_panel

def show_dashboard():

    st.title(
        "Disease Outbreak Prediction & Early Warning System"
    )

    st.caption(
        "AI Powered Public Health Intelligence Platform"
    )

    render_metrics()

    st.divider()

    st.subheader(
    "India Outbreak Risk Map"
    )

    import folium
    from streamlit_folium import st_folium

    def render_heatmap():

        india_map = folium.Map(
            location=[22.9734, 78.6569],
            zoom_start=5,
            tiles="CartoDB positron"
        )

        st_folium(
            india_map,
            use_container_width=True,
            height=600
        )
    from components.heatmap import render_heatmap

    st.subheader(
        "India Outbreak Risk Map"
    )

    render_heatmap()
    st.divider()

    st.subheader(
        "6 Month Forecast"
    )

    render_forecast_chart()

    st.divider()

    st.subheader(
        "Top Risk States"
    )

    render_risk_table()

    st.divider()

    st.subheader(
        "Recent Alerts"
    )

    render_alert_panel()