import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.utils.api_client import APIClient

def show_analytics():
    st.markdown('<div class="page-title">Epidemiological Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Historical patterns, disease distribution trends, and seasonal analysis</div>', unsafe_allow_html=True)

    with st.spinner("Fetching analytics report..."):
        data = APIClient.get("/analytics")

    if not data:
        st.warning("Unable to retrieve analytics from backend.")
        return

    # Disease Trends & State Comparison
    c_col1, c_col2 = st.columns(2)
    
    with c_col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Case Distribution by Disease</div>', unsafe_allow_html=True)
        df_dis = pd.DataFrame(data["disease_trends"])
        if not df_dis.empty:
            fig = px.bar(
                df_dis,
                x="disease",
                y="cases",
                color="disease",
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Disease",
                yaxis_title="Total Cases",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Top Affected States/UTs</div>', unsafe_allow_html=True)
        df_state = pd.DataFrame(data["state_trends"])
        if not df_state.empty:
            fig = px.bar(
                df_state,
                x="cases",
                y="state",
                orientation="h",
                color="cases",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Total Cases",
                yaxis_title="State/UT",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Seasonal Analysis & Correlation Matrix
    c_col3, c_col4 = st.columns([1, 1])
    
    with c_col3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Seasonal Variation (Average Cases)</div>', unsafe_allow_html=True)
        df_seas = pd.DataFrame(data["seasonal_analysis"])
        if not df_seas.empty:
            fig = px.line(
                df_seas,
                x="month",
                y="average_cases",
                markers=True,
                color_discrete_sequence=["#ef4444"]
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Month",
                yaxis_title="Average Case Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Climate-Outbreak Correlation Analysis</div>', unsafe_allow_html=True)
        corr = data["correlation"]
        if corr:
            fig = go.Figure(
                data=go.Heatmap(
                    z=corr["z"],
                    x=[x.title() for x in corr["x"]],
                    y=[y.title() for y in corr["y"]],
                    colorscale="RdBu",
                    zmin=-1.0,
                    zmax=1.0,
                    text=corr["z"],
                    texttemplate="%{text}",
                )
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Outbreak Frequency
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Outbreak Frequency by Disease Type</div>', unsafe_allow_html=True)
    df_freq = pd.DataFrame(data["outbreak_frequency"])
    if not df_freq.empty:
        fig = px.bar(
            df_freq,
            x="disease",
            y="outbreaks",
            color="outbreaks",
            color_continuous_scale="Purples",
            labels={"outbreaks": "Total Outbreak Incidents"}
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Disease",
            yaxis_title="Total Outbreak Flag Counts",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)