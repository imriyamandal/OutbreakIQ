import streamlit as st
import pandas as pd
import plotly.express as px
from utils.api_client import APIClient
from app.components.heatmap import render_heatmap

def show_overview():
    st.markdown('<div class="page-title">Surveillance Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI-powered epidemiology and early warning intelligence dashboard</div>', unsafe_allow_html=True)

    # Fetch stats from API
    with st.spinner("Fetching stats from server..."):
        stats = APIClient.get("/dashboard/stats")
        
    if not stats:
        st.warning("Unable to fetch stats from backend server.")
        return

    # Render metric cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
            <div class="metric-card metric-teal">
                <div class="metric-label">Total Cases</div>
                <div class="metric-value">{stats['total_cases']:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="metric-card metric-orange">
                <div class="metric-label">Total Outbreaks</div>
                <div class="metric-value">{stats['active_outbreaks']:,}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
            <div class="metric-card metric-red">
                <div class="metric-label">High-Risk States</div>
                <div class="metric-value">{stats['high_risk_states']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
            <div class="metric-card metric-green">
                <div class="metric-label">Top Disease</div>
                <div class="metric-value" style="font-size: 18px; line-height: 2.2; font-weight: 700;">{stats['most_affected_disease']}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Trends and Risk Distribution
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Monthly Disease Cases Trend</div>', unsafe_allow_html=True)
        
        trends_df = pd.DataFrame(stats["monthly_trends"])
        if not trends_df.empty:
            fig = px.line(
                trends_df,
                x="month",
                y="cases",
                markers=True,
                color_discrete_sequence=["#0f766e"]
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Month",
                yaxis_title="Total Cases",
                xaxis=dict(showgrid=True, gridcolor="#e2e8f0"),
                yaxis=dict(showgrid=True, gridcolor="#e2e8f0")
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monthly trends data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Level Distribution</div>', unsafe_allow_html=True)
        
        dist = stats["risk_distribution"]
        dist_df = pd.DataFrame([
            {"Risk Level": k, "Records": v} for k, v in dist.items()
        ])
        
        fig_pie = px.pie(
            dist_df,
            names="Risk Level",
            values="Records",
            color="Risk Level",
            color_discrete_map={
                "Low": "#dcfce7",
                "Medium": "#fef3c7",
                "High": "#f59e0b",
                "Critical": "#fee2e2"
            },
            hole=0.4
        )
        fig_pie.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # India heatmap
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geospatial Outbreak Heatmap</div>', unsafe_allow_html=True)
    with st.spinner("Loading geospatial visualization..."):
        heatmap_res = APIClient.get("/heatmap")
        
    if heatmap_res and "points" in heatmap_res:
        points_df = pd.DataFrame(heatmap_res["points"])
        if not points_df.empty:
            render_heatmap(points_df)
        else:
            st.info("No geographic data available to plot.")
    else:
        st.warning("Failed to load map data.")
    st.markdown('</div>', unsafe_allow_html=True)
