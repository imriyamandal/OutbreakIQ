import streamlit as st
import pandas as pd
import plotly.express as px
from app.utils.api_client import APIClient, API_URL

def show_model_insights():
    st.markdown('<div class="page-title">Model Insights & Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Evaluation metrics, feature importance, and SHAP explanations from trained models</div>', unsafe_allow_html=True)

    # 1. Fetch Metrics & feature importance from API
    with st.spinner("Fetching model evaluation details..."):
        metrics_data = APIClient.get("/model/metrics")
        importance_data = APIClient.get("/model/feature-importance")

    # 2. Load Metrics
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)
    
    col_cls, col_reg = st.columns(2)
    
    with col_cls:
        st.markdown("**Outbreak Classifier Performance**")
        if metrics_data and "classification" in metrics_data and metrics_data["classification"]:
            df_cls = pd.DataFrame(metrics_data["classification"])
            st.dataframe(df_cls, use_container_width=True, hide_index=True)
        else:
            st.info("Classifier metrics not found on backend.")
            
    with col_reg:
        st.markdown("**Case Regressor Performance**")
        if metrics_data and "regression" in metrics_data and metrics_data["regression"]:
            df_reg = pd.DataFrame(metrics_data["regression"])
            st.dataframe(df_reg, use_container_width=True, hide_index=True)
        else:
            st.info("Regressor metrics not found on backend.")
            
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Feature Importance
    col_fi1, col_fi2 = st.columns(2)
    
    with col_fi1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Classifier Feature Importance</div>', unsafe_allow_html=True)
        if importance_data and "classification" in importance_data and importance_data["classification"]:
            df_cls_fi = pd.DataFrame(importance_data["classification"]).head(10)
            fig = px.bar(
                df_cls_fi,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Teal"
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Classifier feature importance not found on backend.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_fi2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Regressor Feature Importance</div>', unsafe_allow_html=True)
        if importance_data and "regression" in importance_data and importance_data["regression"]:
            df_reg_fi = pd.DataFrame(importance_data["regression"]).head(10)
            fig = px.bar(
                df_reg_fi,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Tealgrn"
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Regressor feature importance not found on backend.")
        st.markdown('</div>', unsafe_allow_html=True)

    # 4. SHAP Plots from backend
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SHAP Summary Plot</div>', unsafe_allow_html=True)
    shap_url = f"{API_URL}/model/plots/shap_summary"
    st.image(shap_url, caption="SHAP Summary Plot representing global feature impacts", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">SHAP Dependence Plot</div>', unsafe_allow_html=True)
    shap_dep_url = f"{API_URL}/model/plots/shap_dependence"
    st.image(shap_dep_url, caption="SHAP Dependence Plot for the top environmental feature", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
