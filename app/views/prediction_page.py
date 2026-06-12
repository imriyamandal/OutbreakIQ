import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from app.utils.api_client import APIClient

@st.cache_data
def get_metadata():
    return APIClient.get("/metadata")

def show_prediction():
    st.markdown('<div class="page-title">Disease Outbreak Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Input location and environmental factors to estimate outbreak risks</div>', unsafe_allow_html=True)

    metadata = get_metadata()
    if not metadata:
        st.error("Failed to load location/disease metadata from backend API.")
        return

    states = metadata.get("states", [])
    diseases = metadata.get("diseases", [])
    state_districts = metadata.get("state_districts", {})

    # Sidebar parameters or two-column form
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox(
            "Select State/UT",
            options=states
        )
        
        # Filter districts based on state
        districts = state_districts.get(state, [])
        district = st.selectbox(
            "Select District",
            options=districts
        )

    with col2:
        disease = st.selectbox(
            "Select Disease",
            options=diseases
        )
        
        month_name = st.selectbox(
            "Select Month",
            options=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        )
        month = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"].index(month_name) + 1

    with col3:
        year = st.number_input(
            "Select Year",
            min_value=2010,
            max_value=2050,
            value=2026
        )
        
        day = st.number_input(
            "Select Day of Month",
            min_value=1,
            max_value=31,
            value=1
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Environmental Variables</div>', unsafe_allow_html=True)

    col_env1, col_env2, col_env3 = st.columns(3)

    with col_env1:
        temp = st.slider(
            "Temperature (°C)",
            min_value=10.0,
            max_value=45.0,
            value=28.0,
            step=0.5
        )
        
    with col_env2:
        # Precipitation in the dataset is a small decimal. Let's make it intuitive (mm)
        # Precipitation in dataset is in [0, 5]. Let's scale or keep it matching typical values
        precipitation = st.slider(
            "Precipitation (mm/day)",
            min_value=0.0,
            max_value=150.0,
            value=25.0,
            step=1.0
        )
        # Dataset precipitation is typically in range [0, 3]. Let's scale it down for model, say divide by 50
        model_precipitation = precipitation / 50.0

    with col_env3:
        # Leaf Area Index (LAI)
        lai = st.slider(
            "Leaf Area Index (LAI)",
            min_value=0.0,
            max_value=60.0,
            value=15.0,
            step=0.5
        )

    # Let's map temp to kelvin if model expects it
    # In training data, temp is around 300K. So let's convert Celsius to Kelvin: K = C + 273.15
    model_temp = temp + 273.15

    predict_clicked = st.button("Run Prediction Models")
    st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        payload = {
            "state_ut": state,
            "district": district,
            "disease": disease,
            "month": month,
            "year": year,
            "day": day,
            "temp": model_temp,
            "precipitation": model_precipitation,
            "lai": lai
        }

        with st.spinner("Processing machine learning model pipeline..."):
            result = APIClient.post("/predict", payload)

        if result:
            st.markdown('<div class="forecast-result">', unsafe_allow_html=True)
            st.markdown('<div class="forecast-title">🔮 MODEL PREDICTION COMPLETED</div>', unsafe_allow_html=True)
            
            c_res1, c_res2, c_res3, c_res4 = st.columns(4)
            
            with c_res1:
                st.metric("Predicted Cases", f"{result['predicted_cases']}")
                
            with c_res2:
                st.metric("Outbreak Probability", f"{result['outbreak_probability'] * 100:.2f}%")
                
            with c_res3:
                risk = result["risk_level"]
                risk_style = "high-risk" if risk in ["High", "Critical"] else ("medium-risk" if risk == "Medium" else "low-risk")
                st.markdown(f"**Risk Level** <br> <span class='{risk_style}'>{risk}</span>", unsafe_allow_html=True)
                
            with c_res4:
                st.metric("Confidence Score", f"{result['confidence_score'] * 100:.2f}%")

            st.markdown('</div>', unsafe_allow_html=True)

            # Alerts generated if any
            if "alert" in result:
                st.markdown("<br>", unsafe_allow_html=True)
                alert = result["alert"]
                alert_level = alert["alert_level"]
                if alert_level == "CRITICAL":
                    st.error(f"🚨 **{alert_level} ALERT**: {alert['message']}")
                elif alert_level == "HIGH":
                    st.warning(f"⚠️ **{alert_level} ALERT**: {alert['message']}")
                elif alert_level == "MODERATE":
                    st.info(f"ℹ️ **{alert_level} ALERT**: {alert['message']}")
                else:
                    st.success(f"✅ **{alert_level} ALERT**: {alert['message']}")

            # SHAP Top Contributors
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart, col_explain = st.columns([2, 1])
            
            with col_chart:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">SHAP Feature Contributions</div>', unsafe_allow_html=True)
                
                shap_list = result.get("shap_top_contributors", [])
                if shap_list:
                    shap_df = pd.DataFrame(shap_list)
                    # Create nice bar chart
                    fig = px.bar(
                        shap_df,
                        x="shap_value",
                        y="feature",
                        orientation="h",
                        color="shap_value",
                        color_continuous_scale="RdBu_r",
                        labels={"shap_value": "SHAP Contribution Value", "feature": "Feature"}
                    )
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=10, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No feature contribution calculations available.")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_explain:
                st.markdown('<div class="section-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Key Drivers</div>', unsafe_allow_html=True)
                if shap_list:
                    for item in shap_list[:5]:
                        direction = "increases" if item["shap_value"] > 0 else "decreases"
                        icon = "📈" if item["shap_value"] > 0 else "📉"
                        feat_display = item["feature"].replace("_enc", "").replace("_", " ").title()
                        st.markdown(f"{icon} **{feat_display}** {direction} the probability of outbreak.")
                else:
                    st.info("No explanations found.")
                st.markdown('</div>', unsafe_allow_html=True)
