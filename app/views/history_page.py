import streamlit as st
import pandas as pd
from app.utils.api_client import APIClient

def show_history():
    st.markdown('<div class="page-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Logs of all previous model runs and forecast predictions</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Filters & Search</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        search = st.text_input("Search (District or Disease)", placeholder="Type to search...")
        
    with col2:
        # Fetch list of diseases from API or hardcode standard ones
        disease = st.selectbox(
            "Filter by Disease",
            options=["All", "Dengue", "Malaria", "Chikungunya", "Acute Diarrhoeal Disease", "Cholera", "Typhoid", "Viral Hepatitis"]
        )
        disease_filter = None if disease == "All" else disease

    with col3:
        # Fetch list of states
        state = st.selectbox(
            "Filter by State/UT",
            options=["All", "Andhra Pradesh", "Assam", "Bihar", "Delhi", "Gujarat", "Jharkhand", "Karnataka", "Maharashtra", "Tamil Nadu", "West Bengal"]
        )
        state_filter = None if state == "All" else state

    st.markdown('</div>', unsafe_allow_html=True)

    # Fetch history logs from API
    params = {}
    if search:
        params["search"] = search
    if disease_filter:
        params["disease"] = disease_filter
    if state_filter:
        params["state"] = state_filter

    with st.spinner("Loading prediction history..."):
        res = APIClient.get("/history", params=params)

    if res and "history" in res:
        history_list = res["history"]
        if history_list:
            df_hist = pd.DataFrame(history_list)
            
            # Format and display
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">Logged Predictions ({len(df_hist)})</div>', unsafe_allow_html=True)
            
            # Clean up columns for presentation
            cols_to_show = [
                "created_at", "state_ut", "district", "disease",
                "temp", "precipitation", "lai",
                "predicted_cases", "outbreak_probability", "risk_level"
            ]
            display_df = df_hist[cols_to_show].copy()
            display_df.rename(columns={
                "created_at": "Timestamp",
                "state_ut": "State/UT",
                "district": "District",
                "disease": "Disease",
                "temp": "Temp (K)",
                "precipitation": "Precipitation",
                "lai": "LAI",
                "predicted_cases": "Cases Predicted",
                "outbreak_probability": "Outbreak Prob",
                "risk_level": "Risk Level"
            }, inplace=True)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export CSV
            csv_data = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_data,
                file_name="outbreak_prediction_history.csv",
                mime="text/csv"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No prediction records match the selected criteria.")
    else:
        st.warning("Failed to fetch history logs from the server.")
StreamlitHistoryPage = show_history
