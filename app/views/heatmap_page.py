import streamlit as st
import pandas as pd
from app.utils.api_client import APIClient
from app.components.heatmap import render_heatmap

@st.cache_data
def get_metadata():
    return APIClient.get("/metadata")

def show_heatmap():
    st.markdown('<div class="page-title">Disease Hotspot Heatmap (latest data - 2022)</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Geospatial visualization of outbreak intensity and regional risk levels</div>', unsafe_allow_html=True)

    metadata = get_metadata()
    if not metadata:
        st.error("Failed to load location/disease metadata from backend API.")
        return

    diseases = ["All"] + metadata.get("diseases", [])
    states = ["All"] + metadata.get("states", [])

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Geospatial Filters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        disease = st.selectbox(
            "Filter Heatmap by Disease",
            options=diseases
        )

    with col2:
        state = st.selectbox(
            "Zoom to State/UT",
            options=states
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Load heat points from backend
    with st.spinner("Fetching coordinates and case counts..."):
        res = APIClient.get("/heatmap", params={"disease": disease, "state": state})

    if res and "points" in res:
        points = res["points"]
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Risk Map</div>', unsafe_allow_html=True)

        if points:
            df_mapped = pd.DataFrame(points)
            render_heatmap(df_mapped)
        else:
            st.info("No geospatial data matches the filter criteria.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Failed to retrieve heatmap points from backend API.")
