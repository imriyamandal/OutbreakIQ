import sys
from pathlib import Path
import streamlit as st
API_URL = st.secrets.get("API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.components.navbar import render_navbar
from app.components.sidebar import render_sidebar

from app.views.overview_page import show_overview
from app.views.prediction_page import show_prediction
from app.views.analytics_page import show_analytics
from app.views.model_insights_page import show_model_insights
from app.views.history_page import show_history
from app.views.heatmap_page import show_heatmap
from app.views.alerts_page import show_alerts

st.set_page_config(
    page_title="OutbreakIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    with open("app/assets/style.css") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
except FileNotFoundError:
    pass

render_navbar()

page = render_sidebar()

if page == "Overview":
    show_overview()

elif page == "Prediction":
    show_prediction()

elif page == "Analytics":
    show_analytics()

elif page == "Model Insights":
    show_model_insights()

elif page == "History":
    show_history()

elif page == "Heatmap":
    show_heatmap()

elif page == "Alerts":
    show_alerts()