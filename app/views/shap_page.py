import streamlit as st
from pathlib import Path

def show_shap():

    st.title(
        "Model Explainability"
    )

    shap_img = Path(
        "ml/reports/shap_summary.png"
    )

    if shap_img.exists():

        st.image(
            str(shap_img),
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP plot not found"
        )