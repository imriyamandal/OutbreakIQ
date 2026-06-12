from pathlib import Path
import streamlit as st

def render_navbar():

    logo_path = (
        Path(__file__).parent.parent
        / "assets"
        / "logo.jpg"
    )

    col1,col2 = st.columns([1,5])

    with col1:
        st.image(str(logo_path), width=80)

    with col2:
        st.markdown(
            """
            <h1 style="color:#166534;">
            OutbreakIQ
            </h1>
            """,
            unsafe_allow_html=True
        )