from pathlib import Path
import streamlit as st

def render_navbar():

    st.markdown(
        """
        <div style="padding-top:10px;">
            <h1 style="
                color:#166534;
                margin:0;
                font-size:42px;
                font-weight:800;
            ">
                OutbreakIQ
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )