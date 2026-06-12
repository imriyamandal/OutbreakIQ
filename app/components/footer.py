import streamlit as st

def render_footer():

    st.divider()

    st.markdown(
        """
        <center>
        OutbreakIQ • Disease Outbreak Prediction &
        Early Warning System
        </center>
        """,
        unsafe_allow_html=True
    )