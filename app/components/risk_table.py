import streamlit as st
import pandas as pd

def render_risk_table():

    df = pd.DataFrame(
        {
            "State":[
                "Maharashtra",
                "West Bengal",
                "Karnataka",
                "Tamil Nadu",
                "Bihar"
            ],
            "Risk":[
                "High",
                "High",
                "Medium",
                "Medium",
                "Low"
            ]
        }
    )

    st.dataframe(
        df,
        use_container_width=True
    )