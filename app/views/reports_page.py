import streamlit as st
import pandas as pd

def show_reports():

    st.title(
        "Model Reports"
    )

    metrics = pd.DataFrame(
        {
            "Model":[
                "XGBoost",
                "LSTM",
                "ARIMA"
            ],
            "RMSE":[
                12.5,
                15.1,
                18.3
            ]
        }
    )

    st.dataframe(
        metrics,
        use_container_width=True
    )