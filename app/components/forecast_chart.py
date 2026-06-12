import streamlit as st
import pandas as pd
import plotly.express as px

def render_forecast_chart():

    df = pd.DataFrame(
        {
            "Month":[
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec"
            ],
            "Cases":[
                150,
                220,
                280,
                350,
                300,
                260
            ]
        }
    )

    fig = px.line(
        df,
        x="Month",
        y="Cases",
        markers=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )