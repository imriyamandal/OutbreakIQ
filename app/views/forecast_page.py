import streamlit as st
import requests

def show_forecast():

    st.title("Disease Forecast")

    state = st.selectbox(
        "State",
        sorted(df["state_ut"].unique())
    )

    district = st.selectbox(
        "District",
        sorted(
            df[
                df["state_ut"] == state
            ]["district"].unique()
        )
    )
    

    disease = st.selectbox(
        "Disease",
        sorted(df["disease"].unique())
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=30.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        value=100.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=75.0
    )

    if st.button("Generate Prediction"):

        payload = {
            "state": state,
            "disease": disease,
            "temperature": temperature,
            "rainfall": rainfall,
            "humidity": humidity
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=30
        )

        result = response.json()

        st.success("Prediction Generated")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Cases",
            result.get("predicted_cases", 0)
        )

        c2.metric(
            "Probability",
            f"{result.get('probability',0)}%"
        )

        c3.metric(
            "Risk Level",
            result.get("risk","Unknown")
        )