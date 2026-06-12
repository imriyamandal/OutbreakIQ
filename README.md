# OutbreakIQ

## AI-Powered Disease Outbreak Forecasting and Early Warning System

OutbreakIQ is an intelligent disease surveillance platform that uses machine learning, time-series forecasting, explainable AI, and geospatial analytics to predict disease outbreaks and provide early warning alerts.

The system combines environmental, epidemiological, temporal, and geographic features to estimate future disease cases and outbreak risk levels.

---

## Features

### Disease Case Forecasting

* XGBoost Regressor
* LSTM Neural Network
* ARIMA Forecasting
* Prophet Forecasting

### Outbreak Classification

* XGBoost Classifier
* Random Forest Baseline
* SMOTE for class balancing

### Explainable AI

* SHAP Summary Plot
* SHAP Dependence Plot
* Feature Importance Analysis

### Early Warning System

* Low Risk Alerts
* Moderate Risk Alerts
* High Risk Alerts
* Critical Risk Alerts

### Geographic Intelligence

* District-Level Risk Mapping
* Heatmap Visualization
* Environmental Risk Factors

### Web Application

* FastAPI Backend
* Streamlit Dashboard
* Real-Time Prediction Interface

---

## Project Structure

```text
OutbreakIQ

app/
├── main.py
├── routers/
├── services/
├── schemas/

dashboard/
├── app.py
├── pages/

ml/
├── data/
├── models/
├── reports/
├── explainability/

training/
├── models/
├── evaluation/
```

---

## Machine Learning Models

### Forecasting Models

| Model             | Purpose                   |
| ----------------- | ------------------------- |
| XGBoost Regressor | Disease Case Forecasting  |
| LSTM              | Deep Learning Forecasting |
| Prophet           | Seasonal Forecasting      |
| ARIMA             | Statistical Forecasting   |

### Classification Models

| Model              | Purpose                 |
| ------------------ | ----------------------- |
| XGBoost Classifier | Outbreak Prediction     |
| Random Forest      | Baseline Classification |

---

## Technologies Used

### Backend

* FastAPI
* Uvicorn

### Machine Learning

* Scikit-Learn
* XGBoost
* TensorFlow
* Prophet
* StatsModels

### Explainability

* SHAP

### Visualization

* Matplotlib
* Plotly
* Folium

### Dashboard

* Streamlit

### Data Processing

* Pandas
* NumPy

---

## Installation

Clone repository:

```bash
git clone https://github.com/your-username/OutbreakIQ.git

cd OutbreakIQ
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux/Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training Models

Run complete training pipeline:

```bash
python -m training.models.train
```

Run Prophet:

```bash
python -m training.models.prophet_model
```

Run ARIMA:

```bash
python -m training.models.arima_model
```

Run LSTM:

```bash
python -m training.models.lstm_model
```

---

## SHAP Explainability

```bash
python ml/explainability/shap_analysis.py
```

Outputs:

```text
ml/reports/shap_summary.png
ml/reports/shap_dependence.png
```

---

## Model Comparison

```bash
python -m training.evaluation.model_comparison
```

Outputs:

```text
forecasting_model_comparison.csv
classification_model_comparison.csv
```

---

## Run FastAPI

```bash
uvicorn app.main:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## Research Objectives

1. Forecast disease outbreaks using machine learning.
2. Generate outbreak risk probabilities.
3. Provide explainable predictions using SHAP.
4. Visualize disease hotspots geographically.
5. Support public health decision-making through early warning alerts.

---


OutbreakIQ – Disease Outbreak Forecasting and Early Warning System
