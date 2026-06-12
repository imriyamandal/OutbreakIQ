from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_dashboard_stats():
    response = client.get("/dashboard/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_cases" in data
    assert "active_outbreaks" in data
    assert "high_risk_states" in data
    assert "most_affected_disease" in data
    assert "monthly_trends" in data
    assert "risk_distribution" in data

def test_analytics_data():
    response = client.get("/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "disease_trends" in data
    assert "state_trends" in data
    assert "seasonal_analysis" in data
    assert "correlation" in data
    assert "outbreak_frequency" in data

def test_heatmap_data():
    response = client.get("/heatmap")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    if len(data["points"]) > 0:
        pt = data["points"][0]
        assert "state_ut" in pt
        assert "latitude" in pt
        assert "longitude" in pt
        assert "cases" in pt

def test_alerts_data():
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    if len(data["alerts"]) > 0:
        alert = data["alerts"][0]
        assert "state_ut" in alert
        assert "district" in alert
        assert "disease" in alert
        assert "cases" in alert
        assert "risk_level" in alert
        assert "message" in alert

def test_history_endpoint():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data

def test_predict_endpoint():
    payload = {
        "state_ut": "Jharkhand",
        "district": "Ranchi",
        "disease": "Dengue",
        "disease_category": "Vector-borne",
        "day": 1,
        "month": 8,
        "year": 2024,
        "latitude": 23.35,
        "longitude": 85.33,
        "precipitation": 100.0,
        "lai": 2.0,
        "temp": 303.15,  # Kelvin
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_cases" in data
    assert "outbreak_probability" in data
    assert "risk_level" in data
    assert "confidence_score" in data
    assert "shap_top_contributors" in data

def test_forecast_endpoint():
    payload = {
        "state_ut": "Jharkhand",
        "district": "Ranchi",
        "disease": "Dengue",
        "steps": 6
    }
    response = client.post("/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 6
    step = data["forecast"][0]
    assert "step" in step
    assert "predicted_cases" in step
    assert "outbreak_probability" in step
    assert "risk_level" in step


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "diseases" in data
    assert "states" in data
    assert "state_districts" in data
    assert len(data["diseases"]) > 0
    assert len(data["states"]) > 0


def test_model_metrics_endpoint():
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "regression" in data


def test_model_feature_importance_endpoint():
    response = client.get("/model/feature-importance")
    assert response.status_code == 200
    data = response.json()
    assert "classification" in data
    assert "regression" in data


def test_model_plots_endpoint():
    response = client.get("/model/plots/shap_summary")
    assert response.status_code in [200, 404]
