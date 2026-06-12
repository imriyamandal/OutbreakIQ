from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from app.schemas import PredictionRequest
from app.services.predictor import Predictor
from app.services.alerts import generate_alert, scan_and_generate_alerts
from app.utils.history_db import save_prediction, get_history, init_db

router = APIRouter()
predictor_service = Predictor()

init_db()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "ml" / "data" / "final" / "ml_data.csv"

if DATA_PATH.exists():
    try:
        df_global = pd.read_csv(DATA_PATH)
        df_global.columns = df_global.columns.str.strip().str.lower()
    except Exception as e:
        print(f"Error loading global dataset: {e}")
        df_global = pd.DataFrame()
else:
    print(f"Global dataset not found at {DATA_PATH}")
    df_global = pd.DataFrame()


class ForecastRequest(BaseModel):
    state_ut: str
    district: str
    disease: str
    steps: int = Field(default=6, ge=1, le=12)


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "dataset_loaded": not df_global.empty,
        "records_count": len(df_global) if not df_global.empty else 0
    }

_cached_alerts = None

@router.post("/predict")
def predict(payload: PredictionRequest):
    global _cached_alerts
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded on backend.")
    
    input_dict = payload.model_dump()
    result = predictor_service.predict(input_dict)

    save_prediction(result["enriched_input"], result)
    
    _cached_alerts = None

    alert_info = generate_alert(
        probability=result["outbreak_probability"],
        predicted_cases=result["predicted_cases"]
    )
    result["alert"] = alert_info
    
    return result


@router.get("/metadata")
def get_metadata():
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded.")
    
    diseases = sorted(df_global["disease"].dropna().unique().tolist())
    states = sorted(df_global["state_ut"].dropna().unique().tolist())
    
    state_districts = {}
    for state in states:
        districts = sorted(df_global[df_global["state_ut"] == state]["district"].dropna().unique().tolist())
        state_districts[state] = districts
        
    return {
        "diseases": diseases,
        "states": states,
        "state_districts": state_districts
    }


def clean_records(records):
    cleaned = []
    for record in records:
        cleaned_rec = {}
        for k, v in record.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                cleaned_rec[k] = None
            else:
                cleaned_rec[k] = v
        cleaned.append(cleaned_rec)
    return cleaned


@router.get("/model/metrics")
def get_model_metrics():
    reports_dir = BASE_DIR / "ml" / "reports"
    cls_path = reports_dir / "metrics" / "classification_metrics.csv"
    reg_path = reports_dir / "metrics" / "xgb_regression_metrics.csv"
    
    cls_metrics = []
    reg_metrics = []
    
    if cls_path.exists():
        cls_metrics = clean_records(pd.read_csv(cls_path).to_dict("records"))
    if reg_path.exists():
        reg_metrics = clean_records(pd.read_csv(reg_path).to_dict("records"))
        
    return {
        "classification": cls_metrics,
        "regression": reg_metrics
    }


@router.get("/model/feature-importance")
def get_feature_importance():
    reports_dir = BASE_DIR / "ml" / "reports"
    cls_fi_path = reports_dir / "xgb_cls_feature_importance.csv"
    reg_fi_path = reports_dir / "xgb_reg_feature_importance.csv"
    
    cls_fi = []
    reg_fi = []
    
    if cls_fi_path.exists():
        cls_fi = clean_records(pd.read_csv(cls_fi_path).to_dict("records"))
    if reg_fi_path.exists():
        reg_fi = clean_records(pd.read_csv(reg_fi_path).to_dict("records"))
        
    return {
        "classification": cls_fi,
        "regression": reg_fi
    }


@router.get("/model/plots/{plot_name}")
def get_plot_image(plot_name: str):
    reports_dir = BASE_DIR / "ml" / "reports"
    filename = f"{plot_name}.png"
    filepath = reports_dir / filename
    if filepath.exists():
        return FileResponse(str(filepath))
    raise HTTPException(status_code=404, detail="Plot image not found.")


@router.post("/forecast")
def forecast(payload: ForecastRequest):
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded on backend.")
        
    state = payload.state_ut
    district = payload.district
    disease = payload.disease
    steps = payload.steps
    
    filtered = df_global[
        (df_global["state_ut"].str.lower() == state.lower()) &
        (df_global["district"].str.lower() == district.lower()) &
        (df_global["disease"].str.lower() == disease.lower())
    ].sort_values(by=["year", "month", "day"])
    
    forecast_results = []
    
    if len(filtered) >= 3:
        last_row = filtered.iloc[-1]
        last_year = int(last_row.get("year", 2024))
        last_month = int(last_row.get("month", 6))

        monthly_avg = filtered.groupby("month")["cases"].mean().to_dict()
        
        recent_cases = filtered.tail(6)["cases"].mean()
        overall_cases = filtered["cases"].mean()
        trend_factor = (recent_cases / overall_cases) if overall_cases > 0 else 1.0
        trend_factor = np.clip(trend_factor, 0.5, 2.0)
        
        curr_year = last_year
        curr_month = last_month
        
        for i in range(1, steps + 1):
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1
                
            base_cases = monthly_avg.get(curr_month, overall_cases)
            predicted = int(round(base_cases * trend_factor))
            predicted = max(predicted, 0)
            
            prob = 0.1
            if predicted > overall_cases * 1.5:
                prob = 0.75
            elif predicted > overall_cases:
                prob = 0.45
            
            if prob >= 0.8:
                risk = "Critical"
            elif prob >= 0.6:
                risk = "High"
            elif prob >= 0.3:
                risk = "Medium"
            else:
                risk = "Low"
                
            forecast_results.append({
                "step": i,
                "year": curr_year,
                "month": curr_month,
                "month_name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][curr_month - 1],
                "predicted_cases": predicted,
                "outbreak_probability": round(prob, 4),
                "risk_level": risk
            })
    else:
        curr_month = 6
        curr_year = 2024
        for i in range(1, steps + 1):
            curr_month += 1
            if curr_month > 12:
                curr_month = 1
                curr_year += 1
            forecast_results.append({
                "step": i,
                "year": curr_year,
                "month": curr_month,
                "month_name": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][curr_month - 1],
                "predicted_cases": 150 + i * 20,
                "outbreak_probability": 0.25 + i * 0.05,
                "risk_level": "Low" if i < 3 else "Medium"
            })
            
    return {"forecast": forecast_results}


@router.get("/dashboard/stats")
def dashboard_stats():
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded.")
        
    total_cases = int(df_global["cases"].sum())
    total_outbreaks = int(df_global["outbreak"].sum())
    
    state_outbreaks = df_global.groupby("state_ut")["outbreak"].sum()
    high_risk_states = int((state_outbreaks > 10).sum())
    
    most_affected = str(df_global.groupby("disease")["cases"].sum().idxmax())
    
    trends = df_global.groupby(["year", "month"])["cases"].sum().reset_index()
    trends = trends.sort_values(["year", "month"]).tail(12)
    monthly_trends = [
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(row["month"]) - 1] + f" {int(row['year'])}",
            "cases": int(row["cases"])
        }
        for _, row in trends.iterrows()
    ]
    
    low_cnt = int((df_global["cases"] <= 10).sum())
    med_cnt = int(((df_global["cases"] > 10) & (df_global["cases"] <= 50)).sum())
    high_cnt = int(((df_global["cases"] > 50) & (df_global["cases"] <= 100)).sum())
    crit_cnt = int((df_global["cases"] > 100).sum())
    
    return {
        "total_cases": total_cases,
        "active_outbreaks": total_outbreaks,
        "high_risk_states": high_risk_states,
        "most_affected_disease": most_affected,
        "monthly_trends": monthly_trends,
        "risk_distribution": {
            "Low": low_cnt,
            "Medium": med_cnt,
            "High": high_cnt,
            "Critical": crit_cnt
        }
    }


@router.get("/analytics")
def analytics_data():
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded.")
        
    disease_cases = df_global.groupby("disease")["cases"].sum().reset_index()
    disease_trends = [
        {"disease": str(row["disease"]), "cases": int(row["cases"])}
        for _, row in disease_cases.iterrows()
    ]
    
    state_comparison = df_global.groupby("state_ut")["cases"].sum().reset_index()
    state_comparison = state_comparison.sort_values("cases", ascending=False).head(10)
    state_trends = [
        {"state": str(row["state_ut"]), "cases": int(row["cases"])}
        for _, row in state_comparison.iterrows()
    ]
    
    seasonal = df_global.groupby("month")["cases"].mean().reset_index()
    seasonal_analysis = [
        {
            "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(row["month"]) - 1],
            "average_cases": round(float(row["cases"]), 2)
        }
        for _, row in seasonal.iterrows()
    ]
    
    cols = ["temp", "precipitation", "lai", "cases", "outbreak"]
    corr_matrix = df_global[cols].corr().fillna(0).to_dict()
    
    corr_data = {
        "x": cols,
        "y": cols,
        "z": [[round(corr_matrix[col1][col2], 3) for col2 in cols] for col1 in cols]
    }
    
    freq = df_global.groupby("disease")["outbreak"].sum().reset_index()
    outbreak_frequency = [
        {"disease": str(row["disease"]), "outbreaks": int(row["outbreak"])}
        for _, row in freq.iterrows()
    ]
    
    return {
        "disease_trends": disease_trends,
        "state_trends": state_trends,
        "seasonal_analysis": seasonal_analysis,
        "correlation": corr_data,
        "outbreak_frequency": outbreak_frequency
    }


@router.get("/heatmap")
def heatmap_data(
    disease: Optional[str] = Query(None),
    state: Optional[str] = Query(None)
):
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded.")
        
    df_filtered = df_global.copy()
    if disease and disease != "All":
        df_filtered = df_filtered[df_filtered["disease"].str.lower() == disease.lower()]
    if state and state != "All":
        df_filtered = df_filtered[df_filtered["state_ut"].str.lower() == state.lower()]
        
    df_filtered = df_filtered.dropna(subset=["latitude", "longitude", "cases"])
    if df_filtered.empty:
        return {"points": []}
        
    idx_latest = df_filtered.groupby(["state_ut", "district", "disease"])["date"].idxmax()
    df_latest = df_filtered.loc[idx_latest]
    
    points = []
    for _, row in df_latest.iterrows():
        prob = 0.85 if row["outbreak"] == 1 else (0.50 if row["cases"] > 50 else 0.15)
        risk = "Critical" if prob >= 0.8 else ("High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low"))
        
        points.append({
            "state_ut": str(row["state_ut"]),
            "district": str(row["district"]),
            "disease": str(row["disease"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "cases": int(row["cases"]),
            "risk_level": risk,
            "outbreak_probability": prob
        })
        
    return {"points": points}


@router.get("/alerts")
def alerts_data():
    global _cached_alerts
    if df_global.empty:
        raise HTTPException(status_code=500, detail="Historical dataset not loaded on backend.")
    
    if _cached_alerts is None:
        _cached_alerts = scan_and_generate_alerts(df_global, predictor_service)
        
    return {"alerts": _cached_alerts}


@router.get("/history")
def history(
    search: Optional[str] = Query(None),
    disease: Optional[str] = Query(None),
    state: Optional[str] = Query(None)
):
    history_logs = get_history(
        search_query=search,
        disease_filter=disease,
        state_filter=state
    )
    return {"history": history_logs}