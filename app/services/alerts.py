from datetime import datetime


def generate_alert(
    probability: float,
    predicted_cases: int
):

    if probability >= 0.80:
        level = "CRITICAL"
        message = "Immediate outbreak intervention required"

    elif probability >= 0.60:
        level = "HIGH"
        message = "Potential outbreak likely"

    elif probability >= 0.40:
        level = "MODERATE"
        message = "Monitor disease activity"

    else:
        level = "LOW"
        message = "Situation stable"

    return {
        "alert_level": level,
        "message": message,
        "predicted_cases": predicted_cases,
        "generated_at": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }


def scan_and_generate_alerts(df_global, predictor_service):
    """
    Scans the latest record for each state-district-disease combination (recent records from 2021/2022)
    and generates dynamic, prediction-driven warnings.
    """
    if df_global.empty:
        return []
        
    # Drop rows with NaN lat/lon
    df_clean = df_global.dropna(subset=["latitude", "longitude", "cases"])
    
    # Filter for combinations with records in 2021 or 2022
    df_recent = df_clean[df_clean["year"] >= 2021]
    if df_recent.empty:
        df_recent = df_clean  # Fallback if no recent data exists
        
    # Get index of the latest record for each combination
    idx_latest = df_recent.groupby(["state_ut", "district", "disease"])["date"].idxmax()
    df_latest = df_recent.loc[idx_latest]
    
    alerts = []
    for _, row in df_latest.iterrows():
        input_dict = row.to_dict()
        try:
            # We predict without SHAP for speed
            pred = predictor_service.predict(input_dict, calculate_shap=False)
            prob = pred["outbreak_probability"]
            predicted = pred["predicted_cases"]
        except Exception as e:
            # Fallback based on historical flag
            prob = 0.85 if row["outbreak"] == 1 else 0.15
            predicted = int(row["cases"])
            
        # Determine warning severity and message
        if prob >= 0.85:
            level = "CRITICAL"
            msg = f"Critical Outbreak: Immediate intervention required. {predicted} predicted cases."
        elif prob >= 0.65:
            level = "HIGH"
            msg = f"Emerging Outbreak: Potential spike likely. {predicted} predicted cases."
        elif prob >= 0.45:
            level = "HIGH"
            msg = f"High Risk: Monitor closely. {predicted} predicted cases."
        elif prob >= 0.30:
            level = "MODERATE"
            msg = f"Moderate Risk: Stable but elevated activity."
        else:
            continue  # Low risk locations are omitted from active alerts to reduce clutter
            
        alerts.append({
            "state_ut": str(row["state_ut"]),
            "district": str(row["district"]),
            "disease": str(row["disease"]),
            "cases": predicted,
            "risk_level": level,
            "message": msg,
            "year": int(row["year"]),
            "month": int(row["month"])
        })
        
    # Sort alerts so critical/high are at the top
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    alerts = sorted(alerts, key=lambda x: severity_order.get(x["risk_level"], 3))
    
    return alerts