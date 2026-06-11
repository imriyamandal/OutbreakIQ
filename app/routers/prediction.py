from fastapi import APIRouter

from app.schemas import PredictionRequest

from training.models.predict import DiseasePredictor

from app.services.alerts import generate_alert

router = APIRouter()

predictor = DiseasePredictor()


@router.post("/predict")
def predict(payload: PredictionRequest):

    result = predictor.predict(
        payload.model_dump()
    )

    alert_info = generate_alert(
        probability=result["outbreak_probability"],
        predicted_cases=result["predicted_cases"]
    )

    result["alert"] = alert_info

    return result


@router.post("/risk")
def risk(payload: PredictionRequest):

    result = predictor.predict(
        payload.model_dump()
    )

    return {
        "risk_level": result["risk_level"],
        "outbreak_probability":
        result["outbreak_probability"]
    }


@router.post("/alert")
def alert(payload: PredictionRequest):

    result = predictor.predict(
        payload.model_dump()
    )

    alert_info = generate_alert(
        probability=result["outbreak_probability"],
        predicted_cases=result["predicted_cases"]
    )

    return alert_info


@router.get("/health")
def health():

    return {
        "status": "healthy"
    }


@router.get("/feature-importance")
def feature_importance():

    return {
        "image":
        "ml/reports/shap_summary.png"
    }