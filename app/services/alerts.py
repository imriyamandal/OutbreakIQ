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