from app.services.alerts import generate_alert


def test_critical_alert():

    result = generate_alert(
        probability=0.90,
        predicted_cases=200
    )

    assert result["alert_level"] == "CRITICAL"


def test_high_alert():

    result = generate_alert(
        probability=0.70,
        predicted_cases=100
    )

    assert result["alert_level"] == "HIGH"


def test_moderate_alert():

    result = generate_alert(
        probability=0.50,
        predicted_cases=50
    )

    assert result["alert_level"] == "MODERATE"


def test_low_alert():

    result = generate_alert(
        probability=0.10,
        predicted_cases=10
    )

    assert result["alert_level"] == "LOW"
from app.services.alerts import generate_alert


def test_alert_generation():

    result = generate_alert(
        probability=0.9,
        predicted_cases=100
    )

    assert result["alert_level"] == "CRITICAL"