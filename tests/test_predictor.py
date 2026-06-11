from training.models.predict import DiseasePredictor


def test_predictor_loads():

    predictor = DiseasePredictor()

    assert predictor.regressor is not None
    assert predictor.classifier is not None

def test_prediction_returns_expected_keys():

    predictor = DiseasePredictor()

    sample = {

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
        "temp": 30.0,

        "death_rate": 0.0,

        "lag_1": 10,
        "lag_2": 8,
        "lag_3": 7,
        "lag_6": 5,
        "lag_12": 3,

        "roll_3_mean": 8,
        "roll_6_mean": 7,

        "roll_3_std": 1,
        "roll_6_std": 1,

        "month_sin": 0,
        "month_cos": 0,

        "temp_change": 0,
        "precipitation_change": 0,
        "lai_change": 0,

        "spike_ratio": 1.2,
        "acceleration": 1,
        "growth_rate": 0.2,

        "outbreak_frequency_12m": 2
    }

    result = predictor.predict(sample)

    assert "predicted_cases" in result
    assert "outbreak_probability" in result
    assert "risk_level" in result