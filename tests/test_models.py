from training.models.predict import DiseasePredictor


def test_model_loading():

    predictor = DiseasePredictor()

    assert predictor is not None