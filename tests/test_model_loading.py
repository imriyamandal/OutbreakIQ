from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "ml" / "models"


def test_regressor_exists():

    assert (
        MODELS_DIR / "regressor.pkl"
    ).exists()


def test_classifier_exists():

    assert (
        MODELS_DIR / "classifier.pkl"
    ).exists()


def test_feature_schema_exists():

    assert (
        MODELS_DIR / "feature_schema.pkl"
    ).exists()


def test_threshold_exists():

    assert (
        MODELS_DIR / "threshold.pkl"
    ).exists()