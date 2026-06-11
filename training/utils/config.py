from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Paths
RAW_DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "raw_data.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "processed_data.csv"
)

FINAL_DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "final"
    / "ml_data.csv"
)

# Models
CLASSIFIER_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "classifier.pkl"
)

REGRESSOR_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "regressor.pkl"
)

FEATURE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "feature_schema.pkl"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "ml"
    / "models"
    / "threshold.pkl"
)

# Reports
REPORTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "reports"
)

FIGURES_DIR = (
    REPORTS_DIR
    / "figures"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

SUMMARIES_DIR = (
    REPORTS_DIR
    / "summaries"
)