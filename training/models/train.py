import logging
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATA_PATH  = PROJECT_ROOT / "ml" / "data" / "final" / "ml_data.csv"
MODELS_DIR       = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR      = PROJECT_ROOT / "ml" / "reports"
METRICS_DIR      = REPORTS_DIR / "metrics"
ENCODERS_DIR     = MODELS_DIR / "encoders"

for d in [MODELS_DIR, METRICS_DIR, ENCODERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    "state_ut",
    "district",
    "disease",
    "disease_category",
]

NUMERIC_FEATURES = [
    "day",
    "month",
    "year",
    "latitude",
    "longitude",
    "precipitation",
    "lai",
    "temp",
    "death_rate",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "roll_3_mean",
    "roll_6_mean",
    "roll_3_std",
    "roll_6_std",
    "month_sin",
    "month_cos",
    "temp_change",
    "precipitation_change",
    "lai_change",
    "spike_ratio",
    "acceleration",
    "growth_rate",
    "outbreak_frequency_12m",
]

ENCODED_CAT_FEATURES = [f"{c}_enc" for c in CATEGORICAL_COLS]
ALL_FEATURES         = NUMERIC_FEATURES + ENCODED_CAT_FEATURES

TARGET_REG = "log_cases"
TARGET_CLS = "outbreak"

def load_and_prepare(path: Path) -> pd.DataFrame:

    logger.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.sort_values(
        ["state_ut", "district", "disease", "year", "month", "day"]
    ).reset_index(drop=True)

    logger.info(f"Dataset shape: {df.shape}")

    import joblib
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(
                df[col].astype(str).fillna("unknown")
            )
            enc_path = ENCODERS_DIR / f"le_{col}.pkl"
            joblib.dump(le, enc_path)
            logger.info(f"  Encoder saved: {enc_path.name}")

    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(0)

    return df

def temporal_split(df: pd.DataFrame):

    logger.info("Performing TimeSeriesSplit (5 folds) ...")
    df = df.sort_values(
        ["state_ut","district","disease","year","month","day"]
    )
    df = df.sort_values("date")
    X = df[ALL_FEATURES].values
    y_reg = df[TARGET_REG].values
    y_cls = df[TARGET_CLS].values

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))

    train_idx, val_idx   = splits[-2]
    _,          test_idx = splits[-1]

    X_train = X[train_idx];   X_val = X[val_idx];   X_test = X[test_idx]
    y_reg_train = y_reg[train_idx]; y_reg_val = y_reg[val_idx]; y_reg_test = y_reg[test_idx]
    y_cls_train = y_cls[train_idx]; y_cls_val = y_cls[val_idx]; y_cls_test = y_cls[test_idx]

    logger.info(
        f"  Train: {len(train_idx):,} | "
        f"Val: {len(val_idx):,} | "
        f"Test: {len(test_idx):,}"
    )

    return (
        X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        ALL_FEATURES,
    )

def main():
    logger.info("MASTER TRAINING PIPELINE STARTED")
    df = load_and_prepare(FINAL_DATA_PATH)

    (
        X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        feature_names,
    ) = temporal_split(df)

    logger.info("Saving feature schema...")

    feature_schema = {
    "all_features": ALL_FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_cols": CATEGORICAL_COLS,
    "encoded_cat_features": ENCODED_CAT_FEATURES,
    "target_reg": TARGET_REG,
    "target_cls": TARGET_CLS,
    }

    schema_path = MODELS_DIR / "feature_schema.pkl"
    joblib.dump(feature_schema, schema_path)

    logger.info(f"Feature schema saved: {schema_path}")

    from training.models.train_regressor import train_regressor 
    reg_model = train_regressor(
        X_train, y_reg_train,
        X_val,   y_reg_val,
        feature_names,
    )

    from training.models.train_classifier import train_classifier
    cls_model = train_classifier(
        X_train, y_cls_train,
        X_val,   y_cls_val,
        feature_names,
    )

    threshold = { "outbreak_threshold": 0.5 }

    threshold_path = MODELS_DIR / "threshold.pkl"
    joblib.dump(threshold, threshold_path)
    logger.info(f"Threshold saved: {threshold_path}")

    from training.models.evaluate import evaluate_regressor, evaluate_classifier
    evaluate_regressor(reg_model, X_test, y_reg_test, split="test")
    evaluate_classifier(cls_model, X_test, y_cls_test, split="test")

    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Models saved to : {MODELS_DIR}")
    logger.info(f"Reports saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()