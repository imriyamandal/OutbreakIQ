import logging
import warnings
import joblib
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "ml" / "models"
ENCODERS_DIR = MODELS_DIR / "encoders"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

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

CATEGORICAL_COLS    = ["state_ut", "district", "disease", "disease_category"]
ENCODED_CAT_FEATURES = [f"{c}_enc" for c in CATEGORICAL_COLS]
ALL_FEATURES         = NUMERIC_FEATURES + ENCODED_CAT_FEATURES

RISK_THRESHOLDS = {
    "Low"      : (0.00, 0.35),
    "Moderate" : (0.35, 0.60),
    "High"     : (0.60, 0.80),
    "Critical" : (0.80, 1.01),
}

def _get_risk_level(prob: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= prob < hi:
            return level
    return "Unknown"

class DiseasePredictor:
    _instance: Optional["DiseasePredictor"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_models()
            self._loaded = True

    def _load_models(self):

        logger.info("Loading trained models ...")

        reg_path = MODELS_DIR / "regressor.pkl"
        cls_path = MODELS_DIR / "classifier.pkl"

        if not reg_path.exists():
            raise FileNotFoundError(
                f"Regressor not found at {reg_path}. "
                "Run: python src/models/train.py"
            )
        if not cls_path.exists():
            raise FileNotFoundError(
                f"Classifier not found at {cls_path}. "
                "Run: python src/models/train.py"
            )

        self.regressor  = joblib.load(reg_path)
        self.classifier = joblib.load(cls_path)
        threshold_path = MODELS_DIR / "threshold.pkl"

        if threshold_path.exists():
            threshold_data = joblib.load(threshold_path)
            self.outbreak_threshold = threshold_data.get(
                "outbreak_threshold",0.5
            )
        else:
            self.outbreak_threshold = 0.5

        logger.info(f"  Regressor loaded : {type(self.regressor).__name__}")
        logger.info(f"  Classifier loaded: {type(self.classifier).__name__}")

        self.encoders: Dict[str, Any] = {}
        for col in CATEGORICAL_COLS:
            enc_path = ENCODERS_DIR / f"le_{col}.pkl"
            if enc_path.exists():
                self.encoders[col] = joblib.load(enc_path)
                logger.info(f"  Encoder loaded: le_{col}.pkl")
            else:
                logger.warning(f"  Encoder missing: {enc_path} — will use 0 for {col}")

    def _prepare_input(self, data: Dict[str, Any]) -> np.ndarray:
        row = {}

        for feat in NUMERIC_FEATURES:
            row[feat] = float(data.get(feat, 0.0))

        month = row["month"]
        if row.get("month_sin", 0.0) == 0.0 and month != 0:
            row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
            row["month_cos"] = float(np.cos(2 * np.pi * month / 12))

        for col in CATEGORICAL_COLS:
            raw_val = str(data.get(col, "unknown"))
            enc_key = f"{col}_enc"
            if col in self.encoders:
                le = self.encoders[col]
                if raw_val in le.classes_:
                    row[enc_key] = int(le.transform([raw_val])[0])
                else:
                    logger.warning(
                        f"  Unknown {col}='{raw_val}'; encoding as 0"
                    )
                    row[enc_key] = 0
            else:
                row[enc_key] = 0

        print("\nFEATURE ORDER")
        for i,feat in enumerate(ALL_FEATURES):
            print(i,feat)

        feature_vector = np.array(
            [row[feat] for feat in ALL_FEATURES],
            dtype=np.float64,
        ).reshape(1, -1)

        return feature_vector

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:

        X = self._prepare_input(data)

        print("\n")
        print("PREDICTOR FEATURE COUNT")
        print(X.shape)

        log_cases = float(self.regressor.predict(X)[0])
        log_cases = max(log_cases, 0.0)               
        cases     = int(round(np.expm1(log_cases)))    

        outbreak_prob = float(self.classifier.predict_proba(X)[0][1])
        outbreak = bool(outbreak_prob >= self.outbreak_threshold)
        risk_level    = _get_risk_level(outbreak_prob)

        result = {
            "predicted_log_cases"  : round(log_cases, 4),
            "predicted_cases"      : cases,
            "outbreak_probability" : round(outbreak_prob, 4),
            "outbreak"             : outbreak,
            "risk_level"           : risk_level,
        }

        logger.info(
            f"Prediction → cases: {cases} | "
            f"outbreak: {outbreak} ({outbreak_prob:.2%}) | "
            f"risk: {risk_level}"
        )
        return result

    def predict_cases(self, data: Dict[str, Any]) -> int:
        return self.predict(data)["predicted_cases"]

    def predict_outbreak(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = self.predict(data)
        return {
            "outbreak_probability" : result["outbreak_probability"],
            "outbreak"             : result["outbreak"],
            "risk_level"           : result["risk_level"],
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "regressor"  : type(self.regressor).__name__,
            "classifier" : type(self.classifier).__name__,
            "features"   : len(ALL_FEATURES),
            "feature_list": ALL_FEATURES,
            "known_states"   : list(self.encoders.get("state_ut", {}).classes_ if "state_ut" in self.encoders else []),
            "known_diseases" : list(self.encoders.get("disease", {}).classes_  if "disease"  in self.encoders else []),
        }

def predict_cases(data: Dict[str, Any]) -> int:

    return DiseasePredictor().predict_cases(data)


def predict_outbreak(data: Dict[str, Any]) -> Dict[str, Any]:

    return DiseasePredictor().predict_outbreak(data)

if __name__ == "__main__":
    sample = {
        "state_ut"         : "Jharkhand",
        "district"         : "Ranchi",
        "disease"          : "Dengue",
        "disease_category" : "Vector-borne",
        "month"            : 8,
        "year"             : 2024,
        "day"              : 1,
        "temp"             : 34.0,
        "precipitation"    : 210.0,
        "lai"              : 2.5,
        "latitude"         : 23.35,
        "longitude"        : 85.33,
        "lag_1"            : 45.0,
        "lag_2"            : 38.0,
        "lag_3"            : 30.0,
        "lag_6"            : 25.0,
        "lag_12"           : 20.0,
        "roll_3_mean"      : 37.6,
        "roll_6_mean"      : 29.5,
        "roll_3_std"       : 7.5,
        "roll_6_std"       : 9.2,
        "growth_rate"      : 0.18,
        "spike_ratio"      : 1.5,
        "outbreak_flag"    : 1,
        "outbreak_frequency_12m": 3.0,
    }

    predictor = DiseasePredictor()
    result    = predictor.predict(sample)

    print("PREDICTION RESULT")

    for k, v in result.items():
        print(f"  {k:<28}: {v}")

    print("\nModel Info:")
    info = predictor.get_model_info()
    print(f"  Regressor  : {info['regressor']}")
    print(f"  Classifier : {info['classifier']}")
    print(f"  Features   : {info['features']}")
