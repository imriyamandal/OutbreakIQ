import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR   = PROJECT_ROOT / "ml" / "reports"
METRICS_DIR   = REPORTS_DIR / "metrics"

for d in [MODELS_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

XGB_PARAMS = {
    "n_estimators"      : 500,
    "max_depth"         : 6,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "min_child_weight"  : 3,
    "reg_alpha"         : 0.1,       # L1
    "reg_lambda"        : 1.0,       # L2
    "objective"         : "reg:squarederror",
    "eval_metric"       : "rmse",
    "random_state"      : 42,
    "n_jobs"            : -1,
    "early_stopping_rounds": 30,
}

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.array(y_true)
    mask   = y_true != 0
    return (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0 else np.nan
    )

def save_feature_importance(model, feature_names: list, prefix: str = "xgb"):

    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    csv_path = REPORTS_DIR / f"{prefix}_feature_importance.csv"
    fi_df.to_csv(csv_path, index=False)
    logger.info(f"Feature importance CSV saved: {csv_path}")

    top = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2196F3")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top 20 Feature Importances ({prefix.upper()} Regressor)")
    plt.tight_layout()

    png_path = REPORTS_DIR / f"{prefix}_feature_importance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {png_path}")

    return fi_df

def train_regressor(
    X_train, y_train,
    X_val,   y_val,
    feature_names: list,
):

    logger.info("=" * 50)
    logger.info("TRAINING REGRESSION MODEL")
    logger.info(f"  Train samples : {len(X_train):,}")
    logger.info(f"  Val samples   : {len(X_val):,}")
    logger.info(f"  Features      : {len(feature_names)}")
    logger.info("=" * 50)

    logger.info("Fitting XGBoost Regressor ...")
    xgb_model = XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_pred = xgb_model.predict(X_val)
    mae  = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    r2   = r2_score(y_val, val_pred)
    mape = mean_absolute_percentage_error(y_val, val_pred)

    logger.info("XGBoost Validation Metrics:")
    logger.info(f"  MAE  : {mae:.4f}")
    logger.info(f"  RMSE : {rmse:.4f}")
    logger.info(f"  R²   : {r2:.4f}")
    logger.info(f"  MAPE : {mape:.2f}%")

    logger.info("Fitting Random Forest baseline ...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_val)
    rf_r2   = r2_score(y_val, rf_pred)
    logger.info(f"  RF Baseline R²: {rf_r2:.4f}")

    best_model = xgb_model if r2 >= rf_r2 else rf_model
    best_name  = "XGBoost" if r2 >= rf_r2 else "RandomForest"
    logger.info(f"  Best model selected: {best_name}")

    metrics = {
        "model"  : ["XGBoost",    "RandomForest"],
        "split"  : ["validation", "validation"],
        "MAE"    : [round(mae, 4),  round(mean_absolute_error(y_val, rf_pred), 4)],
        "RMSE"   : [round(rmse, 4), round(np.sqrt(mean_squared_error(y_val, rf_pred)), 4)],
        "R2"     : [round(r2, 4),   round(rf_r2, 4)],
        "MAPE"   : [round(mape, 2), round(mean_absolute_percentage_error(y_val, rf_pred), 2)],
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_path = METRICS_DIR / "xgb_regression_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Regression metrics saved: {metrics_path}")

    save_feature_importance(xgb_model, feature_names, prefix="xgb_reg")

    model_path = MODELS_DIR / "regressor.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Regressor saved: {model_path}")

    return best_model

if __name__ == "__main__":
    from training.models.train import load_and_prepare, temporal_split, FINAL_DATA_PATH, ALL_FEATURES

    df = load_and_prepare(FINAL_DATA_PATH)
    (
        X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        feature_names,
    ) = temporal_split(df)

    train_regressor(X_train, y_reg_train, X_val, y_reg_val, feature_names)