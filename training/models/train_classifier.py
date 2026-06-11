import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR   = PROJECT_ROOT / "ml" / "models"
REPORTS_DIR  = PROJECT_ROOT / "ml" / "reports"
METRICS_DIR  = REPORTS_DIR / "metrics"

for d in [MODELS_DIR, METRICS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def compute_scale_pos_weight(y_train) -> float:
    neg = np.sum(y_train == 0)
    pos = np.sum(y_train == 1)
    ratio = neg / pos if pos > 0 else 1.0
    logger.info(
        f"Class balance – Neg: {neg:,}  Pos: {pos:,}  "
        f"scale_pos_weight: {ratio:.2f}"
    )
    return ratio


def save_confusion_matrix(y_true, y_pred, title: str, filename: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Outbreak", "Outbreak"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {path}")


def save_feature_importance(model, feature_names: list, prefix: str = "xgb_cls"):
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
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#E91E63")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top 20 Feature Importances ({prefix.upper()} Classifier)")
    plt.tight_layout()

    png_path = REPORTS_DIR / f"{prefix}_feature_importance.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info(f"Feature importance plot saved: {png_path}")

    return fi_df

def train_classifier(
    X_train, y_train,
    X_val,   y_val,
    feature_names: list,
):
    logger.info("TRAINING CLASSIFICATION MODEL")
    logger.info(f"  Train samples : {len(X_train):,}")
    logger.info(f"  Val samples   : {len(X_val):,}")
    logger.info(f"  Features      : {len(feature_names)}")

    smote = SMOTE(
    sampling_strategy="auto",
    random_state=42,
    k_neighbors=5
    )
    X_train,y_train = smote.fit_resample(X_train,y_train)
    spw = compute_scale_pos_weight(y_train)

    logger.info("Fitting XGBoost Classifier ...")
    xgb_model = XGBClassifier(
        n_estimators          = 500,
        max_depth             = 5,
        learning_rate         = 0.05,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 3,
        scale_pos_weight      = spw,          
        reg_alpha             = 0.1,
        reg_lambda            = 1.0,
        objective             = "binary:logistic",
        eval_metric           = "auc",
        use_label_encoder     = False,
        random_state          = 42,
        n_jobs                = -1,
        early_stopping_rounds = 30,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    val_pred      = xgb_model.predict(X_val)
    val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]

    acc       = accuracy_score(y_val, val_pred)
    prec      = precision_score(y_val, val_pred, zero_division=0)
    rec       = recall_score(y_val, val_pred, zero_division=0)
    f1        = f1_score(y_val, val_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_val, val_pred_prob)

    logger.info("XGBoost Validation Metrics:")
    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  F1 Score  : {f1:.4f}")
    logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
    logger.info("\n" + classification_report(y_val, val_pred, target_names=["No Outbreak", "Outbreak"]))

    logger.info("Fitting Random Forest baseline ...")
    rf_model = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 10,
        class_weight  = "balanced",
        min_samples_leaf = 5,
        random_state  = 42,
        n_jobs        = -1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred      = rf_model.predict(X_val)
    rf_pred_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_f1        = f1_score(y_val, rf_pred, zero_division=0)
    rf_auc       = roc_auc_score(y_val, rf_pred_prob)
    logger.info(f"  RF Baseline F1: {rf_f1:.4f}  AUC: {rf_auc:.4f}")

    best_model = xgb_model if roc_auc >= rf_auc else rf_model
    best_name  = "XGBoost" if roc_auc >= rf_auc else "RandomForest"
    logger.info(f"  Best model selected: {best_name}")

    metrics = {
        "model"     : ["XGBoost",    "RandomForest"],
        "split"     : ["validation", "validation"],
        "Accuracy"  : [round(acc, 4),  round(accuracy_score(y_val, rf_pred), 4)],
        "Precision" : [round(prec, 4), round(precision_score(y_val, rf_pred, zero_division=0), 4)],
        "Recall"    : [round(rec, 4),  round(recall_score(y_val, rf_pred, zero_division=0), 4)],
        "F1"        : [round(f1, 4),   round(rf_f1, 4)],
        "ROC_AUC"   : [round(roc_auc, 4), round(rf_auc, 4)],
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_path = METRICS_DIR / "classification_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Classification metrics saved: {metrics_path}")

    save_confusion_matrix(
        y_val, val_pred,
        title    = "XGBoost Classifier – Validation Set",
        filename = "confusion_matrix_validation.png",
    )

    save_feature_importance(xgb_model, feature_names, prefix="xgb_cls")

    model_path = MODELS_DIR / "classifier.pkl"
    joblib.dump(best_model, model_path)
    logger.info(f"Classifier saved: {model_path}")

    return best_model

if __name__ == "__main__":
    from training.models.train import load_and_prepare, temporal_split, FINAL_DATA_PATH

    df = load_and_prepare(FINAL_DATA_PATH)
    (
        X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        feature_names,
    ) = temporal_split(df)

    train_classifier(X_train, y_cls_train, X_val, y_cls_val, feature_names)