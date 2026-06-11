import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    # Regression
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    # Classification
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

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

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask   = y_true != 0
    return (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        if mask.sum() > 0 else np.nan
    )

def evaluate_regressor(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    split: str = "test",
) -> dict:

    logger.info(f"Evaluating regressor on [{split}] set ...")

    y_pred = model.predict(X)

    mae_val  = mean_absolute_error(y_true, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_val   = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)

    metrics = {
        "model" : [type(model).__name__],
        "split" : [split],
        "MAE"   : [round(mae_val,  4)],
        "RMSE"  : [round(rmse_val, 4)],
        "R2"    : [round(r2_val,   4)],
        "MAPE"  : [round(mape_val, 2)],
    }
    metrics_df = pd.DataFrame(metrics)

    metrics_path = METRICS_DIR / "xgb_regression_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    logger.info(f"  MAE  : {mae_val:.4f}")
    logger.info(f"  RMSE : {rmse_val:.4f}")
    logger.info(f"  R²   : {r2_val:.4f}")
    logger.info(f"  MAPE : {mape_val:.2f}%")

    _plot_actual_vs_predicted(y_true, y_pred, split)

    _plot_residuals(y_true, y_pred, split)

    return {"MAE": mae_val, "RMSE": rmse_val, "R2": r2_val, "MAPE": mape_val}


def _plot_actual_vs_predicted(y_true, y_pred, split: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="#2196F3", label="Predictions")
    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.5, label="Perfect fit")
    ax.set_xlabel("Actual log_cases")
    ax.set_ylabel("Predicted log_cases")
    ax.set_title(f"Actual vs Predicted – {split.capitalize()} Set")
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / f"actual_vs_predicted_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Actual vs Predicted plot saved: {path}")


def _plot_residuals(y_true, y_pred, split: str):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="#FF5722")
    axes[0].axhline(0, color="black", lw=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted log_cases")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs Predicted – {split.capitalize()}")

    axes[1].hist(residuals, bins=40, color="#9C27B0", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution – {split.capitalize()}")

    plt.tight_layout()
    path = REPORTS_DIR / f"residuals_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Residuals plot saved: {path}")

def evaluate_classifier(
    model,
    X: np.ndarray,
    y_true: np.ndarray,
    split: str = "test",
) -> dict:

    logger.info(f"Evaluating classifier on [{split}] set ...")

    y_pred      = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    acc_val  = accuracy_score(y_true, y_pred)
    prec_val = precision_score(y_true, y_pred, zero_division=0)
    rec_val  = recall_score(y_true, y_pred, zero_division=0)
    f1_val   = f1_score(y_true, y_pred, zero_division=0)
    auc_val  = roc_auc_score(y_true, y_pred_prob)
    ap_val   = average_precision_score(y_true, y_pred_prob)

    metrics = {
        "model"         : [type(model).__name__],
        "split"         : [split],
        "Accuracy"      : [round(acc_val,  4)],
        "Precision"     : [round(prec_val, 4)],
        "Recall"        : [round(rec_val,  4)],
        "F1"            : [round(f1_val,   4)],
        "ROC_AUC"       : [round(auc_val,  4)],
        "Avg_Precision" : [round(ap_val,   4)],
    }
    metrics_df = pd.DataFrame(metrics)

    metrics_path = METRICS_DIR / "classification_metrics.csv"
    if metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        metrics_df = pd.concat([existing, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)

    logger.info(f"  Accuracy       : {acc_val:.4f}")
    logger.info(f"  Precision      : {prec_val:.4f}")
    logger.info(f"  Recall         : {rec_val:.4f}")
    logger.info(f"  F1 Score       : {f1_val:.4f}")
    logger.info(f"  ROC-AUC        : {auc_val:.4f}")
    logger.info(f"  Avg Precision  : {ap_val:.4f}")
    logger.info(
        "\n" + classification_report(
            y_true, y_pred,
            target_names=["No Outbreak", "Outbreak"],
        )
    )
    _plot_confusion_matrix(y_true, y_pred, split)

    _plot_roc_curve(y_true, y_pred_prob, auc_val, split)

    _plot_pr_curve(y_true, y_pred_prob, ap_val, split)

    return {
        "Accuracy"  : acc_val,
        "Precision" : prec_val,
        "Recall"    : rec_val,
        "F1"        : f1_val,
        "ROC_AUC"   : auc_val,
    }

def _plot_confusion_matrix(y_true, y_pred, split: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Outbreak", "Outbreak"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {split.capitalize()} Set")
    plt.tight_layout()
    path = REPORTS_DIR / f"confusion_matrix_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {path}")


def _plot_roc_curve(y_true, y_prob, auc_val: float, split: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"ROC curve (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E91E63")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {split.capitalize()} Set")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = REPORTS_DIR / f"roc_curve_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"ROC curve saved: {path}")


def _plot_pr_curve(y_true, y_prob, ap_val: float, split: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color="#4CAF50", lw=2,
            label=f"PR curve (AP = {ap_val:.3f})")
    ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve – {split.capitalize()} Set")
    ax.legend()
    plt.tight_layout()
    path = REPORTS_DIR / f"precision_recall_curve_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Precision-Recall curve saved: {path}")

if __name__ == "__main__":
    from training.models.train import load_and_prepare, temporal_split, FINAL_DATA_PATH

    df = load_and_prepare(FINAL_DATA_PATH)
    (
        X_train, X_val, X_test,
        y_reg_train, y_reg_val, y_reg_test,
        y_cls_train, y_cls_val, y_cls_test,
        feature_names,
    ) = temporal_split(df)

    reg_path = PROJECT_ROOT / "ml" / "models" / "regressor.pkl"
    cls_path = PROJECT_ROOT / "ml" / "models" / "classifier.pkl"

    if reg_path.exists():
        reg_model = joblib.load(reg_path)
        evaluate_regressor(reg_model, X_test, y_reg_test, split="test")
    else:
        logger.warning(f"Regressor not found at {reg_path}. Run train.py first.")

    if cls_path.exists():
        cls_model = joblib.load(cls_path)
        evaluate_classifier(cls_model, X_test, y_cls_test, split="test")
    else:
        logger.warning(f"Classifier not found at {cls_path}. Run train.py first.")