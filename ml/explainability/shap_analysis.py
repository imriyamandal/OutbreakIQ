import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))
from training.models.train import (
    load_and_prepare,
    temporal_split,
    FINAL_DATA_PATH
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "ml/models"
REPORTS_DIR = PROJECT_ROOT / "ml/reports"

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading data...")

df = load_and_prepare(
    FINAL_DATA_PATH
)

(
    X_train,
    X_val,
    X_test,
    y_reg_train,
    y_reg_val,
    y_reg_test,
    y_cls_train,
    y_cls_val,
    y_cls_test,
    feature_names,
) = temporal_split(df)

print("Loading model...")

model = joblib.load(
    MODELS_DIR / "regressor.pkl"
)

print("Generating SHAP values...")

explainer = shap.TreeExplainer(model)

sample_size = min(
    1000,
    len(X_test)
)

X_sample = X_test[:sample_size]

shap_values = explainer.shap_values(
    X_sample
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_summary.png")

top_feature = feature_names[
    abs(shap_values).mean(axis=0).argmax()
]

plt.figure()

shap.dependence_plot(
    top_feature,
    shap_values,
    X_sample,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "shap_dependence.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved: shap_dependence.png")
print("SHAP analysis completed.")