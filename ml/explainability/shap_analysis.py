import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from training.models.train import (
    load_and_prepare,
    temporal_split,
    FINAL_DATA_PATH
)

PROJECT_ROOT=Path(__file__).resolve().parents[2]

MODEL_PATH=PROJECT_ROOT/"ml/models/classifier.pkl"
REPORTS_DIR=PROJECT_ROOT/"ml/reports"

df=load_and_prepare(FINAL_DATA_PATH)

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
)=temporal_split(df)

model=joblib.load(MODEL_PATH)

X_test_df=pd.DataFrame(
    X_test,
    columns=feature_names
)

explainer=shap.TreeExplainer(model)

shap_values=explainer.shap_values(X_test_df)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_df,
    show=False
)

plt.savefig(
    REPORTS_DIR/"shap_summary.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()

top_feature=feature_names[0]

plt.figure()

shap.dependence_plot(
    top_feature,
    shap_values,
    X_test_df,
    show=False
)

plt.savefig(
    REPORTS_DIR/"shap_dependence.png",
    bbox_inches="tight",
    dpi=300
)

plt.close()

print("SHAP reports generated")
@router.get("/feature-importance")
def feature_importance():

    return {
        "image":
        "ml/reports/shap_summary.png"
    }