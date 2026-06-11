import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

METRICS_DIR = PROJECT_ROOT / "ml/reports/metrics"

REPORT_DIR = PROJECT_ROOT / "ml/reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

forecasting_files = {
    "XGBoost Regression": METRICS_DIR / "xgb_regression_metrics.csv",
    "LSTM": METRICS_DIR / "lstm_metrics.csv",
    "Prophet": METRICS_DIR / "prophet_metrics.csv",
    "ARIMA": METRICS_DIR / "arima_metrics.csv"
}

forecast_rows = []

for model_name, file_path in forecasting_files.items():

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        continue

    df = pd.read_csv(file_path)

    forecast_rows.append({
        "Model": model_name,
        "RMSE": df["RMSE"].iloc[0],
        "R2": df["R2"].iloc[0]
    })

forecast_df = pd.DataFrame(forecast_rows)

if not forecast_df.empty:

    forecast_df = forecast_df.sort_values(
        by="RMSE",
        ascending=True
    )

    forecast_df.to_csv(
        REPORT_DIR / "forecasting_model_comparison.csv",
        index=False
    )

classification_file = (
    METRICS_DIR / "classification_metrics.csv"
)

classification_df = pd.DataFrame()

if classification_file.exists():

    classification_df = pd.read_csv(
        classification_file
    )

    if "split" in classification_df.columns:

        classification_df = classification_df[
            classification_df["split"].str.lower() == "test"
        ]

    classification_df.to_csv(
        REPORT_DIR / "classification_model_comparison.csv",
        index=False
    )

with open(
    REPORT_DIR / "model_comparison.csv",
    "w",
    encoding="utf-8"
) as f:

    f.write("FORECASTING MODELS\n")

    if not forecast_df.empty:
        forecast_df.to_csv(
            f,
            index=False
        )

    f.write("\n\n")

    f.write("CLASSIFICATION MODELS\n")

    if not classification_df.empty:
        classification_df.to_csv(
            f,
            index=False
        )

if not forecast_df.empty:

    plt.figure(figsize=(10, 6))

    plt.bar(
        forecast_df["Model"],
        forecast_df["RMSE"]
    )

    plt.title(
        "Forecasting Models Comparison (RMSE)"
    )

    plt.xlabel("Model")
    plt.ylabel("RMSE")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "forecasting_model_comparison.png",
        dpi=300
    )

    plt.close()
if not classification_df.empty:

    metric_columns = [
        col
        for col in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1"
        ]
        if col in classification_df.columns
    ]

    if metric_columns:

        row = classification_df.iloc[0]

        values = [
            row[col]
            for col in metric_columns
        ]

        plt.figure(figsize=(8, 5))

        plt.bar(
            metric_columns,
            values
        )

        plt.title(
            f"Classification Metrics ({row['model']})"
        )

        plt.ylabel("Score")

        plt.ylim(0, 1.05)

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / "classification_model_comparison.png",
            dpi=300
        )

        plt.close()

print("\n===== FORECASTING MODELS =====")

if not forecast_df.empty:
    print(forecast_df)

print("\n===== CLASSIFICATION MODELS =====")

if not classification_df.empty:
    print(classification_df)

print("\nReports generated successfully.")