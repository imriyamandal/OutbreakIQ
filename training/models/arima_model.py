import pandas as pd

from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    root_mean_squared_error,
    r2_score
)

from training.models.train import (
    load_and_prepare,
    FINAL_DATA_PATH
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "ml/reports/metrics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = load_and_prepare(
    FINAL_DATA_PATH
)

series = df["log_cases"]

split = int(len(series) * 0.8)

train = series[:split]
test = series[split:]

model = ARIMA(
    train,
    order=(2, 1, 2)
)

fit = model.fit()

pred = fit.forecast(
    steps=len(test)
)

rmse = root_mean_squared_error(
    test,
    pred
)

r2 = r2_score(
    test,
    pred
)

pd.DataFrame(
    [{
        "Model": "ARIMA",
        "RMSE": rmse,
        "R2": r2
    }]
).to_csv(
    REPORT_DIR / "arima_metrics.csv",
    index=False
)

print("ARIMA completed")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")