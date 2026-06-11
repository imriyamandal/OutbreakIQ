import pandas as pd

from prophet import Prophet

from sklearn.metrics import (
    root_mean_squared_error,
    r2_score
)

from pathlib import Path

from training.models.train import (
    load_and_prepare,
    FINAL_DATA_PATH
)

PROJECT_ROOT=Path(__file__).resolve().parents[2]

REPORT_DIR=PROJECT_ROOT/"ml/reports/metrics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df=load_and_prepare(FINAL_DATA_PATH)

prophet_df=df[
    ["date","log_cases"]
].rename(
    columns={
        "date":"ds",
        "log_cases":"y"
    }
)

split=int(len(prophet_df)*0.8)

train=prophet_df.iloc[:split]

test=prophet_df.iloc[split:]

model=Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(train)

future=model.make_future_dataframe(
    periods=len(test),
    freq="D"
)

forecast=model.predict(future)

pred = forecast["yhat"].iloc[-len(test):]
rmse = root_mean_squared_error(
    test["y"],
    pred
)

r2 = r2_score(
    test["y"],
    pred
)

pd.DataFrame(
    [{
        "Model":"Prophet",
        "RMSE":rmse,
        "R2":r2
    }]
).to_csv(
    REPORT_DIR/"prophet_metrics.csv",
    index=False
)

print("Prophet completed")