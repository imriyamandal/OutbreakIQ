import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score
)
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import ( # pyright: ignore[reportMissingModuleSource]
    LSTM,
    Dense,
    Dropout
)

from training.models.train import (
    load_and_prepare,
    FINAL_DATA_PATH
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "ml/reports/metrics"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = load_and_prepare(
    FINAL_DATA_PATH
)

series = df["log_cases"].values.reshape(-1, 1)
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(series)
LOOKBACK = 14

X = []
y = []

for i in range(LOOKBACK, len(scaled_data)):
    X.append(
        scaled_data[i - LOOKBACK:i, 0]
    )
    y.append(
        scaled_data[i, 0]
    )

X = np.array(X)
y = np.array(y)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    1
)

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(LOOKBACK, 1)
    ),

    Dropout(0.2),

    LSTM(32),

    Dropout(0.2),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)
model.save(
    MODEL_DIR / "lstm.h5"
)
pred = model.predict(X_test)

pred = scaler.inverse_transform(pred)

y_test_actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

rmse = root_mean_squared_error(
    y_test_actual,
    pred
)

r2 = r2_score(
    y_test_actual,
    pred
)

metrics_df = pd.DataFrame([
    {
        "Model": "LSTM",
        "RMSE": rmse,
        "R2": r2
    }
])

metrics_df.to_csv(
    REPORT_DIR / "lstm_metrics.csv",
    index=False
)

print("\nLSTM completed")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")
print(f"Model saved to: {MODEL_DIR/'lstm.h5'}")
print(f"Metrics saved to: {REPORT_DIR/'lstm_metrics.csv'}")