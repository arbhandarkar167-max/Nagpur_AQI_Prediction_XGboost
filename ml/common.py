from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "data" / "nagpur_final_preprocessed.csv"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
RESULT_DIR = BASE_DIR / "results"

FEATURES = [
    "PM2.5", "PM10", "NO", "NO2", "SO2", "NH3",
    "Hour_sin", "Hour_cos", "Month_sin", "Month_cos",
    "DOW_sin", "DOW_cos", "IsWeekend"
]

TARGET = "AQI"
STATIONS = ["Ambazari", "Mahal", "Civil_Lines", "Ram_Nagar"]
TEST_RATIO = 0.15
VAL_RATIO = 0.15
SEASONAL_S = 24

def load_station(df, station):
    s = df[df["Station"] == station].copy()
    if s.empty:
        raise ValueError(f"No rows found for station: {station}")
    s["Datetime"] = pd.to_datetime(s["Datetime"])
    s = s.sort_values("Datetime").set_index("Datetime")
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(idx)
    missing = [c for c in FEATURES + [TARGET] if c not in s.columns]
    if missing:
        raise ValueError(f"Missing columns for {station}: {missing}")
    cols = FEATURES + [TARGET]
    s[cols] = s[cols].apply(pd.to_numeric, errors="coerce")
    s[cols] = s[cols].interpolate(method="time", limit=6)
    s[cols] = s[cols].ffill().bfill()
    return s.dropna(subset=cols)

def make_sequences(feat, tgt, window, horizon):
    X, y = [], []
    for i in range(len(feat) - window - horizon + 1):
        X.append(feat[i:i + window])
        y.append(tgt[i + window:i + window + horizon])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def train_val_test_split(X, y, val_r=VAL_RATIO, test_r=TEST_RATIO):
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough sequences to split: {n}")
    nt = int(n * test_r)
    nv = int(n * val_r)
    if n - nt - nv <= 0 or nt <= 0 or nv <= 0:
        raise ValueError(f"Invalid split sizes for {n} sequences")
    return (
        X[:n - nt - nv], y[:n - nt - nv],
        X[n - nt - nv:n - nt], y[n - nt - nv:n - nt],
        X[n - nt:], y[n - nt:]
    )

def fit_station_scalers(s):
    fs = MinMaxScaler()
    ts = MinMaxScaler()
    split = int(len(s) * (1 - VAL_RATIO - TEST_RATIO))
    if split <= 0:
        raise ValueError("Training split is empty")
    fs.fit(s.iloc[:split][FEATURES].values)
    ts.fit(np.log1p(s.iloc[:split][[TARGET]].values))
    Xd = fs.transform(s[FEATURES].values)
    yd = ts.transform(np.log1p(s[[TARGET]].values)).ravel()
    return fs, ts, Xd, yd

def inv_log_target(ts, arr):
    return np.expm1(ts.inverse_transform(np.asarray(arr).reshape(-1, 1)).ravel())

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 4)
    }
