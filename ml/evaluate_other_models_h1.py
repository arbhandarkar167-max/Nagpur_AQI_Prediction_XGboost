import os
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA

from common import DATA_PATH, MODEL_DIR, RESULT_DIR, FEATURES, TARGET, STATIONS, load_station, make_sequences, train_val_test_split, fit_station_scalers, inv_log_target, compute_metrics

WINDOW_SIZE = 48
HORIZON = 24
N_STEPS = 8
N_LENGTH = 6

os.makedirs(RESULT_DIR, exist_ok=True)

def reshape_cnn_lstm(X):
    return X.reshape(X.shape[0], N_STEPS, N_LENGTH, X.shape[2])

def evaluate_arima_h1(s):
    series = np.log1p(s[TARGET].values.astype(float))
    n_test = int(len(series) * 0.15)
    n_val = int(len(series) * 0.15)
    val_end = len(series) - n_test
    train_val = series[:val_end]
    test = series[val_end:]
    try:
        fitted = ARIMA(train_val, order=(2, 1, 2)).fit()
        preds = []
        current = fitted
        for actual in test:
            preds.append(float(current.forecast(1)[0]))
            current = current.append([actual], refit=False)
        y_pred = np.clip(np.expm1(preds), 0, 600)
        y_true = np.expm1(test)
    except Exception:
        y_true = np.expm1(test)
        y_pred = np.expm1(series[val_end - 1:-1])
    return compute_metrics(y_true, y_pred)

def evaluate_gru_h1(s, station):
    _, ts, Xd, yd = fit_station_scalers(s)
    X_input = np.column_stack([Xd, yd])
    X, y = make_sequences(X_input, yd, WINDOW_SIZE, HORIZON)
    _, _, _, _, Xte, yte = train_val_test_split(X, y)
    model_path = MODEL_DIR / f"gru_aqi_{station}.keras"
    model = tf.keras.models.load_model(model_path)
    pred = model.predict(Xte, verbose=0)
    y_true = inv_log_target(ts, yte[:, 0])
    y_pred = inv_log_target(ts, pred[:, 0])
    return compute_metrics(y_true, y_pred)

def evaluate_cnn_lstm_h1(s, station):
    _, ts, Xd, yd = fit_station_scalers(s)
    X_input = np.column_stack([Xd, yd])
    X, y = make_sequences(X_input, yd, WINDOW_SIZE, HORIZON)
    X = reshape_cnn_lstm(X)
    _, _, _, _, Xte, yte = train_val_test_split(X, y)
    model_path = MODEL_DIR / f"cnnlstm_aqi_{station}.keras"
    model = tf.keras.models.load_model(model_path)
    pred = model.predict(Xte, verbose=0)
    y_true = inv_log_target(ts, yte[:, 0])
    y_pred = inv_log_target(ts, pred[:, 0])
    return compute_metrics(y_true, y_pred)

df = pd.read_csv(DATA_PATH)
all_metrics = {}

for station in STATIONS:
    print(f"Evaluating H1 metrics for {station}")
    s = load_station(df, station)
    all_metrics[station] = {
        "ARIMA": evaluate_arima_h1(s),
        "GRU": evaluate_gru_h1(s, station),
        "CNN-LSTM": evaluate_cnn_lstm_h1(s, station)
    }
    for model_name, score in all_metrics[station].items():
        print(f"  {model_name}: R2={score['R2']} MAE={score['MAE']} RMSE={score['RMSE']} MAPE={score['MAPE']}")

with open(RESULT_DIR / "other_model_h1_metrics.json", "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2)

print(f"Saved {RESULT_DIR / 'other_model_h1_metrics.json'}")
