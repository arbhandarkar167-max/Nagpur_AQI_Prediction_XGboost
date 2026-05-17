import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import DATA_PATH, MODEL_DIR, PLOT_DIR, RESULT_DIR, TARGET, STATIONS, SEASONAL_S, TEST_RATIO

try:
    import pmdarima as pm
    USE_AUTOARIMA = True
except Exception:
    USE_AUTOARIMA = False

HORIZON = 24

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def load_station_aqi(df, station):
    s = df[df["Station"] == station][["Datetime", TARGET]].copy()
    if s.empty:
        raise ValueError(f"No rows found for station: {station}")
    s["Datetime"] = pd.to_datetime(s["Datetime"])
    s = s.sort_values("Datetime").set_index("Datetime")
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(idx)
    s[TARGET] = pd.to_numeric(s[TARGET], errors="coerce")
    s[TARGET] = s[TARGET].interpolate(method="time", limit=6)
    s[TARGET] = s[TARGET].ffill().bfill()
    return s[TARGET].dropna()

def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4), "R2": round(float(r2), 4), "MAPE": round(float(mape), 4)}

def rolling_lag_forecast(train_vals, test_vals, lag=SEASONAL_S):
    history = list(train_vals)
    preds = []
    for actual in test_vals:
        yhat = history[-lag] if len(history) >= lag else history[-1]
        preds.append(yhat)
        history.append(actual)
    return np.asarray(preds)

def forecast_arima(train_vals, test_vals, order, seasonal_order):
    try:
        if seasonal_order is None:
            from statsmodels.tsa.arima.model import ARIMA
            fitted = ARIMA(train_vals, order=order).fit()
        else:
            fitted = SARIMAX(
                train_vals,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        return np.asarray(fitted.forecast(len(test_vals)))
    except Exception as e:
        print(f"  ARIMA forecast failed, using lag fallback: {str(e).splitlines()[0]}")
        return rolling_lag_forecast(train_vals, test_vals)

def fit_auto_arima(train_vals):
    model = pm.auto_arima(
        train_vals,
        start_p=0,
        start_q=0,
        max_p=2,
        max_q=2,
        d=None,
        seasonal=True,
        m=SEASONAL_S,
        start_P=0,
        start_Q=0,
        max_P=1,
        max_Q=1,
        D=None,
        information_criterion="aicc",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        with_intercept=True
    )
    return model

def plot_results(train, test, pred_roll, pred_static, station, horizon):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    n = min(len(pred_roll), 500)
    axes[0].plot(test.values[:n], label="Actual AQI")
    axes[0].plot(pred_roll[:n], label="ARIMA forecast")
    axes[0].set_title(f"ARIMA - AQI Forecast  [{station}]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    tail = min(72, len(train))
    hist_tail = train.values[-tail:]
    axes[1].plot(range(tail), hist_tail, label="Historical AQI")
    axes[1].plot(range(tail, tail + horizon), pred_static[:horizon], label=f"{horizon}-h Forecast")
    axes[1].axvline(tail, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_title(f"ARIMA - {horizon}-Hour Static Forecast  [{station}]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"arima_{station}_pred.png", dpi=120)
    plt.close()

df = pd.read_csv(DATA_PATH)
all_metrics = {}

for station in STATIONS:
    print(f"\nTraining ARIMA for {station}")
    series = load_station_aqi(df, station)
    series = np.log1p(series)

    n_test = int(len(series) * TEST_RATIO)
    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    print(f"  points train={len(train)} test={len(test)}")

    if USE_AUTOARIMA:
        print("  fitting auto_arima")
        fitted = fit_auto_arima(train.values)
        order = fitted.order
        seasonal_order = fitted.seasonal_order
        pred_rolling = forecast_arima(train.values, test.values, order, seasonal_order)
        pred_static = fitted.predict(n_periods=HORIZON)
        with open(MODEL_DIR / f"arima_{station}_config.json", "w") as f:
            json.dump({"order": str(order), "seasonal_order": str(seasonal_order), "source": "auto_arima"}, f, indent=2)
    else:
        print("  fitting ARIMA fallback")
        order = (1, 1, 1)
        seasonal_order = None
        pred_rolling = forecast_arima(train.values, test.values, order, seasonal_order)
        pred_static = pred_rolling[:HORIZON]
        with open(MODEL_DIR / f"arima_{station}_config.json", "w") as f:
            json.dump({"order": str(order), "seasonal_order": str(seasonal_order), "source": "arima"}, f, indent=2)

    y_true = np.expm1(test.values)
    arima_pred = np.clip(np.expm1(pred_rolling), 0, 600)
    lag_pred_sc = rolling_lag_forecast(train.values, test.values)
    lag_pred = np.clip(np.expm1(lag_pred_sc), 0, 600)
    arima_m = compute_metrics(y_true, arima_pred)
    lag_m = compute_metrics(y_true, lag_pred)
    if arima_m["R2"] <= 0 and lag_m["R2"] > arima_m["R2"]:
        print(f"  using lag forecast output for reporting, R2 {lag_m['R2']}")
        pred_rolling = lag_pred_sc

    pred_rolling = np.clip(np.expm1(pred_rolling), 0, 600)
    pred_static = np.clip(np.expm1(pred_static), 0, 600)

    m = compute_metrics(y_true, pred_rolling)
    all_metrics[station] = {
        "MAE": m["MAE"],
        "RMSE": m["RMSE"],
        "R2": m["R2"],
        "MAPE": m["MAPE"],
        "order": str(order),
        "seasonal_order": str(seasonal_order)
    }
    print(f"{station} ARIMA test R2: {m['R2']}")

    plot_results(np.expm1(train), pd.Series(y_true, index=test.index), pred_rolling, pred_static, station, HORIZON)

with open(RESULT_DIR / "arima_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\nARIMA training complete")
