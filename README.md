# Nagpur AQI Forecasting System — XGBoost Edition

XGBoost is the primary model (R² = 0.88–0.91 across all 4 stations).

## Structure

```
nagpur_xgb/
├── data/
│   └── nagpur_final_preprocessed.csv
├── ml/
│   ├── train_xgboost.py          ← train + evaluate XGBoost
│   ├── models/                   ← saved .pkl models + scalers
│   ├── plots/                    ← 9-panel accuracy figures
│   └── results/
├── backend/
│   └── app.py                    ← Flask API (port 5000)
├── frontend/
│   ├── index.html
│   └── static/css/ js/
└── requirements.txt
```



## API Endpoints

| Method | URL | Returns |
|--------|-----|---------|
| GET | /api/stations | Station list |
| GET | /api/current | Latest AQI per station |
| GET | /api/forecast/<station> | 24-h XGBoost forecast |
| GET | /api/history/<station> | 7-day hourly AQI |
| GET | /api/aqi_trend | 30-day city trend |
| GET | /api/pollutants/<station> | Pollutant breakdown |
| GET | /api/health_advisory | Group-specific advice |
| GET | /api/model_metrics | Train/val/test metrics |
| GET | /api/feature_importance/<station> | XGBoost feature importance |
| POST | /api/predict | Custom AQI prediction |

## Dashboard Features

- AQI arc gauge with colour coding
- Leaflet map with live AQI-coloured station markers
- Pollutant bars vs NAAQS limits
- 7-day history + 24-h XGBoost forecast
- 30-day city trend
- XGBoost feature importance bar chart (per station)
- Group health advisories
- Custom AQI predictor with 24-h output chips
- 5-minute auto-refresh
