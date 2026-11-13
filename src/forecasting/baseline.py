
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

DATA_PATH = Path("data/processed/delhi_multimodal_clean.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("data/processed")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_HOURS = 6
LAGS = [0, 1, 3, 6, 12, 24]

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "boundary_layer_height",
]


# --------------------------------------------------
# LOAD AND STANDARDIZE DATA
# --------------------------------------------------

print("=" * 60)
print("DIWALI PLUME - 6 HOUR PM2.5 BASELINE")
print("=" * 60)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Support common column-name variations.
aliases = {
    "station_id": ["station_id", "location_id"],
    "timestamp": ["timestamp", "hour", "timestamp_ist"],
    "pm25": ["pm25", "pm2.5", "value_pm25"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon"],
}

for standard, candidates in aliases.items():
    if standard not in df.columns:
        for candidate in candidates:
            if candidate in df.columns:
                df.rename(columns={candidate: standard}, inplace=True)
                break

required = ["station_id", "timestamp", "pm25", "latitude", "longitude"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")

for col in ["latitude", "longitude"] + WEATHER_FEATURES:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["station_id", "timestamp", "pm25"])
df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

# One station-hour should have one observation.
df = df.drop_duplicates(["station_id", "timestamp"], keep="first")

print(f"Loaded rows: {len(df):,}")
print(f"Stations: {df['station_id'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# --------------------------------------------------
# BUILD EXACT-TIME LAGS
# --------------------------------------------------

keys = ["station_id", "timestamp"]

data = df[keys + ["pm25"]].copy()

for lag in LAGS:
    if lag == 0:
        data["pm25_lag_0"] = data["pm25"]
        continue

    lagged = df[keys + ["pm25"]].copy()
    lagged["timestamp"] += pd.Timedelta(hours=lag)
    lagged.rename(columns={"pm25": f"pm25_lag_{lag}"}, inplace=True)

    data = data.merge(
        lagged,
        on=keys,
        how="left",
        validate="one_to_one"
    )

# Target must be exactly six hours in the future.
future = df[keys + ["pm25"]].copy()
future["timestamp"] -= pd.Timedelta(hours=FORECAST_HOURS)
future.rename(columns={"pm25": "target_pm25"}, inplace=True)

data = data.merge(
    future,
    on=keys,
    how="left",
    validate="one_to_one"
)

# Add static station coordinates and weather at prediction time.
extra_features = ["latitude", "longitude"] + [
    c for c in WEATHER_FEATURES if c in df.columns
]

metadata = df[keys + extra_features].copy()

data = data.merge(
    metadata,
    on=keys,
    how="left",
    validate="one_to_one"
)

# Cyclical time features.
hour = data["timestamp"].dt.hour
day = data["timestamp"].dt.day_of_year

data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * hour / 24)
data["day_sin"] = np.sin(2 * np.pi * day / 365.25)
data["day_cos"] = np.cos(2 * np.pi * day / 365.25)

feature_cols = (
    [f"pm25_lag_{lag}" for lag in LAGS]
    + ["latitude", "longitude"]
    + [c for c in WEATHER_FEATURES if c in data.columns]
    + ["hour_sin", "hour_cos", "day_sin", "day_cos"]
)

# Keep rows with a valid target and current PM2.5.
data = data.dropna(subset=["target_pm25", "pm25_lag_0"])

print(f"\nUsable rows after feature creation: {len(data):,}")


# --------------------------------------------------
# CHRONOLOGICAL SPLIT
# --------------------------------------------------

# Train: 2022 + January-August 2025
# Validation: September-December 2025
# Test: January-October 2026
#
# 2024 is excluded because the downloaded data has
# only one station for that year.

ts = data["timestamp"]

train_mask = (
    (ts.dt.year == 2022)
    | (
        (ts.dt.year == 2025)
        & (ts.dt.month <= 8)
    )
)

valid_mask = (
    (ts.dt.year == 2025)
    & (ts.dt.month >= 9)
)

test_mask = (
    (ts.dt.year == 2026)
    & (ts.dt.month <= 10)
)

train = data.loc[train_mask].copy()
valid = data.loc[valid_mask].copy()
test = data.loc[test_mask].copy()

if train.empty or valid.empty or test.empty:
    raise ValueError(
        "One or more data splits are empty. "
        "Check timestamp coverage in the processed dataset."
    )

X_train = train[feature_cols]
y_train = train["target_pm25"]

X_valid = valid[feature_cols]
y_valid = valid["target_pm25"]

X_test = test[feature_cols]
y_test = test["target_pm25"]

print("\nDATA SPLITS")
print(f"Train:      {len(train):,}")
print(f"Validation: {len(valid):,}")
print(f"Test:       {len(test):,}")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

def evaluate(y_true, predictions, model_name):
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)

    print(f"\n{model_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2:   {r2:.4f}")

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


# --------------------------------------------------
# PERSISTENCE BASELINE
# --------------------------------------------------

# Predict that PM2.5 remains at its current value.
persistence_predictions = X_test["pm25_lag_0"].to_numpy()

results = []
results.append(
    evaluate(y_test, persistence_predictions, "Persistence")
)


# --------------------------------------------------
# XGBOOST WITH EARLY STOPPING
# --------------------------------------------------

print("\nTraining XGBoost...")

model = XGBRegressor(
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    objective="reg:squarederror",
    eval_metric="mae",
    early_stopping_rounds=50,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=50,
)

print(f"\nBest iteration: {model.best_iteration}")

xgb_predictions = model.predict(X_test)

results.append(
    evaluate(y_test, xgb_predictions, "XGBoost")
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

joblib.dump(model, MODEL_DIR / "xgboost_pm25_6h.pkl")

predictions = test[
    ["station_id", "timestamp", "target_pm25"]
].copy()

predictions.rename(
    columns={"target_pm25": "actual_pm25"},
    inplace=True
)

predictions["persistence_prediction"] = persistence_predictions
predictions["xgboost_prediction"] = xgb_predictions

predictions.to_csv(
    OUTPUT_DIR / "test_predictions_6h.csv",
    index=False
)

metrics = pd.DataFrame(results)
metrics.to_csv(
    OUTPUT_DIR / "baseline_metrics_6h.csv",
    index=False
)

with open(MODEL_DIR / "baseline_features.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print("\n" + "=" * 60)
print("BASELINE EXPERIMENT COMPLETED")
print("=" * 60)
print(metrics.to_string(index=False))
print("\nSaved:")
print("- models/xgboost_pm25_6h.pkl")
print("- models/baseline_features.json")
print("- data/processed/test_predictions_6h.csv")
print("- data/processed/baseline_metrics_6h.csv")
