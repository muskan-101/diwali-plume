
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT = Path("data/processed/test_predictions_6h.csv")
OUTPUT = Path("reports/error_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

MIN_STATION_SAMPLES = 100

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("=" * 60)
print("DIWALI PLUME - ERROR ANALYSIS")
print("=" * 60)

if not INPUT.exists():
    raise FileNotFoundError(f"Predictions not found: {INPUT}")

df = pd.read_csv(INPUT)

required = [
    "station_id",
    "timestamp",
    "actual_pm25",
    "persistence_prediction",
    "xgboost_prediction",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

for col in required[2:]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=required).copy()

df["month"] = df["timestamp"].dt.strftime("%Y-%m")

print(f"\nTest observations: {len(df):,}")
print(f"Stations: {df['station_id'].nunique()}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


# --------------------------------------------------
# METRIC FUNCTION
# --------------------------------------------------

def calculate_metrics(group):
    actual = group["actual_pm25"].to_numpy()

    result = {"samples": len(group)}

    for name, column in [
        ("persistence", "persistence_prediction"),
        ("xgboost", "xgboost_prediction"),
    ]:
        pred = group[column].to_numpy()

        result[f"{name}_MAE"] = mean_absolute_error(actual, pred)
        result[f"{name}_RMSE"] = np.sqrt(
            mean_squared_error(actual, pred)
        )
        result[f"{name}_bias"] = np.mean(pred - actual)

    return pd.Series(result)


# --------------------------------------------------
# 1. MONTH-WISE PERFORMANCE
# --------------------------------------------------

monthly = (
    df.groupby("month")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

monthly.to_csv(OUTPUT / "monthly_performance.csv", index=False)

print("\n" + "=" * 60)
print("1. MONTH-WISE PERFORMANCE")
print("=" * 60)

print(monthly.round(2).to_string(index=False))


# --------------------------------------------------
# 2. STATION-WISE PERFORMANCE
# --------------------------------------------------

station = (
    df.groupby("station_id")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

station.to_csv(OUTPUT / "station_performance.csv", index=False)

eligible = station[
    station["samples"] >= MIN_STATION_SAMPLES
].copy()

print("\n" + "=" * 60)
print("2. STATION-WISE PERFORMANCE")
print("=" * 60)

print(f"Stations with at least {MIN_STATION_SAMPLES} samples: "
      f"{len(eligible)}")

print("\nBest 10 stations (XGBoost MAE):")
print(
    eligible.nsmallest(10, "xgboost_MAE")[
        ["station_id", "samples", "persistence_MAE", "xgboost_MAE"]
    ].round(2).to_string(index=False)
)

print("\nWorst 10 stations (XGBoost MAE):")
print(
    eligible.nlargest(10, "xgboost_MAE")[
        ["station_id", "samples", "persistence_MAE", "xgboost_MAE"]
    ].round(2).to_string(index=False)
)


# --------------------------------------------------
# 3. POLLUTION-CONCENTRATION BANDS
# --------------------------------------------------

# These are descriptive PM2.5 concentration bands,
# not official AQI categories.

df["concentration_band"] = pd.cut(
    df["actual_pm25"],
    bins=[-np.inf, 60, 120, 250, np.inf],
    labels=["Below 60", "60-120", "120-250", "250+"]
)

bands = (
    df.groupby("concentration_band", observed=False)
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

bands.to_csv(OUTPUT / "concentration_band_performance.csv", index=False)

print("\n" + "=" * 60)
print("3. PERFORMANCE BY ACTUAL PM2.5 CONCENTRATION")
print("=" * 60)

print(bands.round(2).to_string(index=False))


# --------------------------------------------------
# 4. EXTREME POLLUTION ANALYSIS
# --------------------------------------------------

extreme = df[df["actual_pm25"] >= 300].copy()

print("\n" + "=" * 60)
print("4. EXTREME POLLUTION (ACTUAL PM2.5 >= 300)")
print("=" * 60)

if len(extreme) > 0:
    extreme_metrics = calculate_metrics(extreme)

    print(f"Extreme observations: {len(extreme):,}")
    print(extreme_metrics.round(3).to_string())

    extreme.to_csv(
        OUTPUT / "extreme_pollution_predictions.csv",
        index=False
    )
else:
    print("No observations with PM2.5 >= 300 found.")


# --------------------------------------------------
# 5. OVERALL COMPARISON
# --------------------------------------------------

overall = calculate_metrics(df)

print("\n" + "=" * 60)
print("5. OVERALL TEST PERFORMANCE")
print("=" * 60)

print(overall.round(3).to_string())

print("\nAnalysis files saved in:", OUTPUT)
print("ERROR ANALYSIS COMPLETED")
