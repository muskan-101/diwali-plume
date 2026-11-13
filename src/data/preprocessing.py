
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/air_quality/delhi_hourly_air_quality.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading air-quality data...")
df = pd.read_csv(RAW)

df["timestamp_ist"] = pd.to_datetime(
    df["timestamp_ist"], errors="coerce"
)
df["value"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(subset=["timestamp_ist", "value"])
df = df[df["pollutant"].isin(["pm25", "pm10"])].copy()

# Assign each reading to its IST hour
df["hour"] = df["timestamp_ist"].dt.floor("h")

# Stage 1: aggregate readings within each sensor-hour
sensor_hourly = (
    df.groupby([
        "location_id", "station_name", "latitude", "longitude",
        "sensor_id", "pollutant", "hour"
    ])
    .agg(
        value=("value", "mean"),
        reading_count=("value", "size"),
        flagged=("has_flags", "max")
    )
    .reset_index()
)

# Stage 2: combine sensors at each station-hour
station_hourly = (
    sensor_hourly.groupby([
        "location_id", "station_name", "latitude",
        "longitude", "pollutant", "hour"
    ])
    .agg(
        value=("value", "mean"),
        sensor_count=("sensor_id", "nunique"),
        reading_count=("reading_count", "sum"),
        flagged_sensor_count=("flagged", "sum")
    )
    .reset_index()
)

# Convert pollutant values into separate columns
wide = station_hourly.pivot(
    index=["location_id", "station_name", "latitude",
           "longitude", "hour"],
    columns="pollutant",
    values="value"
).reset_index()

wide.columns.name = None

# Add sensor coverage counts
counts = station_hourly.pivot(
    index=["location_id", "station_name", "latitude",
           "longitude", "hour"],
    columns="pollutant",
    values="sensor_count"
).reset_index()

counts.columns.name = None
counts = counts.rename(columns={
    "pm25": "pm25_sensor_count",
    "pm10": "pm10_sensor_count"
})

result = wide.merge(
    counts,
    on=["location_id", "station_name", "latitude",
        "longitude", "hour"],
    how="left"
)

result = result.sort_values(["location_id", "hour"])

output = OUT / "delhi_station_hourly.csv"
result.to_csv(output, index=False)

print("\nPREPROCESSING COMPLETE")
print("Rows:", len(result))
print("Stations:", result["location_id"].nunique())
print("Date range:", result["hour"].min(), "to", result["hour"].max())
print("\nMissing pollutant values:")
print(result[["pm25", "pm10"]].isna().sum())
print("\nSaved:", output)
