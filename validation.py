from pathlib import Path
Path("data/processed").mkdir(parents=True, exist_ok=True)
import pandas as pd

path = "data/raw/air_quality/delhi_hourly_air_quality.csv"
df = pd.read_csv(path)

df["timestamp_ist"] = pd.to_datetime(df["timestamp_ist"])
df = df.sort_values(["sensor_id", "timestamp_ist"])

# Check duplicate sensor-time observations
duplicates = df.duplicated(["sensor_id", "timestamp_ist"]).sum()

# Check time gaps for each sensor
results = []

for sensor, group in df.groupby("sensor_id"):
    times = group["timestamp_ist"].drop_duplicates().sort_values()
    gaps = times.diff().dropna()

    results.append({
        "sensor_id": sensor,
        "station": group["station_name"].iloc[0],
        "records": len(times),
        "median_interval_hours": gaps.median().total_seconds() / 3600
            if len(gaps) else None,
        "max_gap_hours": gaps.max().total_seconds() / 3600
            if len(gaps) else None
    })

report = pd.DataFrame(results)

print("Duplicate sensor-time records:", duplicates)
print("\nMedian interval distribution:")
print(report["median_interval_hours"].value_counts().head(10))

print("\nLargest gaps:")
print(report.nlargest(10, "max_gap_hours").to_string(index=False))

print("\nSensors with median interval other than 1 hour:")
print(report[report["median_interval_hours"] != 1].shape[0])

report.to_csv("data/processed/sensor_temporal_validation.csv", index=False)
print("\nSaved sensor_temporal_validation.csv")