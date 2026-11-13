
import pandas as pd
from pathlib import Path

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

air = pd.read_csv(OUT / "delhi_station_hourly.csv")
weather = pd.read_csv("data/raw/weather/delhi_era5_hourly.csv")

# Both datasets are aligned internally using UTC
air["hour"] = pd.to_datetime(air["hour"], utc=True)
weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True)

# Prepare weather time series
weather = weather.set_index("timestamp").sort_index()

# Interpolate weather at the exact pollution timestamps
target_times = pd.DatetimeIndex(air["hour"].drop_duplicates().sort_values())

combined_index = weather.index.union(target_times).sort_values()

weather_aligned = (
    weather.reindex(combined_index)
    .interpolate(method="time", limit_area="inside")
    .reindex(target_times)
    .reset_index()
)

weather_aligned = weather_aligned.rename(columns={"index": "hour"})

# Merge
merged = air.merge(
    weather_aligned,
    on="hour",
    how="left",
    validate="many_to_one"
)

print("\nMERGE RESULTS")
print("Rows:", len(merged))
print("Stations:", merged["location_id"].nunique())
print(
    "Weather match rate:",
    round(merged["temperature_2m"].notna().mean() * 100, 2),
    "%"
)

print("\nMissing weather values:")
print(merged[weather.columns].isna().sum())

# Restore IST for readable timestamps
merged["hour"] = merged["hour"].dt.tz_convert("Asia/Kolkata")

merged.to_csv(OUT / "delhi_multimodal_hourly.csv", index=False)

print("\nSaved:", OUT / "delhi_multimodal_hourly.csv")
