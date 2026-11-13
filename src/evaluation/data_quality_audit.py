
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/delhi_multimodal_hourly.csv")
OUT = Path("reports/error_analysis")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["hour"] = pd.to_datetime(df["hour"], errors="coerce")

df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
d = df.dropna(subset=["pm25"]).copy()

print("=" * 60)
print("PM2.5 DATA QUALITY AUDIT")
print("=" * 60)

print(f"\nTotal PM2.5 records: {len(d):,}")
print(f"Negative values: {(d.pm25 < 0).sum():,}")
print(f"Above 500: {(d.pm25 > 500).sum():,}")
print(f"Above 1000: {(d.pm25 > 1000).sum():,}")
print(f"Exactly -10000: {(d.pm25 == -10000).sum():,}")

print("\nDISTRIBUTION")
print(d.pm25.describe(
    percentiles=[.01, .05, .50, .90, .95, .99, .999]
))

# Inspect suspicious observations.
suspicious = d[
    (d.pm25 < 0) | (d.pm25 > 500)
].copy()

columns = [
    c for c in ["location_id", "hour", "pm25", "latitude", "longitude"]
    if c in suspicious.columns
]

print("\nSUSPICIOUS RECORDS:", len(suspicious))
print(suspicious[columns].sort_values("pm25").to_string(index=False))

suspicious.to_csv(OUT / "suspicious_pm25_records.csv", index=False)

# Station-level summary.
station_summary = (
    d.groupby("location_id")
    .agg(
        records=("pm25", "size"),
        minimum=("pm25", "min"),
        maximum=("pm25", "max"),
        negative_count=("pm25", lambda x: (x < 0).sum()),
        above_500=("pm25", lambda x: (x > 500).sum()),
        above_1000=("pm25", lambda x: (x > 1000).sum()),
    )
    .reset_index()
    .sort_values("maximum", ascending=False)
)

station_summary.to_csv(OUT / "station_data_quality.csv", index=False)

print("\nTOP 15 STATIONS WITH HIGHEST MAXIMUM VALUES")
print(station_summary.head(15).to_string(index=False))

print("\nSaved audit files in:", OUT)
