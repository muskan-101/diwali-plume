
from pathlib import Path
import pandas as pd

INPUT = Path("data/processed/delhi_multimodal_hourly.csv")
OUTPUT = Path("data/processed/delhi_multimodal_clean.csv")
REPORT = Path("reports/error_analysis")

REPORT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

df["pm25_original"] = df["pm25"]

# Quality flags
df["pm25_negative_flag"] = df["pm25"] < 0
df["pm25_10000_flag"] = df["pm25"] == 10000
df["pm25_high_flag"] = df["pm25"] > 500

# Invalidate only clearly impossible/suspicious readings.
invalid = df["pm25_negative_flag"] | df["pm25_10000_flag"]
df.loc[invalid, "pm25"] = float("nan")

# Save the cleaned dataset; preserve all other columns.
df.to_csv(OUTPUT, index=False)

summary = {
    "total_rows": len(df),
    "negative_readings": int(df["pm25_negative_flag"].sum()),
    "exact_10000_readings": int(df["pm25_10000_flag"].sum()),
    "readings_above_500": int(df["pm25_high_flag"].sum()),
    "invalidated_readings": int(invalid.sum()),
    "remaining_pm25_values": int(df["pm25"].notna().sum()),
}

print("=" * 55)
print("DIWALI PLUME - QUALITY CONTROL")
print("=" * 55)

for key, value in summary.items():
    print(f"{key}: {value}")

pd.DataFrame([summary]).to_csv(
    REPORT / "quality_control_summary.csv",
    index=False
)

print("\nSaved:", OUTPUT)
