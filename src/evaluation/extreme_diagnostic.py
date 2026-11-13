
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/test_predictions_6h.csv")
OUTPUT = Path("reports/error_analysis")

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["timestamp"] = pd.to_datetime(df["timestamp"])

actual = df["actual_pm25"]
pred = df["xgboost_prediction"]

print("=" * 60)
print("EXTREME POLLUTION DIAGNOSTICS")
print("=" * 60)

# 1. Prediction distribution
print("\n1. ACTUAL VS PREDICTED PM2.5 DISTRIBUTION")

distribution = pd.DataFrame({
    "Actual": actual.describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    ),
    "XGBoost": pred.describe(
        percentiles=[0.50, 0.75, 0.90, 0.95, 0.99]
    )
})

print(distribution.round(2))
distribution.to_csv(OUTPUT / "prediction_distribution.csv")

# 2. Event detection at different thresholds
print("\n2. HIGH-POLLUTION EVENT DETECTION")

rows = []

for threshold in [60, 120, 250, 300]:
    actual_event = actual >= threshold
    predicted_event = pred >= threshold

    tp = int((actual_event & predicted_event).sum())
    fp = int((~actual_event & predicted_event).sum())
    fn = int((actual_event & ~predicted_event).sum())
    tn = int((~actual_event & ~predicted_event).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    rows.append({
        "threshold": threshold,
        "actual_events": int(actual_event.sum()),
        "detected_events": tp,
        "missed_events": fn,
        "false_alarms": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

events = pd.DataFrame(rows)
print(events.round(3).to_string(index=False))
events.to_csv(OUTPUT / "event_detection_metrics.csv", index=False)

# 3. Peak underprediction
print("\n3. EXTREME PEAK UNDERPREDICTION")

df["error"] = pred - actual
df["absolute_error"] = df["error"].abs()

extreme = df[actual >= 300].copy()

if len(extreme):
    print(f"Extreme samples: {len(extreme):,}")
    print(f"Mean actual PM2.5: {extreme.actual_pm25.mean():.2f}")
    print(f"Mean predicted PM2.5: {extreme.xgboost_prediction.mean():.2f}")
    print(f"Mean underprediction: {-extreme.error.mean():.2f}")
    print(f"Median underprediction: {-extreme.error.median():.2f}")
    print(f"Maximum actual PM2.5: {extreme.actual_pm25.max():.2f}")
    print(f"Maximum predicted PM2.5: {extreme.xgboost_prediction.max():.2f}")

# 4. Monthly extreme event detection
print("\n4. MONTHLY EXTREME EVENT PERFORMANCE")

df["month"] = df["timestamp"].dt.strftime("%Y-%m")

monthly_rows = []

for month, group in df.groupby("month"):
    actual_events = group["actual_pm25"] >= 300
    predicted_events = group["xgboost_prediction"] >= 300

    count = int(actual_events.sum())
    detected = int((actual_events & predicted_events).sum())

    monthly_rows.append({
        "month": month,
        "actual_extreme_events": count,
        "detected": detected,
        "missed": count - detected,
        "recall": detected / count if count else np.nan,
        "mean_actual": group["actual_pm25"].mean(),
        "mean_prediction": group["xgboost_prediction"].mean(),
    })

monthly = pd.DataFrame(monthly_rows)
print(monthly.round(2).to_string(index=False))

monthly.to_csv(
    OUTPUT / "monthly_extreme_detection.csv",
    index=False
)

# 5. Save the largest missed peaks
missed = df[
    (df["actual_pm25"] >= 300)
    & (df["xgboost_prediction"] < 300)
].copy()

missed = missed.sort_values("actual_pm25", ascending=False)

missed.to_csv(
    OUTPUT / "missed_extreme_events.csv",
    index=False
)

print(f"\nMissed extreme events saved: {len(missed):,}")
print("\nDiagnostics saved in:", OUTPUT)
print("COMPLETED")
