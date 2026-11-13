"""
end_to_end_replay.py

Manual end-to-end historical replay test.
Uses smoke_test.mp4 at station R K Puram (location_id=17),
timestamp 2025-10-01 06:30 UTC (confirmed good lag availability).

Run from project root:
    venv\Scripts\python.exe end_to_end_replay.py
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.integration.integration_service import IntegrationService

VIDEO = ROOT / "data" / "raw" / "visual" / "input_videos" / "smoke_test.mp4"

# R K Puram, Delhi — location_id=17
LAT = 28.5633
LON = 77.1869
# 2025-10-01 06:30 UTC = 12:00 IST  — confirmed lag availability
DT  = datetime(2025, 10, 1, 6, 30, 0, tzinfo=timezone.utc)

def main():
    if not VIDEO.exists():
        print(f"ERROR: Demo video not found at {VIDEO}")
        sys.exit(1)

    print("=" * 60)
    print("DIWALI PLUME — END-TO-END HISTORICAL REPLAY TEST")
    print("=" * 60)
    print(f"Video:     {VIDEO.name}")
    print(f"Location:  lat={LAT}, lon={LON}")
    print(f"Time (UTC):{DT.isoformat()}")
    print()

    svc = IntegrationService(
        output_dir=ROOT / "data" / "outputs",
        model_path=ROOT / "models" / "vision" / "fire_smoke_yolov8n_dfire_best.pt",
        xgb_path=ROOT / "models" / "xgboost_pm25_6h.pkl",
        features_path=ROOT / "models" / "baseline_features.json",
        data_path=ROOT / "data" / "processed" / "delhi_multimodal_clean.csv",
        max_distance_km=50.0,
        max_reading_age_hours=72.0,
    )

    result = svc.analyze(
        video_path=VIDEO,
        latitude=LAT,
        longitude=LON,
        recording_datetime=DT,
        mode="historical_replay",
    )

    print("\n" + "=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)

    print(f"\nResult ID:  {result['result_id']}")
    print(f"Mode:       {result['mode']}")
    print(f"Warnings:   {len(result['warnings'])}")
    for w in result['warnings']:
        print(f"  ⚠  {w}")

    # Video analysis
    va = result["video_analysis"]
    print(f"\n--- Visual Evidence (NOT PM2.5 measurements) ---")
    print(f"  Frames processed:      {va['total_frames']}")
    print(f"  Frames with smoke:     {va['frames_with_smoke']} ({va['pct_frames_with_smoke']:.1f}%)")
    print(f"  Frames with fire:      {va['frames_with_fire']}  ({va['pct_frames_with_fire']:.1f}%)")
    print(f"  Smoke detections:      {va['smoke_detection_count']}")
    print(f"  Fire detections:       {va['fire_detection_count']}")
    print(f"  Max confidence:        {va['max_confidence']}")
    print(f"  Annotated video:       {va['annotated_video_path']}")

    # Station
    ns = result["nearest_station"]
    print(f"\n--- Nearest Monitoring Station ---")
    print(f"  Found:                 {ns['found']}")
    if ns['found']:
        print(f"  Station:               {ns['station_name']}")
        print(f"  Distance:              {ns['distance_km']:.2f} km")
        print(f"  PM2.5 (measured):      {ns['pm25_ugm3']} µg/m³")
        print(f"  Reading timestamp:     {ns['reading_timestamp']}")
        print(f"  Reading age:           {ns['reading_age_hours']:.1f} h")
        print(f"  Data quality:          {ns['data_quality']}")

    # Forecast
    fc = result["pm25_forecast_6h"]
    print(f"\n--- 6-Hour Forecast (historical-baseline XGBoost) ---")
    print(f"  Available:             {fc.get('available', False)}")
    if fc.get("available"):
        print(f"  Forecast PM2.5:        {fc['forecast_pm25_ugm3']:.1f} µg/m³")
        print(f"  Issue time:            {fc['issue_timestamp']}")
        print(f"  Target time (t+6h):    {fc['target_timestamp']}")
        print(f"  Data quality:          {fc['data_quality']}")
        if fc['missing_features']:
            print(f"  Missing features:      {fc['missing_features']}")

    print(f"\n--- System Note ---")
    print(f"  {result['system_note']}")

    # Save full result
    out = ROOT / "reports" / "integration" / "end_to_end_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"\nFull result saved to: {out}")
    print("\n✓ End-to-end test complete.")

if __name__ == "__main__":
    main()
