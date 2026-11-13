"""
DIWALI PLUME — OpenAQ air-quality recovery downloader

This version intentionally:
- Reuses the existing station_sensor_metadata.csv (no station/sensor discovery).
- Does NOT download weather (your existing ERA5 file is preserved).
- Uses the correct OpenAQ v3 timestamp path: period.datetimeFrom.utc.
- Writes to NEW *_fixed files, preserving the old checkpoint as a backup.
- Saves each page immediately, so interrupted downloads can be resumed safely.
"""

import os
import time
import json
from pathlib import Path

import pandas as pd
import requests


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]
AIR_DIR = BASE_DIR / "data" / "raw" / "air_quality"
AIR_DIR.mkdir(parents=True, exist_ok=True)

OPENAQ_BASE = "https://api.openaq.org/v3"

START_YEAR = 2022
END_YEAR = 2026
END_DATE = "2026-11-06"

PAGE_LIMIT = 1000
REQUEST_DELAY = 1.1
MAX_RETRIES = 5

API_KEY = os.getenv("OPENAQ_API_KEY")

# Reuse the metadata created during your previous 4-hour run.
SENSOR_METADATA_FILE = AIR_DIR / "station_sensor_metadata.csv"

# NEW files: do not overwrite the old timestamp-broken checkpoint.
CHECKPOINT_FILE = AIR_DIR / "air_quality_checkpoint_fixed.csv"
COMPLETED_FILE = AIR_DIR / "completed_sensors_fixed.json"
FAILED_FILE = AIR_DIR / "failed_sensors_fixed.csv"
FINAL_FILE = AIR_DIR / "delhi_hourly_air_quality.csv"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "DiwaliPlume-Research/1.0"})


# ==========================================================
# REQUEST / CHECKPOINT HELPERS
# ==========================================================

def get_json(url, params):
    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(
                url,
                params=params,
                headers={"X-API-Key": API_KEY},
                timeout=90,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(float(retry_after)) if retry_after else 65
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt == MAX_RETRIES - 1:
                    response.raise_for_status()
                wait = min(30, 5 * (2 ** attempt))
                print(f"OpenAQ server error {response.status_code}; retrying in {wait}s")
                time.sleep(wait)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = min(30, 5 * (2 ** attempt))
            time.sleep(wait)

    raise RuntimeError(f"Request failed: {url}")


def load_completed():
    if not COMPLETED_FILE.exists():
        return set()
    try:
        with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
            return set(str(x) for x in json.load(f))
    except (OSError, json.JSONDecodeError):
        return set()


def save_completed(completed):
    with open(COMPLETED_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(completed), f, indent=2)


def append_page(records):
    if not records:
        return
    pd.DataFrame(records).to_csv(
        CHECKPOINT_FILE,
        mode="a",
        header=not CHECKPOINT_FILE.exists(),
        index=False,
    )


# ==========================================================
# DOWNLOAD ONE SENSOR
# ==========================================================

def download_sensor(sensor):
    sensor_id = str(sensor["sensor_id"])
    total_for_sensor = 0

    for year in range(START_YEAR, END_YEAR + 1):
        date_from = f"{year}-01-01T00:00:00Z"
        date_to = (
            f"{END_DATE}T23:59:59Z"
            if year == END_YEAR
            else f"{year}-12-31T23:59:59Z"
        )

        page = 1
        year_count = 0

        while True:
            payload = get_json(
                f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
                {
                    "datetime_from": date_from,
                    "datetime_to": date_to,
                    "limit": PAGE_LIMIT,
                    "page": page,
                },
            )

            results = payload.get("results", [])
            if not results:
                break

            page_records = []
            for item in results:
                # Correct OpenAQ v3 response structure
                period = item.get("period") or {}
                datetime_from = period.get("datetimeFrom") or {}
                timestamp = datetime_from.get("utc")

                # Do not silently save records without timestamps.
                if not timestamp:
                    continue

                page_records.append({
                    "timestamp_utc": timestamp,
                    "location_id": sensor["location_id"],
                    "station_name": sensor["station_name"],
                    "latitude": sensor.get("latitude"),
                    "longitude": sensor.get("longitude"),
                    "sensor_id": sensor_id,
                    "pollutant": sensor["pollutant"],
                    "value": item.get("value"),
                    "unit": sensor.get("unit"),
                    "has_flags": (item.get("flagInfo") or {}).get("hasFlags"),
                })

            # Save every page immediately. If interrupted, the sensor can
            # be retried; final deduplication removes any repeated rows.
            append_page(page_records)

            count = len(page_records)
            year_count += count
            total_for_sensor += count

            if len(results) < PAGE_LIMIT:
                break

            page += 1
            time.sleep(REQUEST_DELAY)

        print(f"      {year}: {year_count:,} records")

    return total_for_sensor


# ==========================================================
# BUILD FINAL DATASET
# ==========================================================

def build_final_dataset():
    if not CHECKPOINT_FILE.exists():
        print("No timestamped readings were downloaded.")
        return

    print("\nBuilding final dataset...")
    df = pd.read_csv(CHECKPOINT_FILE)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp_utc", "value"])

    df["timestamp_ist"] = df["timestamp_utc"].dt.tz_convert("Asia/Kolkata")

    df = df.drop_duplicates(subset=["sensor_id", "timestamp_utc"])
    df = df.sort_values(["station_name", "pollutant", "timestamp_utc"])

    df.to_csv(FINAL_FILE, index=False)

    print("\nFINAL DATASET")
    print("-------------")
    print("Saved:", FINAL_FILE)
    print("Observations:", f"{len(df):,}")
    print("Stations:", df["location_id"].nunique())
    print("Sensors:", df["sensor_id"].nunique())
    print("Pollutants:")
    print(df["pollutant"].value_counts())
    print("Date range:", df["timestamp_ist"].min(), "to", df["timestamp_ist"].max())


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("DIWALI PLUME — AIR-QUALITY TIMESTAMP RECOVERY")
    print("=" * 60)

    if not API_KEY:
        raise SystemExit("OPENAQ_API_KEY is not set in this terminal.")

    if not SENSOR_METADATA_FILE.exists():
        raise SystemExit(
            f"Sensor metadata not found: {SENSOR_METADATA_FILE}\n"
            "Expected the metadata file from your previous acquisition."
        )

    sensors = pd.read_csv(SENSOR_METADATA_FILE)

    required = {
        "sensor_id", "location_id", "station_name",
        "latitude", "longitude", "pollutant", "unit"
    }
    missing = required - set(sensors.columns)
    if missing:
        raise SystemExit(f"Metadata is missing columns: {sorted(missing)}")

    sensors = sensors.drop_duplicates(subset=["sensor_id"]).reset_index(drop=True)
    completed = load_completed()

    print(f"Existing sensor metadata: {len(sensors)} sensors")
    print(f"Already completed in this recovery run: {len(completed)}")
    print("Weather download: SKIPPED")
    print("Station/sensor discovery: SKIPPED")
    print(f"Recovery checkpoint: {CHECKPOINT_FILE}")

    failures = []

    for index, sensor in sensors.iterrows():
        sensor_id = str(sensor["sensor_id"])

        if sensor_id in completed:
            continue

        print(
            f"\n[{index + 1}/{len(sensors)}] "
            f"{sensor['station_name']} — {sensor['pollutant']} "
            f"(Sensor {sensor_id})"
        )

        try:
            count = download_sensor(sensor)
            completed.add(sensor_id)
            save_completed(completed)
            print(f"    Saved {count:,} timestamped records.")

        except Exception as error:
            print(f"    FAILED: {error}")
            failures.append({
                "sensor_id": sensor_id,
                "station_name": sensor["station_name"],
                "error": str(error),
            })

        time.sleep(REQUEST_DELAY)

    if failures:
        pd.DataFrame(failures).to_csv(FAILED_FILE, index=False)
        print(f"\nFailed sensors recorded in: {FAILED_FILE}")

    print(f"\nCompleted sensors: {len(completed)}/{len(sensors)}")
    build_final_dataset()
    print("\nRecovery run finished.")


if __name__ == "__main__":
    main()
