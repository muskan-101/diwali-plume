# Diwali Plume — Stage A: Integration Prototype

## Status: Implemented and Tested

---

## 1. Architecture Overview

```
User Upload
 (video + lat/lon + datetime)
         │
         ▼
 ┌──────────────────────┐
 │   Flask API (app.py) │
 │   POST /analyze      │
 └──────────┬───────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │       IntegrationService                │
 │  (src/integration/integration_service)  │
 └────┬──────────────┬──────────────┬──────┘
      │              │              │
      ▼              ▼              ▼
 VideoAnalyzer  StationMatcher  ForecastEngine
 (YOLO vision)  (Haversine +   (XGBoost 6-h
                 PM2.5 lookup)  historical)
      │              │              │
      ▼              ▼              ▼
 Annotated MP4   Station ID     Forecast PM2.5
 Detection CSV   PM2.5 reading  t + 6 h
 Statistics      Distance/age   Feature vector
```

Three distinct evidence types are always kept separate in the response:
- Visual evidence (video detector output)
- Measured station reading (historical ground truth)
- Model forecast (6-hour-ahead XGBoost prediction)

---

## 2. Files Created

| File | Description |
|------|-------------|
| src/integration/__init__.py | Package init |
| src/integration/station_matcher.py | Haversine station finder + PM2.5 retrieval |
| src/integration/forecast_engine.py | XGBoost feature construction + prediction |
| src/integration/video_analyzer.py | YOLO inference wrapper |
| src/integration/integration_service.py | Pipeline orchestrator |
| src/api/__init__.py | Package init |
| src/api/app.py | Flask REST API |
| tests/__init__.py | Test package init |
| tests/test_station_matcher.py | 21 station-matcher tests |
| tests/test_forecast_engine.py | 17 forecast-engine tests |
| tests/test_video_analyzer.py | 15 video-analyzer tests |
| tests/test_api.py | 21 API tests (14 unit + 7 integration) |
| end_to_end_replay.py | Manual end-to-end replay script |
| reports/integration/STAGE_A_IMPLEMENTATION.md | This document |

---

## 3. Existing Files Reused (Not Modified)

| File | How Used |
|------|----------|
| src/vision/smoke_fire_detector.py | Inference logic studied; VideoAnalyzer reimplements the same loop in an importable class. The original uses argparse and hardcoded paths — not suitable for API use. The underlying YOLO model and confidence threshold are reused identically. |
| src/forecasting/baseline.py | Feature construction logic, lag formula, cyclical encoding, and column alias resolution are reproduced exactly in forecast_engine.py. |
| models/vision/fire_smoke_yolov8n_dfire_best.pt | Loaded by VideoAnalyzer via ultralytics.YOLO. Not retrained. |
| models/xgboost_pm25_6h.pkl | Loaded by ForecastEngine via joblib. Not retrained. |
| models/baseline_features.json | 22-feature ordered list. Used verbatim. |
| data/processed/delhi_multimodal_clean.csv | Station metadata, PM2.5 readings, weather features. Read-only. |

---

## 4. Model Loading Details

### Vision Model (YOLOv8n)
- Path: models/vision/fire_smoke_yolov8n_dfire_best.pt
- Loaded via ultralytics.YOLO(path) on first inference call (lazy load).
- Device: cpu (explicit).
- Classes: 0 = smoke, 1 = fire.
- Confidence threshold: 0.25 (configurable).

### XGBoost Forecast Model
- Path: models/xgboost_pm25_6h.pkl
- Loaded via joblib.load(path).
- Best iteration: 350 (n_estimators=1000, early stopped).
- Feature schema: models/baseline_features.json (22 features, fixed order).
- Loaded once per server process and shared across requests.

---

## 5. Feature Construction Logic

Exactly reproduces src/forecasting/baseline.py:

### PM2.5 Lags (6 features)
Lags [0, 1, 3, 6, 12, 24] hours before prediction time t.

For lag k:
  target_ts = issue_time - timedelta(hours=k)
  Exact timestamp lookup in station history (+/- 30 min tolerance)
  If not found or NaN: feature marked missing, forecast REJECTED

Critical guarantee: Only data at or before t is used.
Data from t+1 through t+6 is never accessed during feature construction.

### Station Coordinates (2 features)
Static latitude and longitude from the matched station.

### Weather Features (10 features)
ERA5 reanalysis weather from the matching historical row at time t (+/- 1 h tolerance).
Source: delhi_multimodal_clean.csv — same central Delhi ERA5 coordinate for all stations.
If any weather feature is NaN, it is flagged but the forecast is NOT rejected
(weather features are degraded signal; PM2.5 lags are mandatory).

### Cyclical Time Encoding (4 features)
  hour_sin = sin(2pi * hour / 24)
  hour_cos = cos(2pi * hour / 24)
  day_sin  = sin(2pi * day_of_year / 365.25)
  day_cos  = cos(2pi * day_of_year / 365.25)
Identical to baseline.py lines 153-156.

### Feature Order (22 total)
  pm25_lag_0, pm25_lag_1, pm25_lag_3, pm25_lag_6, pm25_lag_12, pm25_lag_24,
  latitude, longitude,
  temperature_2m, relative_humidity_2m, dew_point_2m, precipitation,
  pressure_msl, wind_speed_10m, wind_direction_10m, wind_gusts_10m,
  cloud_cover, boundary_layer_height,
  hour_sin, hour_cos, day_sin, day_cos

---

## 6. Station Matching Methodology

1. Load delhi_multimodal_clean.csv once; build station metadata table.
2. Compute Haversine distance from user coordinates to every station.
3. Filter to stations within max_distance_km (default 50 km).
4. Iterate candidates in ascending distance order.
5. For each candidate, retrieve the latest PM2.5 reading at or before the requested time.
6. First candidate with data_quality = "good" is returned.
7. If no candidate has a good reading, the nearest is returned with its actual quality status.
8. Reading staleness threshold: max_reading_age_hours (default 3 h).

Data quality labels:
- good: non-null reading within staleness threshold
- stale: reading exists but older than threshold
- missing: no PM2.5 reading found at or before requested time
- no_station: no station within configured radius, or invalid coordinates

---

## 7. API Endpoints

### GET /health

    curl http://localhost:5000/health

Response:
    {
      "status": "ok",
      "api": "Diwali Plume Stage A",
      "mode": "historical_replay",
      "vision_model_available": true,
      "forecast_model_available": true,
      "feature_schema_available": true,
      "dataset_available": true
    }

### POST /analyze

    curl -X POST http://localhost:5000/analyze \
      -F "video=@data/raw/visual/input_videos/smoke_test.mp4" \
      -F "latitude=28.5633" \
      -F "longitude=77.1869" \
      -F "recording_datetime=2025-10-01T12:00:00+05:30" \
      -F "mode=historical_replay"

Response includes: result_id, video_analysis, nearest_station, pm25_forecast_6h,
warnings, system_note.

### GET /results/<result_id>

    curl http://localhost:5000/results/<uuid>

### GET /outputs/<result_id>/<filename>

    curl http://localhost:5000/outputs/<uuid>/annotated.mp4 --output annotated.mp4

---

## 8. How to Launch the Backend

    cd "C:\Users\Muskan Thakur\OneDrive\Desktop\diwali-plume"
    venv\Scripts\activate
    python src\api\app.py --host 0.0.0.0 --port 5000 --debug

---

## 9. How to Run Tests

    # Fast unit tests only
    venv\Scripts\python.exe -m pytest tests/test_station_matcher.py tests/test_forecast_engine.py tests/test_api.py -v --tb=short -k "not Integration"

    # Video analyzer unit tests (requires YOLO)
    venv\Scripts\python.exe -m pytest tests/test_video_analyzer.py -v --tb=short -k "not DemoVideos"

    # Integration smoke tests (slower, runs YOLO on demo videos)
    venv\Scripts\python.exe -m pytest tests/test_video_analyzer.py::TestDemoVideos tests/test_api.py::TestAnalyzeIntegration -v --tb=short

    # Full suite
    venv\Scripts\python.exe -m pytest tests/ -v --tb=short

    # Manual end-to-end replay
    venv\Scripts\python.exe end_to_end_replay.py

---

## 10. Test Results

| Test suite | Tests | Passed |
|------------|-------|--------|
| test_station_matcher.py | 21 | 21 |
| test_forecast_engine.py | 17 | 17 |
| test_video_analyzer.py (unit) | 10 | 10 |
| test_video_analyzer.py (integration) | 5 | 5 |
| test_api.py (unit) | 14 | 14 |
| test_api.py (integration) | 7 | 7 |
| Total | 74 | 74 |

---

## 11. End-to-End Test Result

Video: smoke_test.mp4
Location: R K Puram, Delhi (lat=28.5633, lon=77.1869)
Time: 2025-10-01 06:30 UTC (12:00 IST)

| Component | Result |
|-----------|--------|
| Station matched | R K Puram, Delhi - DPCC (0.08 km) |
| PM2.5 measured | 74.0 ug/m3 (data_quality: good) |
| All 6 PM2.5 lags | Available |
| Weather features | Available (ERA5) |
| Forecast PM2.5 (t+6h) | ~68 ug/m3 |
| Smoke detections | Present in video |
| Fire detections | None |

---

## 12. Current Limitations

1. Historical replay only. ERA5 reanalysis weather = hindcast, not operational forecast.
2. 6-hour horizon only. No 12h, 24h, or 48h predictions claimed or supported.
3. Sparse coverage. Stations with few historical rows may have incomplete lag availability.
4. ERA5 single-point weather. All 82 stations share one central Delhi ERA5 coordinate.
5. Nearest != local. The nearest station is not necessarily the user's local air quality.
6. No silent PM2.5 substitution. Stale readings are flagged, never silently replaced.
7. Visual evidence only. Smoke/fire confidence != PM2.5 concentration.
8. No AQI output. PM2.5 in ug/m3 only.
9. Not validated for real-time Diwali forecasting.
10. No API authentication. Add auth before any public exposure.
11. Single-process Flask. Use Waitress or Gunicorn for production.

---

## 13. Issues Requiring Your Decision

| Issue | Description |
|-------|-------------|
| requirements.txt | flask==3.1.3, werkzeug==3.1.9, pytest==9.2.0 were installed but are not in requirements.txt. Should they be added? |
| Live mode | /analyze rejects mode != "historical_replay" with a warning. Enabling live mode requires operational weather forecast integration. Out of scope for Stage A. |
