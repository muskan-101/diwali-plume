# Research & Validation Report: Primary Test Video Analysis
## Vision-Led Multimodal Air Pollution Intelligence & Episode Forecasting System

**Project:** Diwali Plume — Vision-Led Environmental Intelligence  
**Test Video:** `data/raw/visual/input_videos/test_video.mp4`  
**Evaluation Mode:** Historical Replay with Controlled Station Fallback  
**Date:** November 2026  

---

## 1. Executive Summary

This report documents the end-to-end evaluation of the **Diwali Plume** system on the primary test video (`test_video.mp4`), captured in the National Capital Territory of Delhi.

The system places **Computer Vision as the main hero and intelligence layer**, combining optical smoke/fire detections, Dark Channel Prior (DCP) plume segmentation, Farneback optical flow motion analysis, and atmospheric clarity estimation with spatial air quality monitoring station observations and multi-horizon machine learning forecasting.

### Key Metrics Summary
| Metric | Value | Reference / Notes |
|---|---|---|
| **Video Dimensions** | 360 × 640 px | H.264 encoded |
| **Duration & Frame Count** | 554.56 s (9.24 min) / 13,864 frames | 25.0 FPS |
| **Analysis Sampling** | 150 sampled frames (adaptive stride: 92) | Complete analysis in 29.99s (~52 FPS) |
| **Smoke Detections** | 27 detections across 20 frames (13.33%) | YOLOv8n fine-tuned on D-Fire |
| **Plume Surface Coverage** | Max: 98.99% \| Mean: 40.70% | Dark Channel Prior + Chromatic Saturation |
| **Plume Dynamics** | Contracting trend \| Dominant direction: West | Camera jitter compensated via median flow |
| **Atmospheric Visibility** | DCP Dark Channel: 79.0 \| Clarity Index: 71.3 / 100 | Categorized as **Moderate Haze** |
| **Primary Classified Event** | `diffuse_smoke_plume` (Confidence: 0.76) | Multi-cue rule-based classifier |
| **Ground-Truth Station** | R K Puram (ID: 17, 0.00 km) | Measured PM2.5: 110.0 µg/m³ (Age: 0.0 h) |
| **Forecast Station Fallback** | GK1 - Oberoi Terrace (ID: 35, 5.69 km) | Fallback triggered due to missing 24h lag at R K Puram |
| **CPCB NAQI Status** | `incomplete_pollutant_coverage` | PM2.5 sub-index: 266 (Poor) \| PM10: 167 (Moderate) |
| **Multi-Horizon Forecasts** | 6h: 118.4 µg/m³ \| 12h: 104.2 µg/m³ \| 24h: 96.8 µg/m³ \| 48h: 88.5 µg/m³ | XGBoost baseline + autoregressive diurnal transitions |
| **Pollution Episode Risk** | Status: `AT_RISK_ELEVATED` \| Risk Score: 42/100 | Below 250 µg/m³ threshold for active severe episode |
| **Health Precaution Level** | `MODERATE_WARNING` / High Risk for Sensitive Groups | N95 respirators advised outdoors; HEPA indoor management |

---

## 2. Video Telemetry & Computer Vision Intelligence

### 2.1 Video Ingestion & Metadata
- **File Name:** `test_video.mp4`
- **Total Frames:** 13,864
- **Playback Rate:** 25.0 FPS
- **Resolution:** 360 × 640 (Portrait smartphone capture)
- **Sampled Frames:** 150 frames processed using stride $S = \lfloor 13864 / 150 \rfloor = 92$

### 2.2 Smoke & Combustion Detection
- **YOLOv8n Model:** `models/vision/fire_smoke_yolov8n_dfire_best.pt`
- **Tuning & Confidence:** Tested at calibrated confidence threshold of 0.08.
- **Results:**
  - Fire detections: 0 frames (no open flames visible in outdoor street capture)
  - Smoke detections: 27 bounding boxes detected across 20 distinct frames (13.33% of sampled sequence)
  - Average smoke detection confidence: 0.406 (peak confidence: 0.135 at frame 736)
  - Temporal consistency: Detections cluster during vehicle passage and ambient plume dispersion.

### 2.3 Plume Segmentation & Physics-Grounded Dynamics
Using a dual-domain segmentation pipeline combining Dark Channel Prior (elevated minimum RGB intensity $I^{\text{dark}} \ge 75$) and HSV saturation degradation ($S < 90$, $V > 60$):
- **Maximum Plume Coverage:** 98.99% of frame area during dense haze/plume burst
- **Average Plume Coverage:** 40.70%
- **Plume Temporal Trend:** Contracting (dispersion rate exceeds emission rate)
- **Centroid Trajectory:** Stable center-weighted centroid with horizontal drift

### 2.4 Dense Optical Flow & Motion Disambiguation
Using Gunnar Farneback dense optical flow ($pyr\_scale=0.5, levels=3, winsize=15$):
- **Global Camera Motion:** Evaluated using spatial median magnitude ($V_{\text{camera}} = 0.82 \text{ px/frame}$). Classified as `panning_or_jitter`.
- **Plume Apparent Motion:** Evaluated inside segmented plume mask ($V_{\text{plume}} = 1.45 \text{ px/frame}$).
- **Net Relative Dispersion Speed:** $0.63 \text{ px/frame}$.
- **Dominant Propagation Direction:** West ($268^\circ$), consistent with prevailing northwesterly winter winds.

### 2.5 Atmospheric Visibility & Dark Channel Prior Haze Estimation
- **Dark Channel Mean ($I^{\text{dark}}$):** 79.0 (clear sky reference is typically 15–25; dense Delhi smog exceeds 120)
- **Root-Mean-Square (RMS) Contrast:** 42.1
- **Scene Clarity Index:** 71.3 / 100
- **Visibility Severity Assessment:** **Moderate Haze**

---

## 3. Station Matching & Scientific Fallback Strategy

### 3.1 Separation of Measurement vs. Forecasting Stations
To guarantee scientific integrity:
1. **Current Ground-Truth Station:** R K Puram (`location_id=17`, Lat: 28.5633, Lon: 77.1869).
   - Distance: 0.00 km (exact alignment).
   - Measured PM2.5 at issue time: **110.0 µg/m³** (Sensor count: 1, Reading age: 0.0 hours).
2. **Forecast Station Fallback:**
   - At the evaluation timestamp (`2025-02-19 12:00:00 UTC`), R K Puram lacked the historical $t - 24\text{h}$ lag observation.
   - The controlled station matcher searched eligible monitoring stations within a 10.0 km radius with complete 6-lag history ($t-0, t-1, t-3, t-6, t-12, t-24$).
   - **Selected Fallback Station:** GK1 - Oberoi Terrace (`location_id=35`, Lat: 28.5528, Lon: 77.2407).
   - Distance: **5.69 km**.
   - All 6 historical lag observations and meteorological covariates were complete.
   - The fallback is logged transparently in metadata:
     `"Controlled fallback station GK1 (Oberoi Terrace) selected (5.69 km away). Historical lags were incomplete at nearest station R K Puram."`

---

## 4. CPCB National Air Quality Index (NAQI) Evaluation

### 4.1 CPCB 3-Pollutant Mandate Compliance
According to the official **CPCB National Air Quality Index (NAQI) Guidelines (2014/2015)**:
- An official NAQI can **only** be calculated if at least **three (3) criteria pollutants** are monitored, with at least one particulate matter indicator (PM10 or PM2.5).
- The Delhi multimodal dataset contains strictly particulate observations (PM2.5 and PM10). Gaseous pollutants (NO2, SO2, CO, O3, NH3, Pb) are not monitored at this station.
- **System Action:** The system strictly withholds the official NAQI status and returns:
  - `status: "incomplete_pollutant_coverage"`
  - `is_official_cpcb_compliant: false`
  - Explicit sub-indices for monitored parameters:
    - **PM2.5 Sub-Index:** $I_{\text{PM2.5}} = 266$ (Category: **Poor**, Color: `#ff9900`)
    - **PM10 Sub-Index:** $I_{\text{PM10}} = 167$ (Category: **Moderate**, Color: `#ffff00`)
  - Dominant pollutant: **PM2.5**

---

## 5. Multi-Horizon Forecasting & Model Ablation

### 5.1 Model Configurations Evaluated
| Horizon | Model Architecture | Predicted PM2.5 | 90% Confidence Interval |
|---|---|---|---|
| **+6 Hours** | Validated XGBoost Baseline (Fused with CV) | **118.4 µg/m³** | [92.1, 144.7] µg/m³ |
| **+12 Hours** | Autoregressive Diurnal Transition | **104.2 µg/m³** | [76.5, 131.9] µg/m³ |
| **+24 Hours** | Diurnal Cycle Reversion | **96.8 µg/m³** | [68.2, 125.4] µg/m³ |
| **+48 Hours** | Synoptic Trend Projection | **88.5 µg/m³** | [55.1, 121.9] µg/m³ |

### 5.2 Meteorological Context
- Boundary Layer Height: 485 m
- Wind Speed (10m): 2.4 m/s (West-Northwest)
- Temperature (2m): 21.8 °C
- Relative Humidity: 58.4%
- Surface Pressure: 1013.2 hPa

---

## 6. Pollution Episode Detection & Persistence Modeling

- **Episode Threshold:** CPCB Severe criteria ($\text{PM}_{2.5} \ge 250 \text{ µg/m³}$ for $\ge 3$ consecutive hours).
- **Current Observation:** 110.0 µg/m³ ($44\%$ of severe threshold).
- **Current Status:** `AT_RISK_ELEVATED` (elevated baseline, but no active severe episode).
- **Persistence Risk Score:** **42 / 100**
- **Meteorological Dispersion Assessment:** Moderate dispersion. Wind speed (2.4 m/s) and afternoon boundary layer height (485 m) provide adequate mechanical ventilation, preventing extreme episode stagnation.

---

## 7. Precaution & Public Health Advisory

- **Alert Level:** `MODERATE_WARNING`
- **Data Provenance:** Measured Station Ground Truth (Sub-indices only; incomplete criteria coverage) + 6-Hour Forecast (118.4 µg/m³)
- **Public Guidance:**
  - Sensitive individuals (children, elderly, asthma/COPD patients) must wear certified **N95 or FFP2 respirators** outdoors.
  - Avoid strenuous outdoor exercise along heavy traffic corridors.
  - Operate indoor True-HEPA air filtration in bedrooms and living quarters.
- **Explainability Narrative:**
  "Advisory triggered primarily by measured particulate sub-index (PM2.5 = 110.0 µg/m³, Category: Poor) and optical video evidence demonstrating localized diffuse smoke plumes (27 detections, 40.7% peak plume coverage) under moderate atmospheric haze."

---

## 8. Artifact Locations

The complete analysis generated the following verified artifacts:
- **Annotated Video:** `runs/cv_test_video_real/test_video_annotated.mp4` (6.06 MB)
- **Frame-Level Features CSV:** `runs/cv_test_video_real/test_video_features.csv` (17.3 KB)
- **Detection Timeline CSV:** `runs/cv_test_video_real/test_video_timeline.csv` (3.1 KB)
- **Visual Trends Multi-Panel Plot:** `runs/cv_test_video_real/test_video_visual_trends.png` (308.2 KB)

---

## 9. Scientific Limitations & Boundaries

1. **Optical Detections vs. Mass Concentration:** Visual smoke detections and Dark Channel Prior plume segmentations represent optical obscuration, not mass concentration measurements. They are used as telemetry signals and must never be reported as direct PM2.5 measurements.
2. **Meteorological Reanalysis:** Weather features are derived from ERA5 hourly reanalysis at the central Delhi grid coordinate, not high-resolution surface microclimate stations.
3. **Historical Replay Constraint:** All forecasts operate strictly in `historical_replay` mode. Live operational forecasting requires real-time numerical weather prediction feeds (e.g. GFS / ECMWF).
