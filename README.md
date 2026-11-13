# Diwali Plume — Vision-Led Environmental Intelligence System
> **Multimodal Air Pollution Intelligence, Episode Persistence & Multi-Horizon Forecasting for Delhi-NCR**

[![Python 3.15](https://img.shields.io/badge/Python-3.15-blue.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-orange.svg)](https://ultralytics.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-PM2.5-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

**Diwali Plume** is a research-oriented, vision-led environmental intelligence system designed to capture the reality of hyper-local air pollution in Delhi-NCR.

Rather than relying purely on sparse monitoring station averages kilometers away, **Diwali Plume puts Computer Vision as the main hero**:
1. **Visual Combustion & Plume Detection:** Fine-tuned YOLOv8n smoke & fire detection with temporal tracking and IoU deduplication.
2. **Physics-Grounded Plume Segmentation:** Dark Channel Prior (DCP) combined with chromatic saturation degradation to compute localized plume surface area coverage, expansion rates, and spatial drift.
3. **Dense Optical Flow (Farneback):** Calculates image-plane apparent velocity, distinguishing localized plume dispersion from handheld camera motion jitter.
4. **Atmospheric Visibility Estimation:** Scene Clarity Index ($0$ to $100$) and Dark Channel Prior transmission degradation.
5. **Spatial Station Matching with Controlled Fallback:** Matches nearest physical monitoring station for ground truth ($\text{PM}_{2.5}$ & $\text{PM}_{10}$), and identifies candidate stations within $10\text{ km}$ for complete historical time-series lags.
6. **CPCB NAQI 3-Pollutant Mandate:** Strictly adheres to Central Pollution Control Board (CPCB) NAQI guidelines, withholding official AQI when fewer than 3 criteria pollutants are available and reporting explicit sub-indices.
7. **Multi-Horizon Multimodal Forecasting:** Validated 6-hour XGBoost baseline extended to 12h, 24h, and 48h autoregressive diurnal projections.
8. **Pollution Episode Persistence Modeling:** Survival decay analysis predicting remaining duration and time-to-improvement for severe smog episodes.
9. **Actionable Precaution Advisory:** Evidence-based health guidance (N95 respirators, indoor HEPA management, vulnerable population protection).

---

## 🏗️ System Architecture

```
[User-Uploaded Video / CCTV]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│      Computer Vision Intelligence Pipeline             │
│  ├─ YOLOv8n Smoke & Fire Detector                      │
│  ├─ Dark Channel Prior (DCP) Plume Segmenter           │
│  ├─ Farneback Dense Optical Flow Motion Disambiguator  │
│  └─ Scene Clarity & Haze Severity Estimator            │
└───────────────────────┬────────────────────────────────┘
                        │ Visual Telemetry
                        ▼
┌────────────────────────────────────────────────────────┐
│     Multimodal Fusion & Environmental Alignment        │
│  ├─ Spatial Station Matcher (Haversine Distance)       │
│  ├─ Controlled Forecast Station Fallback (<= 10 km)    │
│  ├─ CPCB NAQI Engine (8-Criteria Breakpoints)          │
│  └─ ERA5 Meteorological Reanalysis Integration         │
└───────────────────────┬────────────────────────────────┘
                        │ Feature Schema (22 Features)
                        ▼
┌────────────────────────────────────────────────────────┐
│          Predictive Intelligence & Forecasting         │
│  ├─ 6-Hour Validated XGBoost Baseline Regressor        │
│  ├─ Multi-Horizon Transitions (+12h, +24h, +48h)       │
│  ├─ Pollution Episode Duration & Decay Predictor       │
│  └─ Precaution Advisory & Explainability Engine        │
└───────────────────────┬────────────────────────────────┘
                        │ Structured Output
                        ▼
┌────────────────────────────────────────────────────────┐
│     React 18 + Vite + Tailwind Executive Dashboard     │
│  ├─ Overview & Live Trajectory Charts                  │
│  ├─ Video Analysis Workspace & H.264 Web Player        │
│  ├─ Air Quality Intelligence & CPCB Sub-Indices        │
│  ├─ Multi-Horizon Trajectory Inspector                 │
│  ├─ Delhi-NCR 82-Station Network Map & Telemetry       │
│  └─ Peer-Review Grade Research & Validation Specs      │
└────────────────────────────────────────────────────────┘
```

---

## 🔬 Scientific Integrity & Ethical Notice

* **Visual Evidence vs. Mass Concentration:** Smoke/fire bounding boxes and plume segmentations represent *optical evidence* (photometric attenuation and spatial obscuration). They do **not** constitute direct mass concentration ($\mu\text{g/m}^3$) measurements.
* **CPCB 3-Pollutant Mandate:** In accordance with official CPCB NAQI (2014/2015) rules, an overall AQI score is **only** reported if at least 3 criteria pollutants are present. If only particulate matter ($\text{PM}_{2.5}$ and $\text{PM}_{10}$) is monitored, the system flags `incomplete_pollutant_coverage` and displays sub-indices.
* **Controlled Station Fallback:** Station lag features are **never mixed**. If station A lacks historical lags, station B (within 10 km) is designated as the forecast station while station A remains the ground-truth station.
* **Historical Replay:** Operational live forecasting requires ECMWF/GFS live feeds; current forecasts evaluate historical replay against ERA5 reanalysis.

---

## ⚡ Quickstart & Local Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.15, CPU-only execution)
- Node.js 18+ and npm

### 1. Clone & Set Up Backend
```powershell
git clone https://github.com/muskan-101/diwali-plume.git
cd diwali-plume

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```powershell
python -m pytest tests/ -v
# 141 tests passing across 9 test modules
```

### 3. Launch the Unified Platform
```powershell
# Launches Flask backend serving compiled React frontend
python src/api/app.py --port 5000
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### 4. Optional: Run Frontend in Development Mode
```powershell
cd frontend
npm install
npm run dev
# Vite dev server running on http://localhost:3000 (proxies API to port 5000)
```

---

## 🧪 Evaluation & Test Coverage

| Test Module | Tests | Scope |
|---|:---:|---|
| `test_api.py` | 22 | Health endpoints, input validation, output security, end-to-end integration |
| `test_cpcb_aqi.py` | 35 | CPCB sub-index formula, 3-pollutant mandate, boundary values |
| `test_cv_intelligence.py` | 10 | Plume segmenter, Farneback optical flow, clarity estimator, event classifier |
| `test_episode_predictor.py` | 10 | Severe episode exceedance detection, survival decay modeling |
| `test_forecast_engine.py` | 17 | Baseline XGBoost inference, feature construction, cyclical hour/day encodings |
| `test_multi_horizon_forecaster.py` | 3 | 6h/12h/24h/48h horizons, confidence intervals, rich CV fusion |
| `test_precaution_engine.py` | 8 | Alert levels, N95 respirators, vulnerable groups, explainability triggers |
| `test_station_matcher.py` | 21 | Haversine distance, temporal reading alignment, controlled fallback stations |
| `test_video_analyzer.py` | 15 | YOLO inference, synthetic frames, video decoding verification |
| **Total** | **141** | **100% Passing** |

---

## 📜 License
This project is open-sourced under the MIT License.
