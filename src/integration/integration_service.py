"""
integration_service.py — Diwali Plume Multimodal Environmental Intelligence & Episode Forecasting Engine

Orchestrates:
  1. Computer Vision Intelligence Pipeline (CVIntelligencePipeline)
     - YOLO smoke/fire detection with temporal consistency
     - Physics-grounded plume & smoke segmentation (coverage, centroid, growth rate)
     - Dense optical flow motion analysis (separating camera vs. plume motion)
     - Dark Channel Prior haze estimation & visual scene clarity index
     - Multi-cue event classification & visual trend plotting
  2. Environmental Station Matching (StationMatcher)
     - Nearest measured ground-truth station identification
     - Controlled forecast station fallback (<= 10 km) when nearest lacks complete lags
  3. Official CPCB Air Quality Index & Sub-Indices (CPCBAQICalculator)
     - Complete 8-criteria breakpoint calculations
     - Minimum 3-pollutant criteria check (distinguishing official vs. sub-index-only)
  4. Multimodal Multi-Horizon Forecasting (MultiHorizonForecaster)
     - Preserves validated 6h XGBoost baseline checkpoint
     - Autoregressive diurnal transitions for 12h, 24h, 48h horizons
     - Model 1 (Environmental), Model 2 (Basic visual), Model 3 (Rich CV) configurations
  5. Pollution Episode Persistence & Duration Prediction (EpisodePredictor)
     - Detects active CPCB Severe episodes (PM2.5 >= 250 µg/m³ sustained >= 3h)
     - Predicts expected remaining duration, peak, improvement time, and decay trajectory
  6. Precaution & Environmental Intelligence Engine (PrecautionEngine)
     - Public health advisories, N95 mask guidance, indoor air management
     - Vulnerable population guidance (children, elderly, respiratory/cardiovascular)
     - Multimodal trigger explainability narratives
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from src.integration.station_matcher import StationMatcher, StationMatcherConfig
from src.integration.forecast_engine import ForecastEngine
from src.integration.video_analyzer import VideoAnalyzer
from src.vision.cv_intelligence import CVIntelligencePipeline
from src.air_quality.cpcb_aqi import CPCBAQICalculator
from src.forecasting.multi_horizon_forecaster import MultiHorizonForecaster
from src.forecasting.episode_predictor import EpisodePredictor
from src.intelligence.precaution_engine import PrecautionEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data/outputs")
DEFAULT_MODEL_PATH = Path("models/vision/fire_smoke_yolov8n_dfire_best.pt")
DEFAULT_XGB_PATH = Path("models/xgboost_pm25_6h.pkl")
DEFAULT_FEATURES_PATH = Path("models/baseline_features.json")
DEFAULT_DATA_PATH = Path("data/processed/delhi_multimodal_clean.csv")


# ---------------------------------------------------------------------------
# IntegrationService
# ---------------------------------------------------------------------------

class IntegrationService:
    """
    Top-level service combining Vision Intelligence, CPCB AQI, Multimodal
    Forecasting, Episode Persistence, and Health Precautions.
    """

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        model_path: Path = DEFAULT_MODEL_PATH,
        xgb_path: Path = DEFAULT_XGB_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH,
        data_path: Path = DEFAULT_DATA_PATH,
        max_distance_km: float = 50.0,
        max_reading_age_hours: float = 3.0,
        max_fallback_km: float = 10.0,
        vision_confidence: float = 0.15,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_fallback_km = max_fallback_km

        # Legacy-compatible video analyzer for backwards compatibility with tests
        self._video_analyzer = VideoAnalyzer(
            model_path=model_path,
            confidence=vision_confidence,
        )

        # Advanced Vision Intelligence Pipeline
        self._cv_pipeline = CVIntelligencePipeline(
            model_path=model_path,
            conf_threshold=vision_confidence,
        )

        # Station Matcher with Fallback capability
        self._station_matcher = StationMatcher(
            config=StationMatcherConfig(
                max_distance_km=max_distance_km,
                max_reading_age_hours=max_reading_age_hours,
                data_path=data_path,
            )
        )

        # Baseline 6h Forecast Engine (preserved)
        self._forecast_engine = ForecastEngine(
            model_path=xgb_path,
            features_path=features_path,
            data_path=data_path,
        )

        # Multi-Horizon Multimodal Forecaster (6h, 12h, 24h, 48h + CV fusion)
        self._multi_forecaster = MultiHorizonForecaster(
            model_path=xgb_path,
            features_path=features_path,
            data_path=data_path,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_path: Path,
        latitude: float,
        longitude: float,
        recording_datetime: datetime,
        mode: str = "historical_replay",
        result_id: Optional[str] = None,
        max_sampled_frames: int = 250,
    ) -> dict:
        result_id = result_id or str(uuid.uuid4())
        warnings: list[str] = []
        result_subdir = self.output_dir / result_id
        result_subdir.mkdir(parents=True, exist_ok=True)

        # Mode guard
        if mode != "historical_replay":
            warnings.append(
                f"Mode '{mode}' is not supported. "
                "Only 'historical_replay' is available. Live operational forecasting requires "
                "active GFS/ECMWF weather integration."
            )

        # Normalise datetime to UTC
        if recording_datetime.tzinfo is None:
            recording_datetime = recording_datetime.replace(tzinfo=timezone.utc)
        else:
            recording_datetime = recording_datetime.astimezone(timezone.utc)

        # ==============================================================
        # Step 1 — Advanced Computer Vision Intelligence
        # ==============================================================
        logger.info("Step 1: Running CV Intelligence Pipeline on %s", video_path)
        cv_summary = self._cv_pipeline.analyze_video(
            video_path=video_path,
            output_dir=result_subdir,
            output_stem="annotated",
            max_sampled_frames=max_sampled_frames,
        )

        if not cv_summary.success:
            warnings.append(f"Video analysis failed: {cv_summary.error_message}")
        if cv_summary.confidence_uncertainty_notes:
            warnings.extend(cv_summary.confidence_uncertainty_notes[:1])

        # ==============================================================
        # Step 2 — Station Matching (Nearest Measured & Forecast Fallback)
        # ==============================================================
        logger.info("Step 2: Station matching for lat=%.4f, lon=%.4f", latitude, longitude)
        nearest_station = self._station_matcher.find_nearest_station(
            latitude=latitude,
            longitude=longitude,
            recording_datetime=recording_datetime,
        )
        if nearest_station.warnings:
            warnings.extend(nearest_station.warnings)

        # Fallback Station for Forecasting
        forecast_station_match, is_fallback, fallback_note, missing_lags = self._station_matcher.find_forecast_station(
            latitude=latitude,
            longitude=longitude,
            issue_time=recording_datetime,
            max_fallback_km=self.max_fallback_km,
        )
        if fallback_note:
            warnings.append(fallback_note)

        # ==============================================================
        # Step 3 — Official CPCB AQI & Sub-Indices Calculation
        # ==============================================================
        logger.info("Step 3: Calculating CPCB AQI and Sub-Indices")
        pollutants_dict = {}
        if nearest_station.is_valid and nearest_station.pm25 is not None:
            pollutants_dict["pm25"] = nearest_station.pm25

        # Check if PM10 is available at the nearest station
        nearest_weather = None
        if nearest_station.is_valid and nearest_station.station_id >= 0:
            nearest_weather = self._station_matcher.get_weather_at(
                nearest_station.station_id, recording_datetime
            )
            if nearest_weather is not None and "pm10" in nearest_weather and not nearest_weather.isna()["pm10"]:
                pollutants_dict["pm10"] = float(nearest_weather["pm10"])

        cpcb_result = CPCBAQICalculator.calculate_aqi(pollutants_dict)
        if cpcb_result.warnings:
            warnings.extend(cpcb_result.warnings)

        # ==============================================================
        # Step 4 — Multimodal Multi-Horizon Forecasting
        # ==============================================================
        logger.info("Step 4: Multimodal Multi-Horizon Forecasting")
        active_forecast_station = forecast_station_match if (forecast_station_match and forecast_station_match.is_valid) else nearest_station
        forecast_result = None

        if active_forecast_station.is_valid and active_forecast_station.station_id >= 0:
            st_history = self._station_matcher.get_station_pm25_history(
                active_forecast_station.station_id, recording_datetime, hours_back=48
            )
            st_weather = self._station_matcher.get_weather_at(
                active_forecast_station.station_id, recording_datetime
            )

            cv_stats_dict = {
                "total_smoke_detections": cv_summary.total_smoke_detections,
                "total_fire_detections": cv_summary.total_fire_detections,
                "pct_frames_with_smoke": cv_summary.pct_frames_with_smoke,
                "pct_frames_with_fire": cv_summary.pct_frames_with_fire,
                "avg_smoke_confidence": cv_summary.avg_smoke_confidence,
                "max_plume_coverage_pct": cv_summary.max_plume_coverage_pct,
                "avg_apparent_motion_speed": cv_summary.avg_apparent_motion_speed,
                "avg_dark_channel": cv_summary.avg_dark_channel,
                "avg_scene_clarity_index": cv_summary.avg_scene_clarity_index,
            }

            forecast_result = self._multi_forecaster.forecast_multimodal(
                forecast_station_id=active_forecast_station.station_id,
                forecast_station_name=active_forecast_station.station_name,
                forecast_station_distance_km=active_forecast_station.distance_km,
                is_fallback=is_fallback,
                fallback_note=fallback_note,
                issue_time=recording_datetime,
                station_history_df=st_history,
                weather_row=st_weather,
                station_lat=active_forecast_station.station_latitude,
                station_lon=active_forecast_station.station_longitude,
                cv_summary=cv_stats_dict,
                configuration="rich_cv_fusion",
            )
            if forecast_result.warnings:
                warnings.extend(forecast_result.warnings)
        else:
            warnings.append("Forecast skipped: No eligible monitoring station found.")

        # ==============================================================
        # Step 5 — Pollution Episode Detection & Persistence Prediction
        # ==============================================================
        logger.info("Step 5: Episode Persistence Modeling")
        episode_prediction = None
        if nearest_station.is_valid and nearest_station.station_id >= 0:
            st_history = self._station_matcher.get_station_pm25_history(
                nearest_station.station_id, recording_datetime, hours_back=72
            )
            ep_status = EpisodePredictor.detect_episode(st_history, recording_datetime)
            weather_dict = nearest_weather.to_dict() if nearest_weather is not None else {}
            episode_prediction = EpisodePredictor.predict_persistence(
                ep_status, recording_datetime, weather_dict
            )
            if episode_prediction.warnings:
                warnings.extend(episode_prediction.warnings)

        # ==============================================================
        # Step 6 — Precaution Recommendation Engine & Explainability
        # ==============================================================
        logger.info("Step 6: Generating Actionable Precautions & Explainability")
        weather_dict = nearest_weather.to_dict() if nearest_weather is not None else {}
        cv_dict = {
            "total_smoke_detections": cv_summary.total_smoke_detections,
            "total_fire_detections": cv_summary.total_fire_detections,
            "pct_frames_with_smoke": cv_summary.pct_frames_with_smoke,
            "max_plume_coverage_pct": cv_summary.max_plume_coverage_pct,
            "haze_severity": cv_summary.haze_severity,
        }
        pred_6h_val = forecast_result.horizons[6].predicted_pm25_ugm3 if (forecast_result and 6 in forecast_result.horizons) else None

        precaution_report = PrecautionEngine.generate_recommendations(
            aqi_value=cpcb_result.aqi,
            aqi_category=cpcb_result.category,
            dominant_pollutant=cpcb_result.dominant_pollutant,
            pollutant_concentrations=pollutants_dict,
            is_official_aqi=cpcb_result.is_official_cpcb_compliant,
            cv_summary=cv_dict,
            weather_data=weather_dict,
            forecast_pm25=pred_6h_val,
        )

        # ==============================================================
        # Step 7 — Structured Response Assembly
        # ==============================================================
        return _build_comprehensive_response(
            result_id=result_id,
            mode=mode,
            recording_datetime=recording_datetime,
            user_latitude=latitude,
            user_longitude=longitude,
            cv_summary=cv_summary,
            nearest_station=nearest_station,
            forecast_station=active_forecast_station,
            is_fallback=is_fallback,
            fallback_note=fallback_note,
            cpcb_result=cpcb_result,
            forecast_result=forecast_result,
            episode_prediction=episode_prediction,
            precaution_report=precaution_report,
            warnings=warnings,
        )

    def health(self) -> dict:
        return {
            "vision_model_available": self._video_analyzer.model_available,
            "forecast_model_available": self._forecast_engine.model_available,
            "feature_schema_available": self._forecast_engine.features_available,
            "dataset_available": self._forecast_engine.dataset_available,
            "cpcb_engine_available": True,
            "episode_engine_available": True,
            "precaution_engine_available": True,
        }


# ---------------------------------------------------------------------------
# Structured JSON Response Formatter
# ---------------------------------------------------------------------------

def _build_comprehensive_response(
    result_id: str,
    mode: str,
    recording_datetime: datetime,
    user_latitude: float,
    user_longitude: float,
    cv_summary,
    nearest_station,
    forecast_station,
    is_fallback: bool,
    fallback_note: Optional[str],
    cpcb_result,
    forecast_result,
    episode_prediction,
    precaution_report,
    warnings: list[str],
) -> dict:
    # 1. Video Analysis Section (Strict backwards-compatible fields + rich CV features)
    video_sec = {
        "success": cv_summary.success,
        "error": cv_summary.error_message,
        "total_frames": cv_summary.total_frames,
        "sampled_frames": cv_summary.sampled_frames,
        "sampling_stride": cv_summary.sampling_stride,
        "duration_seconds": cv_summary.duration_seconds,
        "fps": cv_summary.fps,
        "frame_dimensions": f"{cv_summary.frame_width}x{cv_summary.frame_height}",
        "frames_with_smoke": cv_summary.frames_with_smoke,
        "frames_with_fire": cv_summary.frames_with_fire,
        "smoke_detection_count": cv_summary.total_smoke_detections,
        "fire_detection_count": cv_summary.total_fire_detections,
        "avg_smoke_confidence": cv_summary.avg_smoke_confidence,
        "avg_fire_confidence": cv_summary.avg_fire_confidence,
        "max_confidence": cv_summary.max_confidence,
        "pct_frames_with_smoke": cv_summary.pct_frames_with_smoke,
        "pct_frames_with_fire": cv_summary.pct_frames_with_fire,
        "annotated_video_path": cv_summary.annotated_video_path,
        "detection_csv_path": cv_summary.timeline_csv_path,
        "features_csv_path": cv_summary.features_csv_path,
        "visual_trends_graph_path": cv_summary.trend_graph_path,
        # Plume segmentation & dynamics
        "max_plume_coverage_pct": cv_summary.max_plume_coverage_pct,
        "avg_plume_coverage_pct": cv_summary.avg_plume_coverage_pct,
        "plume_expansion_trend": cv_summary.plume_expansion_trend,
        "dominant_plume_direction": cv_summary.dominant_plume_direction,
        "apparent_motion_speed": cv_summary.avg_apparent_motion_speed,
        "camera_motion": cv_summary.camera_motion_assessment,
        # Haze & clarity
        "dark_channel_mean": cv_summary.avg_dark_channel,
        "scene_clarity_index": cv_summary.avg_scene_clarity_index,
        "haze_severity": cv_summary.haze_severity,
        "primary_event_classification": cv_summary.primary_event_classification,
        "event_confidence_score": cv_summary.event_confidence_score,
        "caveat": (
            "Smoke/fire detections and plume segmentations represent OPTICAL VISUAL EVIDENCE. "
            "They do not constitute PM2.5 measurements and do not establish a causal link to measured air quality."
        ),
    }

    # 2. Nearest Station (Measured Ground Truth)
    nearest_sec = {
        "found": nearest_station.is_valid,
        "station_id": nearest_station.station_id if nearest_station.station_id >= 0 else None,
        "station_name": nearest_station.station_name or None,
        "station_latitude": nearest_station.station_latitude if nearest_station.is_valid else None,
        "station_longitude": nearest_station.station_longitude if nearest_station.is_valid else None,
        "distance_km": nearest_station.distance_km if nearest_station.is_valid else None,
        "pm25_ugm3": nearest_station.pm25,
        "reading_timestamp": nearest_station.reading_timestamp.isoformat() if nearest_station.reading_timestamp else None,
        "reading_age_hours": nearest_station.reading_age_hours,
        "pm25_sensor_count": nearest_station.pm25_sensor_count,
        "data_quality": nearest_station.data_quality,
        "caveat": "Represents ground-truth sensor measurements at the physical monitoring station.",
    }

    # 3. Forecast Station (Controlled Fallback Distinction)
    forecast_station_sec = {
        "station_id": forecast_station.station_id if (forecast_station and forecast_station.is_valid) else None,
        "station_name": forecast_station.station_name if (forecast_station and forecast_station.is_valid) else None,
        "distance_km": forecast_station.distance_km if (forecast_station and forecast_station.is_valid) else None,
        "is_fallback": is_fallback,
        "fallback_note": fallback_note,
    }

    # 4. CPCB AQI Section
    cpcb_sub_indices = {}
    for p, sub in cpcb_result.sub_indices.items():
        cpcb_sub_indices[p] = {
            "pollutant": sub.pollutant,
            "concentration": sub.concentration,
            "unit": sub.unit,
            "sub_index": sub.sub_index,
            "category": sub.category,
            "is_valid": sub.is_valid,
        }

    cpcb_sec = {
        "aqi": cpcb_result.aqi,
        "category": cpcb_result.category,
        "color": cpcb_result.color,
        "health_statement": cpcb_result.health_statement,
        "dominant_pollutant": cpcb_result.dominant_pollutant,
        "is_official_cpcb_compliant": cpcb_result.is_official_cpcb_compliant,
        "status": cpcb_result.status,
        "monitored_pollutants_count": cpcb_result.monitored_pollutants_count,
        "sub_indices": cpcb_sub_indices,
        "source_documentation": cpcb_result.source_documentation,
    }

    # 5. Multi-Horizon & 6-Hour Legacy Forecast
    pm25_forecast_6h = {"available": False}
    multi_horizon_sec = {"available": False, "horizons": {}}

    if forecast_result and 6 in forecast_result.horizons:
        f6 = forecast_result.horizons[6]
        pm25_forecast_6h = {
            "available": True,
            "forecast_pm25_ugm3": f6.predicted_pm25_ugm3,
            "forecast_horizon_hours": 6,
            "issue_timestamp": forecast_result.issue_timestamp.isoformat(),
            "target_timestamp": f6.target_timestamp.isoformat(),
            "data_quality": forecast_result.data_quality,
            "confidence_interval_90": f6.confidence_interval_90,
            "caveat": (
                "6-hour PM2.5 prediction produced by XGBoost baseline model fused with visual telemetry. "
                "Weather features represent ERA5 historical reanalysis. Values reported in µg/m³."
            ),
        }

        multi_horizon_sec = {
            "available": True,
            "forecast_station": forecast_result.forecast_station_name,
            "active_configuration": forecast_result.active_configuration,
            "horizons": {
                h: {
                    "horizon_hours": obj.horizon_hours,
                    "target_timestamp": obj.target_timestamp.isoformat(),
                    "predicted_pm25_ugm3": obj.predicted_pm25_ugm3,
                    "confidence_interval_90": obj.confidence_interval_90,
                    "status": obj.status,
                }
                for h, obj in forecast_result.horizons.items()
            },
        }

    # 6. Pollution Episode Prediction
    episode_sec = {"available": False}
    if episode_prediction:
        ep_st = episode_prediction.episode_status
        episode_sec = {
            "available": True,
            "is_in_episode": ep_st.is_in_episode,
            "current_status": ep_st.current_status,
            "threshold_ugm3": ep_st.episode_threshold_ugm3,
            "current_pm25": ep_st.current_pm25,
            "elapsed_duration_hours": ep_st.elapsed_duration_hours,
            "episode_start_time": ep_st.episode_start_time.isoformat() if ep_st.episode_start_time else None,
            "peak_pm25_observed": ep_st.peak_pm25_observed,
            "peak_timestamp": ep_st.peak_timestamp.isoformat() if ep_st.peak_timestamp else None,
            "recent_trend": ep_st.recent_trend,
            "expected_remaining_hours": episode_prediction.expected_remaining_hours,
            "expected_improvement_time": episode_prediction.expected_improvement_time.isoformat() if episode_prediction.expected_improvement_time else None,
            "confidence_interval_hours": episode_prediction.confidence_interval_hours,
            "persistence_risk_score": episode_prediction.persistence_risk_score,
            "meteorological_dispersion": episode_prediction.meteorological_dispersion_assessment,
            "projected_decay_trajectory": episode_prediction.projected_decay_trajectory,
        }

    # 7. Precaution & Intelligence Advisory
    precaution_sec = {
        "alert_level": precaution_report.alert_level,
        "basis_of_recommendation": precaution_report.basis_of_recommendation,
        "data_provenance": precaution_report.data_provenance,
        "general_advice": precaution_report.general_public_advice,
        "outdoor_guidance": precaution_report.outdoor_activity_guidance,
        "respiratory_protection": precaution_report.respiratory_protection,
        "indoor_air_management": precaution_report.indoor_air_management,
        "guidance_children_elderly": precaution_report.guidance_children_and_elderly,
        "guidance_respiratory_cardiac": precaution_report.guidance_respiratory_cardiac,
        "trigger_explainability": precaution_report.trigger_summary,
        "responsible_pollutants": precaution_report.responsible_pollutants,
        "detected_visual_cues": precaution_report.detected_visual_cues,
        "meteorological_stagnation_risk": precaution_report.meteorological_stagnation_risk,
        "medical_disclaimer": precaution_report.medical_disclaimer,
    }

    return {
        "result_id": result_id,
        "mode": mode,
        "recording_datetime": recording_datetime.isoformat(),
        "user_location": {
            "latitude": user_latitude,
            "longitude": user_longitude,
        },
        "success": cv_summary.success or nearest_station.is_valid,
        "warnings": warnings,
        "video_analysis": video_sec,
        "nearest_station": nearest_sec,
        "forecast_station": forecast_station_sec,
        "air_quality_cpcb": cpcb_sec,
        "pm25_forecast_6h": pm25_forecast_6h,
        "multi_horizon_forecast": multi_horizon_sec,
        "pollution_episode": episode_sec,
        "precaution_advisory": precaution_sec,
        "system_note": (
            "Diwali Plume — Vision-Led Environmental Intelligence System. "
            "Combines visual smoke/plume evidence, CPCB AQI sub-indices, controlled station fallback, "
            "multi-horizon forecasting, episode persistence prediction, and health precautions."
        ),
    }
