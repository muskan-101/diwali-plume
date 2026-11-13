"""
test_precaution_engine.py — Tests for the Precaution and Environmental Intelligence Engine.

Covers:
- Alert level assignment: EMERGENCY_SEVERE, HIGH_ALERT, MODERATE_WARNING, LOW_RISK
- N95 respiratory protection triggers
- Indoor air purifier guidance
- Vulnerable population recommendations (children, elderly, respiratory/cardiovascular)
- Explainability triggers (visual plume, smoke detections, atmospheric stagnation, measured AQI)
- Medical disclaimer presence
- Fallback/sub-index only handling
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.intelligence.precaution_engine import PrecautionEngine, PrecautionReport


class TestPrecautionEngine:
    def test_emergency_alert_on_severe_pm25(self):
        """AQI in severe range (>400 or PM2.5 > 250) triggers EMERGENCY_SEVERE alert."""
        report = PrecautionEngine.generate_recommendations(
            aqi_value=450,
            aqi_category="Severe",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 300.0, "pm10": 450.0},
            is_official_aqi=True,
        )
        assert isinstance(report, PrecautionReport)
        assert report.alert_level == "EMERGENCY_SEVERE"
        assert "N95" in report.respiratory_protection or "mask" in report.respiratory_protection.lower()
        assert len(report.guidance_children_and_elderly) > 0
        assert len(report.guidance_respiratory_cardiac) > 0

    def test_high_alert_on_very_poor(self):
        """AQI 301-400 triggers HIGH_ALERT."""
        report = PrecautionEngine.generate_recommendations(
            aqi_value=350,
            aqi_category="Very Poor",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 160.0},
            is_official_aqi=False,
        )
        assert report.alert_level == "HIGH_ALERT"

    def test_moderate_warning_on_poor_aqi(self):
        report = PrecautionEngine.generate_recommendations(
            aqi_value=250,
            aqi_category="Poor",
            dominant_pollutant="pm10",
            pollutant_concentrations={"pm10": 260.0},
            is_official_aqi=True,
        )
        assert report.alert_level == "MODERATE_WARNING"

    def test_low_risk_on_good_aqi(self):
        report = PrecautionEngine.generate_recommendations(
            aqi_value=40,
            aqi_category="Good",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 12.0},
            is_official_aqi=True,
        )
        assert report.alert_level == "LOW_RISK"

    def test_visual_cues_captured_in_explainability(self):
        """Smoke detections and plume coverage in CV summary are surfaced in triggers."""
        cv_summary = {
            "total_smoke_detections": 25,
            "total_fire_detections": 0,
            "pct_frames_with_smoke": 18.5,
            "max_plume_coverage_pct": 35.0,
            "haze_severity": "Moderate Haze",
        }
        report = PrecautionEngine.generate_recommendations(
            aqi_value=250,
            aqi_category="Poor",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 100.0},
            is_official_aqi=False,
            cv_summary=cv_summary,
        )
        assert len(report.detected_visual_cues) > 0
        assert any("smoke" in c.lower() or "plume" in c.lower() or "haze" in c.lower() for c in report.detected_visual_cues)

    def test_meteorological_stagnation_risk(self):
        """Low wind and low boundary layer height flag atmospheric stagnation."""
        weather = {
            "wind_speed_10m": 0.5,
            "boundary_layer_height": 100.0,
        }
        report = PrecautionEngine.generate_recommendations(
            aqi_value=320,
            aqi_category="Very Poor",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 150.0},
            is_official_aqi=False,
            weather_data=weather,
        )
        assert report.meteorological_stagnation_risk != ""

    def test_medical_disclaimer_always_present(self):
        report = PrecautionEngine.generate_recommendations(
            aqi_value=50,
            aqi_category="Good",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 15.0},
            is_official_aqi=True,
        )
        assert isinstance(report.medical_disclaimer, str)
        assert len(report.medical_disclaimer) > 20
        assert "medical" in report.medical_disclaimer.lower()

    def test_data_provenance_reported(self):
        """Provenance distinguishes measured vs forecast."""
        report = PrecautionEngine.generate_recommendations(
            aqi_value=220,
            aqi_category="Poor",
            dominant_pollutant="pm25",
            pollutant_concentrations={"pm25": 95.0},
            is_official_aqi=False,
            forecast_pm25=280.0,
        )
        assert report.data_provenance != ""
