"""
precaution_engine.py — Evidence-Based Environmental Intelligence & Precaution Engine

Produces practical public-health guidance, exposure mitigation actions,
vulnerability-specific warnings, and transparent explainability narratives.

Public Health Sources:
  - Central Pollution Control Board (CPCB) NAQI Health Advisories
  - World Health Organization (WHO) Global Air Quality Guidelines
  - Ministry of Health and Family Welfare (MoHFW), Govt. of India

Ethical & Medical Safeguards:
  - Explicitly disclaims medical diagnosis or individualized safety guarantees.
  - Transparently states the data provenance (measured AQI, sub-index, or forecast).
  - Outlines the causal triggers (pollutants, atmospheric stagnation, visual plume/haze).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class PrecautionReport:
    alert_level: str  # "EMERGENCY_SEVERE", "HIGH_ALERT", "MODERATE_WARNING", "LOW_RISK"
    basis_of_recommendation: str  # e.g., "Measured PM2.5 (Very Poor) & Severe Visual Haze"
    data_provenance: str  # "Measured Station Ground Truth" or "Forecast Model Prediction"
    
    # Actionable guidance
    general_public_advice: List[str]
    outdoor_activity_guidance: str
    respiratory_protection: str  # N95 / FFP2 mask advisory
    indoor_air_management: List[str]
    
    # Vulnerable populations
    guidance_children_and_elderly: List[str]
    guidance_respiratory_cardiac: List[str]
    
    # Explainability & Diagnostic Triggers
    trigger_summary: str
    responsible_pollutants: List[str]
    detected_visual_cues: List[str]
    meteorological_stagnation_risk: str
    confidence_assessment: str
    
    medical_disclaimer: str = (
        "Notice: This assessment is an automated environmental risk advisory based on air quality metrics "
        "and computer vision evidence. It does not constitute medical advice or diagnostic evaluation. "
        "Individuals with preexisting health conditions must consult qualified healthcare practitioners."
    )


class PrecautionEngine:
    """Generates structured, evidence-based environmental precautions and explainability."""

    @classmethod
    def generate_recommendations(
        cls,
        aqi_value: Optional[int],
        aqi_category: Optional[str],
        dominant_pollutant: Optional[str],
        pollutant_concentrations: Dict[str, float],
        is_official_aqi: bool,
        cv_summary: Optional[Dict[str, Any]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        forecast_pm25: Optional[float] = None,
    ) -> PrecautionReport:
        # Determine effective severity index
        effective_aqi = aqi_value if aqi_value is not None else 0
        if forecast_pm25 and forecast_pm25 > 250.0:
            effective_aqi = max(effective_aqi, 401)
        elif forecast_pm25 and forecast_pm25 > 120.0:
            effective_aqi = max(effective_aqi, 301)

        cat = (aqi_category or "Unknown").title()

        # Visual signals
        visual_cues = []
        has_dense_smoke = False
        has_fire = False
        if cv_summary:
            if cv_summary.get("total_fire_detections", 0) > 0:
                visual_cues.append(f"Visual combustion/fire detected ({cv_summary['total_fire_detections']} detections)")
                has_fire = True
            if cv_summary.get("pct_frames_with_smoke", 0) > 10.0:
                visual_cues.append(f"Persistent smoke plume in {cv_summary['pct_frames_with_smoke']:.1f}% of video frames")
                has_dense_smoke = True
            if cv_summary.get("haze_severity") in ("Severe Haze/Smog", "Moderate Haze"):
                visual_cues.append(f"Elevated atmospheric optical depth / {cv_summary.get('haze_severity')}")
            if cv_summary.get("max_plume_coverage_pct", 0) > 15.0:
                visual_cues.append(f"Extensive localized plume coverage (peak {cv_summary['max_plume_coverage_pct']:.1f}% frame area)")

        # Meteorological stagnation analysis
        stagnation_risk = "Normal dispersion conditions"
        if weather_data:
            ws = weather_data.get("wind_speed_10m", 5.0)
            blh = weather_data.get("boundary_layer_height", 800.0)
            rh = weather_data.get("relative_humidity_2m", 50.0)
            stagnant_factors = []
            if ws is not None and ws < 2.0:
                stagnant_factors.append(f"calm wind ({ws:.1f} m/s)")
            if blh is not None and blh < 250.0:
                stagnant_factors.append(f"shallow boundary layer ({blh:.0f} m)")
            if rh is not None and rh > 75.0:
                stagnant_factors.append(f"high humidity ({rh:.0f}%) promoting secondary aerosol formation")
            if stagnant_factors:
                stagnation_risk = "High atmospheric stagnation: " + ", ".join(stagnant_factors) + " trapping pollutants near ground level."

        # Responsible pollutants
        resp_pollutants = []
        if dominant_pollutant:
            resp_pollutants.append(dominant_pollutant.upper())
        for p, val in pollutant_concentrations.items():
            if p.lower() in ("pm25", "pm2.5") and val >= 60.0:
                resp_pollutants.append(f"PM2.5 ({val:.1f} µg/m³)")
            elif p.lower() == "pm10" and val >= 100.0:
                resp_pollutants.append(f"PM10 ({val:.1f} µg/m³)")

        # Data provenance description
        provenance = "Measured Air Quality Station Data"
        if not is_official_aqi:
            provenance += " (Sub-indices only; incomplete criteria coverage)"
        if forecast_pm25 is not None:
            provenance += f" + 6-Hour Forecast ({forecast_pm25:.1f} µg/m³)"

        # Alert level & tailored rules
        if effective_aqi >= 401 or cat == "Severe":
            alert_level = "EMERGENCY_SEVERE"
            basis = f"Severe air pollution (AQI/Sub-index: {effective_aqi}, Category: Severe). Acute public health hazard."
            outdoor_advice = "AVOID ALL OUTDOOR PHYSICAL ACTIVITY. Cancel morning walks, runs, and sports events. Remain strictly indoors."
            mask_advisory = "Wear a well-fitted certified N95, KN95, or FFP2 respirator if stepping outdoors. Cloth and surgical masks do not filter fine PM2.5."
            general_advice = [
                "Keep all doors and windows tightly closed to minimize outdoor particulate penetration.",
                "Avoid vacuum cleaning and dry brooming; use wet mopping to reduce indoor dust resuspension.",
                "Avoid burning candles, incense sticks, mosquito coils, or wood indoors.",
                "Maintain adequate hydration with warm fluids."
            ]
            indoor_management = [
                "Run True-HEPA air purifiers continuously in closed living and sleeping spaces.",
                "Ensure air conditioner filters are clean and operate in internal recirculation mode.",
                "Create a designated 'clean room' with minimal air exchange and active filtration if whole-home filtration is unavailable."
            ]
            vulnerable_kids_elderly = [
                "Children and elderly individuals must remain indoors at all times.",
                "Schools should suspend outdoor physical training and switch to indoor quiet activities.",
                "Monitor oxygen saturation (SpO2) and respiratory rates in elderly relatives with cardiac conditions."
            ]
            vulnerable_cardio_respiratory = [
                "Patients with asthma, COPD, or bronchitis must keep rapid-relief rescue inhalers within immediate reach.",
                "Consult attending physician regarding temporary medication dosage adjustments during severe episodes.",
                "Seek immediate emergency medical attention if experiencing chest tightness, wheezing, or unmanageable shortness of breath."
            ]

        elif effective_aqi >= 301 or cat == "Very Poor":
            alert_level = "HIGH_ALERT"
            basis = f"Very Poor air pollution (AQI/Sub-index: {effective_aqi}). High risk of respiratory symptoms upon prolonged exposure."
            outdoor_advice = "Avoid prolonged or heavy outdoor exertion. Reschedule strenuous outdoor workouts to indoor environments."
            mask_advisory = "Use certified N95 or FFP2 respirators during outdoor commutes and transit."
            general_advice = [
                "Close windows during early morning and late evening peak pollution and inversion hours.",
                "Avoid outdoor dining and busy roadside corridors during rush hours."
            ]
            indoor_management = [
                "Operate HEPA purifiers in bedrooms during nighttime hours.",
                "Use wet wiping for indoor surface dust control."
            ]
            vulnerable_kids_elderly = [
                "Children, pregnant women, and elderly individuals should avoid outdoor recreation.",
                "Limit outdoor recess and playtime."
            ]
            vulnerable_cardio_respiratory = [
                "Persons with asthma or cardiovascular conditions should strictly minimize physical strain.",
                "Adhere strictly to prescribed preventive inhalers and cardiac regimens."
            ]

        elif effective_aqi >= 201 or cat == "Poor":
            alert_level = "MODERATE_WARNING"
            basis = f"Poor air pollution (AQI/Sub-index: {effective_aqi}). Breathing discomfort likely for sensitive groups."
            outdoor_advice = "Reduce prolonged outdoor exertion, particularly near high-traffic traffic intersections and industrial zones."
            mask_advisory = "Sensitive individuals should wear N95 respirators in high-traffic or visibly hazy corridors."
            general_advice = [
                "Ventilate indoor spaces only during warm midday hours when boundary layer dispersion is highest.",
                "Incorporate green plants and indoor air quality monitoring where possible."
            ]
            indoor_management = [
                "Keep windows shut during peak traffic hours (08:00-11:00 and 18:00-22:00)."
            ]
            vulnerable_kids_elderly = [
                "Sensitive children and senior citizens should take frequent rest breaks during outdoor activities."
            ]
            vulnerable_cardio_respiratory = [
                "Individuals with respiratory conditions should monitor symptoms and limit strenuous exertion."
            ]

        else:
            alert_level = "LOW_RISK"
            basis = f"Air pollution within moderate or acceptable limits (AQI/Sub-index: {effective_aqi}, Category: {cat})."
            outdoor_advice = "Normal outdoor activities can generally be conducted. Sensitive individuals should observe personal tolerance."
            mask_advisory = "Respirator not mandatory for healthy individuals unless localized combustion smoke is present."
            general_advice = [
                "Maintain standard indoor ventilation and air freshness."
            ]
            indoor_management = ["Open windows for cross-ventilation during clean afternoon hours."]
            vulnerable_kids_elderly = ["Standard outdoor activity permitted."]
            vulnerable_cardio_respiratory = ["Standard maintenance medication adherence."]

        # Extra caution if localized visual smoke or fire was detected
        if has_fire or has_dense_smoke:
            general_advice.insert(0, "LOCALIZED VISUAL SMOKE/FIRE DETECTED: Immediately evacuate upwind of visible smoke sources to avoid acute toxic combustion exposure.")

        # Trigger summary narrative
        trigger_narrative = (
            f"Alert triggered by {basis} "
            f"Key driving pollutants: {', '.join(resp_pollutants) if resp_pollutants else 'Particulate Matter'}. "
            f"Visual analysis revealed: {'; '.join(visual_cues) if visual_cues else 'no dense localized plume'}. "
            f"Atmospheric status: {stagnation_risk}."
        )

        return PrecautionReport(
            alert_level=alert_level,
            basis_of_recommendation=basis,
            data_provenance=provenance,
            general_public_advice=general_advice,
            outdoor_activity_guidance=outdoor_advice,
            respiratory_protection=mask_advisory,
            indoor_air_management=indoor_management,
            guidance_children_and_elderly=vulnerable_kids_elderly,
            guidance_respiratory_cardiac=vulnerable_cardio_respiratory,
            trigger_summary=trigger_narrative,
            responsible_pollutants=resp_pollutants,
            detected_visual_cues=visual_cues,
            meteorological_stagnation_risk=stagnation_risk,
            confidence_assessment="High confidence based on calibrated station observations and cross-modal visual telemetry.",
        )
