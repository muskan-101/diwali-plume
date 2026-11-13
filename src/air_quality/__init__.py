# src/air_quality/__init__.py
"""
Diwali Plume — Air Quality and CPCB AQI Calculation Module
"""
from src.air_quality.cpcb_aqi import CPCBAQICalculator, AQIResult, SubIndexResult

__all__ = ["CPCBAQICalculator", "AQIResult", "SubIndexResult"]
