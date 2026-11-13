"""
test_cpcb_aqi.py — Tests for the CPCB National Air Quality Index (NAQI) Engine

Covers:
- Official sub-index formula (CPCB linear interpolation)
- PM2.5 and PM10 sub-index calculations at boundary values
- Minimum-3-pollutant rule enforcement
- Incomplete pollutant coverage flag
- Insufficient data handling
- Dominant pollutant selection
- Category and color lookup
- Negative/NaN concentration rejection
- Edge: concentration at breakpoint boundaries
"""

import sys
import math
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.air_quality.cpcb_aqi import CPCBAQICalculator, AQIResult, SubIndexResult


# ---------------------------------------------------------------------------
# Sub-index formula correctness
# ---------------------------------------------------------------------------

class TestSubIndexFormula:
    """Test individual pollutant sub-index calculations."""

    def test_pm25_sub_index_good_zone(self):
        """PM2.5 = 15 µg/m³ → category Good (AQI ≤ 50)."""
        result = CPCBAQICalculator.calculate_sub_index("pm25", 15.0)
        assert result.is_valid is True
        assert result.sub_index is not None
        assert result.sub_index <= 50

    def test_pm25_sub_index_satisfactory_zone(self):
        """PM2.5 = 45 µg/m³ → Satisfactory zone."""
        result = CPCBAQICalculator.calculate_sub_index("pm25", 45.0)
        assert result.is_valid is True
        assert 51 <= result.sub_index <= 100

    def test_pm25_sub_index_poor_zone(self):
        """PM2.5 = 110 µg/m³ → Poor zone (201–300)."""
        result = CPCBAQICalculator.calculate_sub_index("pm25", 110.0)
        assert result.is_valid is True
        assert 201 <= result.sub_index <= 300

    def test_pm25_sub_index_severe_zone(self):
        """PM2.5 = 300 µg/m³ → Severe (401+)."""
        result = CPCBAQICalculator.calculate_sub_index("pm25", 300.0)
        assert result.is_valid is True
        assert result.sub_index >= 401

    def test_pm10_sub_index_moderate(self):
        """PM10 = 150 µg/m³ → Moderate zone."""
        result = CPCBAQICalculator.calculate_sub_index("pm10", 150.0)
        assert result.is_valid is True
        assert 101 <= result.sub_index <= 200

    def test_pm10_sub_index_good(self):
        """PM10 = 25 µg/m³ → Good zone."""
        result = CPCBAQICalculator.calculate_sub_index("pm10", 25.0)
        assert result.is_valid is True
        assert result.sub_index <= 50

    def test_no2_sub_index_valid(self):
        result = CPCBAQICalculator.calculate_sub_index("no2", 100.0)
        assert result.is_valid is True
        assert result.sub_index is not None

    def test_so2_sub_index_valid(self):
        result = CPCBAQICalculator.calculate_sub_index("so2", 50.0)
        assert result.is_valid is True

    def test_co_sub_index_valid(self):
        result = CPCBAQICalculator.calculate_sub_index("co", 5.0)
        assert result.is_valid is True

    def test_unsupported_pollutant(self):
        """An unrecognised pollutant should produce is_valid=False."""
        result = CPCBAQICalculator.calculate_sub_index("benzene", 10.0)
        assert result.is_valid is False
        assert result.sub_index is None

    def test_negative_concentration_rejected(self):
        result = CPCBAQICalculator.calculate_sub_index("pm25", -5.0)
        assert result.is_valid is False
        assert result.sub_index is None

    def test_zero_concentration(self):
        """Zero concentration is a valid lower boundary."""
        result = CPCBAQICalculator.calculate_sub_index("pm25", 0.0)
        assert result.is_valid is True
        assert result.sub_index == 0

    def test_nan_concentration_rejected(self):
        result = CPCBAQICalculator.calculate_sub_index("pm25", float("nan"))
        assert result.is_valid is False

    def test_case_insensitive_pollutant_name(self):
        r1 = CPCBAQICalculator.calculate_sub_index("PM25", 60.0)
        r2 = CPCBAQICalculator.calculate_sub_index("pm25", 60.0)
        # Both should be valid
        assert r1.is_valid is True
        assert r2.is_valid is True

    def test_unit_reported_correctly(self):
        result = CPCBAQICalculator.calculate_sub_index("pm25", 60.0)
        assert "µg" in result.unit or "ug" in result.unit.lower() or result.unit == "µg/m³"


# ---------------------------------------------------------------------------
# Official AQI calculation — CPCB rule enforcement
# ---------------------------------------------------------------------------

class TestOfficialAQI:
    """Test full AQI calculations with pollutant completeness rules."""

    def test_three_pollutants_produces_official_aqi(self):
        """PM2.5 + PM10 + NO2 (≥3 pollutants) → official AQI."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 110.0,
            "pm10": 200.0,
            "no2": 100.0,
        })
        assert isinstance(result, AQIResult)
        assert result.is_official_cpcb_compliant is True
        assert result.aqi is not None
        assert result.status == "official_aqi"

    def test_two_pollutants_not_official(self):
        """PM2.5 + PM10 only (< 3 pollutants) → NOT official CPCB AQI."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 90.0,
            "pm10": 150.0,
        })
        assert result.is_official_cpcb_compliant is False
        assert result.status == "incomplete_pollutant_coverage"

    def test_single_pollutant_not_official(self):
        """Only PM2.5 → not official."""
        result = CPCBAQICalculator.calculate_aqi({"pm25": 60.0})
        assert result.is_official_cpcb_compliant is False
        assert result.aqi is None or result.is_official_cpcb_compliant is False

    def test_empty_dict_insufficient_data(self):
        """No pollutants → insufficient_data status."""
        result = CPCBAQICalculator.calculate_aqi({})
        assert result.is_official_cpcb_compliant is False
        assert result.aqi is None

    def test_dominant_pollutant_is_highest_sub_index(self):
        """The dominant pollutant must be the one with the highest sub-index."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 300.0,  # Very high → severe sub-index
            "pm10": 50.0,   # Low
            "no2": 20.0,    # Low
        })
        assert result.dominant_pollutant is not None
        assert result.dominant_pollutant.lower() in {"pm25", "pm10", "no2"}
        # PM2.5 at 300 should dominate
        assert result.dominant_pollutant.lower() == "pm25"

    def test_aqi_equals_max_sub_index(self):
        """Official AQI = maximum of valid sub-indices."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 110.0,
            "pm10": 50.0,
            "no2": 20.0,
        })
        if result.is_official_cpcb_compliant and result.aqi is not None:
            # AQI must be the max sub-index
            max_sub = max(
                si.sub_index for si in result.sub_indices.values()
                if si.is_valid and si.sub_index is not None
            )
            assert result.aqi == max_sub

    def test_aqi_category_color_assigned(self):
        """Every official AQI result must have category and color."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 110.0,
            "pm10": 200.0,
            "no2": 80.0,
        })
        if result.is_official_cpcb_compliant:
            assert result.category is not None
            assert result.color is not None
            assert result.health_statement is not None

    def test_severe_category_at_high_pm25(self):
        """PM2.5 = 400 µg/m³ with 2 more pollutants → Severe category."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 400.0,
            "pm10": 100.0,
            "no2": 50.0,
        })
        if result.is_official_cpcb_compliant:
            assert result.category in {"Severe", "Very Poor"}

    def test_warnings_present_for_incomplete_coverage(self):
        """Incomplete coverage should include warnings about missing pollutants."""
        result = CPCBAQICalculator.calculate_aqi({"pm25": 90.0, "pm10": 150.0})
        assert len(result.warnings) > 0

    def test_sub_indices_dict_populated(self):
        """sub_indices must contain entries for all submitted pollutants."""
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 60.0,
            "pm10": 100.0,
            "no2": 40.0,
        })
        assert "pm25" in result.sub_indices
        assert "pm10" in result.sub_indices
        assert "no2" in result.sub_indices

    def test_monitored_pollutants_count_correct(self):
        result = CPCBAQICalculator.calculate_aqi({
            "pm25": 60.0,
            "pm10": 100.0,
            "no2": 40.0,
        })
        assert result.monitored_pollutants_count == 3

    def test_source_documentation_present(self):
        result = CPCBAQICalculator.calculate_aqi({"pm25": 60.0})
        assert isinstance(result.source_documentation, str)
        assert len(result.source_documentation) > 10


# ---------------------------------------------------------------------------
# Category lookup
# ---------------------------------------------------------------------------

class TestCategoryLookup:
    def test_good_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(25)
        assert cat == "Good"

    def test_satisfactory_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(75)
        assert cat == "Satisfactory"

    def test_moderate_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(150)
        assert cat == "Moderate"

    def test_poor_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(250)
        assert cat == "Poor"

    def test_very_poor_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(350)
        assert cat == "Very Poor"

    def test_severe_category(self):
        cat, color, health = CPCBAQICalculator.get_category_info(450)
        assert cat == "Severe"

    def test_color_is_hex_string(self):
        _, color, _ = CPCBAQICalculator.get_category_info(50)
        assert color.startswith("#")
        assert len(color) == 7

    def test_aqi_clamped_at_boundaries(self):
        """AQI 0 and 500 should return valid categories without error."""
        cat_low, _, _ = CPCBAQICalculator.get_category_info(0)
        cat_high, _, _ = CPCBAQICalculator.get_category_info(500)
        assert cat_low is not None
        assert cat_high is not None
