"""
cpcb_aqi.py — Central Pollution Control Board (CPCB) National Air Quality Index (NAQI) Engine

Source & Reference:
  Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change,
  Government of India. National Air Quality Index (NAQI) Final Report (2014) & Guidelines (2015).

Pollutants Supported (8 criteria pollutants):
  - PM10 (24-hr avg, µg/m³)
  - PM2.5 (24-hr avg, µg/m³)
  - NO2 (24-hr avg, µg/m³)
  - O3 (8-hr avg, µg/m³)
  - CO (8-hr avg, mg/m³)
  - SO2 (24-hr avg, µg/m³)
  - NH3 (24-hr avg, µg/m³)
  - Pb (24-hr avg, µg/m³)

Mandatory CPCB Rules Enforced:
  1. Individual sub-index formula:
     I_p = ((I_HI - I_LO) / (B_HI - B_LO)) * (C_p - B_LO) + I_LO
  2. Overall AQI requirement:
     According to CPCB NAQI guidelines, overall AQI can only be calculated if:
     - At least THREE (3) pollutants are monitored and valid.
     - At least ONE of the pollutants must be PM10 or PM2.5.
  3. If fewer than 3 pollutants are present (e.g. only PM2.5 and PM10), overall AQI
     is NOT declared as official CPCB AQI. Valid sub-indices are reported alongside
     an explicit `incomplete_pollutant_coverage` status and missing pollutant diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


# ---------------------------------------------------------------------------
# CPCB Breakpoint Tables (CPCB NAQI 2014/2015)
# ---------------------------------------------------------------------------

AQI_CATEGORIES = [
    (0, 50, "Good", "#00b050", "Minimal impact"),
    (51, 100, "Satisfactory", "#92d050", "Minor breathing discomfort to sensitive people"),
    (101, 200, "Moderate", "#ffff00", "Breathing discomfort to people with lung, asthma and heart disease"),
    (201, 300, "Poor", "#ff9900", "Breathing discomfort to most people on prolonged exposure"),
    (301, 400, "Very Poor", "#ff0000", "Respiratory illness on prolonged exposure"),
    (401, 500, "Severe", "#c00000", "Affects healthy people and seriously impacts those with existing diseases"),
]

# (B_LO, B_HI, I_LO, I_HI) for each category interval
BREAKPOINTS: Dict[str, List[tuple[float, float, int, int]]] = {
    "pm10": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 250.0, 101, 200),
        (251.0, 350.0, 201, 300),
        (351.0, 430.0, 301, 400),
        (431.0, 600.0, 401, 500),
    ],
    "pm25": [
        (0.0, 30.0, 0, 50),
        (31.0, 60.0, 51, 100),
        (61.0, 90.0, 101, 200),
        (91.0, 120.0, 201, 300),
        (121.0, 250.0, 301, 400),
        (251.0, 500.0, 401, 500),
    ],
    "no2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 180.0, 101, 200),
        (181.0, 280.0, 201, 300),
        (281.0, 400.0, 301, 400),
        (401.0, 600.0, 401, 500),
    ],
    "o3": [
        (0.0, 50.0, 0, 50),
        (51.0, 100.0, 51, 100),
        (101.0, 168.0, 101, 200),
        (169.0, 208.0, 201, 300),
        (209.0, 748.0, 301, 400),
        (749.0, 1000.0, 401, 500),
    ],
    "co": [
        (0.0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "so2": [
        (0.0, 40.0, 0, 50),
        (41.0, 80.0, 51, 100),
        (81.0, 380.0, 101, 200),
        (381.0, 800.0, 201, 300),
        (801.0, 1600.0, 301, 400),
        (1601.0, 2000.0, 401, 500),
    ],
    "nh3": [
        (0.0, 200.0, 0, 50),
        (201.0, 400.0, 51, 100),
        (401.0, 800.0, 101, 200),
        (801.0, 1200.0, 201, 300),
        (1201.0, 1800.0, 301, 400),
        (1801.0, 2500.0, 401, 500),
    ],
    "pb": [
        (0.0, 0.5, 0, 50),
        (0.6, 1.0, 51, 100),
        (1.1, 2.0, 101, 200),
        (2.1, 3.0, 201, 300),
        (3.1, 3.5, 301, 400),
        (3.6, 5.0, 401, 500),
    ],
}

POLLUTANT_UNITS = {
    "pm10": "µg/m³",
    "pm25": "µg/m³",
    "no2": "µg/m³",
    "o3": "µg/m³",
    "co": "mg/m³",
    "so2": "µg/m³",
    "nh3": "µg/m³",
    "pb": "µg/m³",
}


@dataclass
class SubIndexResult:
    pollutant: str
    concentration: float
    unit: str
    sub_index: Optional[int]
    category: Optional[str]
    is_valid: bool
    warning: Optional[str] = None


@dataclass
class AQIResult:
    aqi: Optional[int]
    category: Optional[str]
    color: Optional[str]
    health_statement: Optional[str]
    dominant_pollutant: Optional[str]
    sub_indices: Dict[str, SubIndexResult]
    monitored_pollutants_count: int
    is_official_cpcb_compliant: bool
    status: str  # "official_aqi" | "incomplete_pollutant_coverage" | "insufficient_data"
    warnings: List[str] = field(default_factory=list)
    source_documentation: str = (
        "CPCB National Air Quality Index (NAQI) Guidelines (2014/2015). "
        "Requires >= 3 pollutants with at least one PM parameter for official AQI."
    )


class CPCBAQICalculator:
    """Official CPCB NAQI calculation implementation."""

    @staticmethod
    def get_category_info(aqi_val: int) -> tuple[str, str, str]:
        """Returns (Category Name, Color Hex, Health Statement) for a given AQI."""
        val = max(0, min(500, aqi_val))
        for lo, hi, cat, col, statement in AQI_CATEGORIES:
            if lo <= val <= hi:
                return cat, col, statement
        return "Severe", "#c00000", "Affects healthy people and seriously impacts those with existing diseases"

    @classmethod
    def calculate_sub_index(cls, pollutant: str, concentration: float) -> SubIndexResult:
        """
        Calculate individual sub-index using CPCB linear interpolation formula.
        """
        p = pollutant.lower().replace(".", "").replace("-", "").strip()
        unit = POLLUTANT_UNITS.get(p, "µg/m³")

        if p not in BREAKPOINTS:
            return SubIndexResult(
                pollutant=pollutant,
                concentration=concentration,
                unit=unit,
                sub_index=None,
                category=None,
                is_valid=False,
                warning=f"Unsupported pollutant: {pollutant}",
            )

        if concentration is None or math.isnan(concentration) or concentration < 0:
            return SubIndexResult(
                pollutant=pollutant,
                concentration=concentration,
                unit=unit,
                sub_index=None,
                category=None,
                is_valid=False,
                warning=f"Invalid or negative concentration: {concentration}",
            )

        c = float(concentration)
        table = BREAKPOINTS[p]

        # Check if beyond highest bracket -> cap at 500 with warning
        max_b_hi = table[-1][1]
        if c > max_b_hi:
            cat, _, _ = cls.get_category_info(500)
            return SubIndexResult(
                pollutant=pollutant,
                concentration=c,
                unit=unit,
                sub_index=500,
                category=cat,
                is_valid=True,
                warning=f"Concentration {c} {unit} exceeds highest CPCB bracket ({max_b_hi}); capped at 500",
            )

        # Standard linear interpolation
        for b_lo, b_hi, i_lo, i_hi in table:
            if b_lo <= c <= b_hi:
                sub_idx = round(((i_hi - i_lo) / (b_hi - b_lo)) * (c - b_lo) + i_lo)
                sub_idx = max(0, min(500, sub_idx))
                cat, _, _ = cls.get_category_info(sub_idx)
                return SubIndexResult(
                    pollutant=pollutant,
                    concentration=c,
                    unit=unit,
                    sub_index=sub_idx,
                    category=cat,
                    is_valid=True,
                )

        # In-between edge cases (e.g. 50.5 between 50 and 51):
        # assign to lower/adjacent bracket
        for idx in range(len(table) - 1):
            if table[idx][1] < c < table[idx + 1][0]:
                sub_idx = table[idx][3]  # upper subindex of lower bracket
                cat, _, _ = cls.get_category_info(sub_idx)
                return SubIndexResult(
                    pollutant=pollutant,
                    concentration=c,
                    unit=unit,
                    sub_index=sub_idx,
                    category=cat,
                    is_valid=True,
                )

        return SubIndexResult(
            pollutant=pollutant,
            concentration=c,
            unit=unit,
            sub_index=None,
            category=None,
            is_valid=False,
            warning="Could not map concentration to CPCB breakpoints",
        )

    @classmethod
    def calculate_aqi(cls, pollutants_dict: Dict[str, float]) -> AQIResult:
        """
        Calculate overall AQI according to official CPCB rules.

        Input:
          dict of {pollutant_name: concentration}
          e.g. {"pm25": 145.0, "pm10": 260.0, "no2": 65.0}

        Returns:
          AQIResult with official status, sub-indices, and explicit warnings.
        """
        sub_indices: Dict[str, SubIndexResult] = {}
        valid_subs: Dict[str, int] = {}
        warnings: List[str] = []

        has_pm = False
        for p_raw, conc in pollutants_dict.items():
            res = cls.calculate_sub_index(p_raw, conc)
            norm_key = p_raw.lower().replace(".", "").replace("-", "").strip()
            sub_indices[norm_key] = res
            if res.is_valid and res.sub_index is not None:
                valid_subs[norm_key] = res.sub_index
                if norm_key in ("pm25", "pm10"):
                    has_pm = True
            elif res.warning:
                warnings.append(res.warning)

        monitored_count = len(valid_subs)

        if not valid_subs:
            return AQIResult(
                aqi=None,
                category=None,
                color=None,
                health_statement=None,
                dominant_pollutant=None,
                sub_indices=sub_indices,
                monitored_pollutants_count=0,
                is_official_cpcb_compliant=False,
                status="insufficient_data",
                warnings=["No valid pollutant sub-indices could be calculated."],
            )

        # Dominant pollutant is the one with highest sub-index
        dominant_pollutant = max(valid_subs, key=lambda k: valid_subs[k])
        max_sub_idx = valid_subs[dominant_pollutant]

        # Enforce CPCB official criteria:
        # Minimum 3 pollutants with at least one PM parameter (PM2.5 or PM10)
        is_official = (monitored_count >= 3) and has_pm

        cat, col, stmt = cls.get_category_info(max_sub_idx)

        if is_official:
            status = "official_aqi"
        else:
            status = "incomplete_pollutant_coverage"
            missing_reasons = []
            if monitored_count < 3:
                missing_reasons.append(
                    f"Only {monitored_count} criteria pollutant(s) available ({', '.join(valid_subs.keys())}); CPCB NAQI requires at least 3."
                )
            if not has_pm:
                missing_reasons.append("Neither PM2.5 nor PM10 is monitored.")
            warnings.append(
                f"Incomplete CPCB pollutant coverage: {' '.join(missing_reasons)} "
                "Overall score reflects prominent sub-index, not a complete official AQI."
            )

        return AQIResult(
            aqi=max_sub_idx,
            category=cat,
            color=col,
            health_statement=stmt,
            dominant_pollutant=dominant_pollutant,
            sub_indices=sub_indices,
            monitored_pollutants_count=monitored_count,
            is_official_cpcb_compliant=is_official,
            status=status,
            warnings=warnings,
        )
