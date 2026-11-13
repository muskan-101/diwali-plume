"""
test_episode_predictor.py — Tests for the Episode Detection and Persistence Prediction Framework.

Covers:
- Active severe episode detection (PM2.5 >= 250 µg/m³ for >= 3 hours)
- Inactive/Normal condition handling
- At-risk elevated status
- Peak PM2.5 calculation and timestamp tracking
- Trend detection (rising, declining, plateaued)
- Expected remaining duration and time-to-improvement calculation
- Meteorological dispersion factor adjustments (wind speed, boundary layer height)
- Projected decay trajectory generation
- Non-empty station history and empty history handling
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.forecasting.episode_predictor import (
    EpisodePredictor,
    EpisodeStatus,
    EpisodePrediction,
    DEFAULT_SEVERE_THRESHOLD,
)


def _make_sample_history(
    pm25_values: list[float],
    start_time: datetime,
    step_hours: int = 1,
) -> pd.DataFrame:
    """Helper to generate mock station history."""
    timestamps = [start_time + timedelta(hours=i * step_hours) for i in range(len(pm25_values))]
    return pd.DataFrame({
        "timestamp": timestamps,
        "pm25": pm25_values,
    })


class TestEpisodeDetection:
    """Test detection of severe pollution episodes."""

    def test_active_severe_episode_detected(self):
        """Sustained >= 250 for 4 hours should be detected as ACTIVE_SEVERE_EPISODE."""
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [150.0, 260.0, 280.0, 310.0, 290.0]
        df = _make_sample_history(values, t0)
        current_time = t0 + timedelta(hours=4)

        status = EpisodePredictor.detect_episode(df, current_time)
        assert isinstance(status, EpisodeStatus)
        assert status.is_in_episode is True
        assert status.current_status == "ACTIVE_SEVERE_EPISODE"
        assert status.current_pm25 == 290.0
        assert status.peak_pm25_observed == 310.0
        assert status.elapsed_duration_hours >= 3.0

    def test_normal_levels_not_in_episode(self):
        """Values under 100 should not trigger an episode."""
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [50.0, 60.0, 75.0, 80.0]
        df = _make_sample_history(values, t0)
        current_time = t0 + timedelta(hours=3)

        status = EpisodePredictor.detect_episode(df, current_time)
        assert status.is_in_episode is False
        assert status.current_status == "NORMAL_LEVELS"

    def test_at_risk_elevated_levels(self):
        """Values between 175 (70% of 250) and 249 are AT_RISK_ELEVATED."""
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [180.0, 190.0, 210.0, 220.0]
        df = _make_sample_history(values, t0)
        current_time = t0 + timedelta(hours=3)

        status = EpisodePredictor.detect_episode(df, current_time)
        assert status.is_in_episode is False
        assert status.current_status == "AT_RISK_ELEVATED"

    def test_transient_spike_not_episode(self):
        """A single 1-hour spike should NOT declare an episode (needs >= 3 consecutive hours)."""
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [120.0, 130.0, 280.0, 140.0]
        df = _make_sample_history(values, t0)
        current_time = t0 + timedelta(hours=3)

        status = EpisodePredictor.detect_episode(df, current_time)
        assert status.is_in_episode is False

    def test_empty_dataframe_safe(self):
        """Empty df returns safe status without crashing."""
        df = pd.DataFrame(columns=["timestamp", "pm25"])
        now = datetime.now(timezone.utc)
        status = EpisodePredictor.detect_episode(df, now)
        assert status.is_in_episode is False
        assert status.current_status == "INSUFFICIENT_DATA"

    def test_rising_trend_detection(self):
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [200.0, 230.0, 270.0, 310.0]
        df = _make_sample_history(values, t0)
        status = EpisodePredictor.detect_episode(df, t0 + timedelta(hours=3))
        assert status.recent_trend == "rising"

    def test_declining_trend_detection(self):
        t0 = datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc)
        values = [320.0, 300.0, 280.0, 250.0]
        df = _make_sample_history(values, t0)
        status = EpisodePredictor.detect_episode(df, t0 + timedelta(hours=3))
        assert status.recent_trend == "declining"


class TestPersistencePrediction:
    """Test episode persistence and duration estimation."""

    def test_prediction_for_active_episode(self):
        status = EpisodeStatus(
            is_in_episode=True,
            episode_threshold_ugm3=250.0,
            current_pm25=300.0,
            current_status="ACTIVE_SEVERE_EPISODE",
            episode_start_time=datetime(2025, 11, 1, 8, 0, tzinfo=timezone.utc),
            elapsed_duration_hours=6.0,
            peak_pm25_observed=320.0,
            peak_timestamp=datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc),
            recent_trend="plateaued",
        )
        current_time = datetime(2025, 11, 1, 14, 0, tzinfo=timezone.utc)
        weather = {
            "wind_speed_10m": 1.5,
            "boundary_layer_height": 200.0,
            "relative_humidity_2m": 75.0,
        }

        pred = EpisodePredictor.predict_persistence(status, current_time, weather)
        assert isinstance(pred, EpisodePrediction)
        assert pred.expected_remaining_hours is not None
        assert pred.expected_remaining_hours > 0
        assert pred.expected_improvement_time is not None
        assert pred.expected_improvement_time > current_time
        assert 0 <= pred.persistence_risk_score <= 100
        assert len(pred.projected_decay_trajectory) > 0

    def test_prediction_for_inactive_episode(self):
        status = EpisodeStatus(
            is_in_episode=False,
            episode_threshold_ugm3=250.0,
            current_pm25=70.0,
            current_status="NORMAL_LEVELS",
            episode_start_time=None,
            elapsed_duration_hours=0.0,
            peak_pm25_observed=70.0,
            peak_timestamp=None,
            recent_trend="stable",
        )
        current_time = datetime(2025, 11, 1, 14, 0, tzinfo=timezone.utc)

        pred = EpisodePredictor.predict_persistence(status, current_time)
        assert pred.expected_remaining_hours == 0.0
        assert pred.persistence_risk_score < 30.0

    def test_high_wind_accelerates_improvement(self):
        """Strong wind (> 5 m/s) and high BLH should result in shorter remaining duration than calm air."""
        status = EpisodeStatus(
            is_in_episode=True,
            episode_threshold_ugm3=250.0,
            current_pm25=300.0,
            current_status="ACTIVE_SEVERE_EPISODE",
            episode_start_time=datetime(2025, 11, 1, 8, 0, tzinfo=timezone.utc),
            elapsed_duration_hours=4.0,
            peak_pm25_observed=310.0,
            peak_timestamp=datetime(2025, 11, 1, 10, 0, tzinfo=timezone.utc),
            recent_trend="plateaued",
        )
        current_time = datetime(2025, 11, 1, 12, 0, tzinfo=timezone.utc)

        calm_weather = {"wind_speed_10m": 0.8, "boundary_layer_height": 150.0}
        windy_weather = {"wind_speed_10m": 6.5, "boundary_layer_height": 800.0}

        pred_calm = EpisodePredictor.predict_persistence(status, current_time, calm_weather)
        pred_windy = EpisodePredictor.predict_persistence(status, current_time, windy_weather)

        assert pred_windy.expected_remaining_hours <= pred_calm.expected_remaining_hours
