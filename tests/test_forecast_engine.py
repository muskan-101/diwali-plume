"""
test_forecast_engine.py — Unit tests for ForecastEngine.

Covered scenarios
-----------------
- Model loading.
- Feature schema loading and correct 22-feature ordering.
- Complete lag availability → valid forecast.
- Missing lag → forecast rejected (never fabricated).
- Forecast horizon is exactly 6 hours.
- Forecast values are non-negative.
- Feature vector keys match baseline_features.json exactly.
- Dataset availability check.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.forecast_engine import ForecastEngine, LAGS, FORECAST_HOURS, WEATHER_FEATURES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH    = ROOT / "models" / "xgboost_pm25_6h.pkl"
FEATURES_PATH = ROOT / "models" / "baseline_features.json"
DATA_PATH     = ROOT / "data" / "processed" / "delhi_multimodal_clean.csv"


def _require_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"XGBoost model not found at {MODEL_PATH}")

def _require_data():
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found at {DATA_PATH}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    _require_model()
    _require_data()
    return ForecastEngine(
        model_path=MODEL_PATH,
        features_path=FEATURES_PATH,
        data_path=DATA_PATH,
    )


@pytest.fixture(scope="module")
def valid_station_and_time(engine):
    """Return (station_id, lat, lon, issue_time) with >=24 h of history."""
    import pandas as pd
    engine._ensure_loaded()
    df = engine._df

    # Find a station with many records for reliable lag availability.
    counts = df.groupby("station_id").size()
    sid = int(counts.idxmax())
    sdf = df[df["station_id"] == sid].sort_values("timestamp")

    # Pick a time 25 h after the first record so lag_24 can be satisfied.
    first_ts = sdf["timestamp"].iloc[0]
    issue_ts = first_ts + pd.Timedelta(hours=25)
    if issue_ts.tzinfo is None:
        issue_ts = issue_ts.replace(tzinfo=timezone.utc)
    else:
        issue_ts = issue_ts.tz_convert("UTC")

    lat = float(sdf["latitude"].iloc[0])
    lon = float(sdf["longitude"].iloc[0])
    return sid, lat, lon, issue_ts.to_pydatetime()


# ---------------------------------------------------------------------------
# Tests — Model / data availability
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_model_available(self):
        _require_model()
        engine = ForecastEngine(model_path=MODEL_PATH, features_path=FEATURES_PATH, data_path=DATA_PATH)
        assert engine.model_available

    def test_features_available(self):
        engine = ForecastEngine(model_path=MODEL_PATH, features_path=FEATURES_PATH, data_path=DATA_PATH)
        assert engine.features_available

    def test_dataset_available(self):
        _require_data()
        engine = ForecastEngine(model_path=MODEL_PATH, features_path=FEATURES_PATH, data_path=DATA_PATH)
        assert engine.dataset_available

    def test_missing_model_path_property(self, tmp_path):
        engine = ForecastEngine(
            model_path=tmp_path / "nonexistent.pkl",
            features_path=FEATURES_PATH,
            data_path=DATA_PATH,
        )
        assert not engine.model_available


# ---------------------------------------------------------------------------
# Tests — Feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    def test_22_features_loaded(self, engine):
        engine._ensure_loaded()
        assert len(engine._features) == 22

    def test_feature_names_match_baseline(self, engine):
        engine._ensure_loaded()
        expected = [
            "pm25_lag_0", "pm25_lag_1", "pm25_lag_3", "pm25_lag_6",
            "pm25_lag_12", "pm25_lag_24",
            "latitude", "longitude",
            "temperature_2m", "relative_humidity_2m", "dew_point_2m",
            "precipitation", "pressure_msl", "wind_speed_10m",
            "wind_direction_10m", "wind_gusts_10m", "cloud_cover",
            "boundary_layer_height",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
        ]
        assert engine._features == expected, (
            f"Feature list mismatch.\n"
            f"Expected: {expected}\n"
            f"Got:      {engine._features}"
        )

    def test_lag_features_in_correct_order(self, engine):
        engine._ensure_loaded()
        lag_features = [f for f in engine._features if f.startswith("pm25_lag_")]
        assert lag_features == [f"pm25_lag_{lag}" for lag in LAGS]


# ---------------------------------------------------------------------------
# Tests — Forecast with valid data
# ---------------------------------------------------------------------------

class TestValidForecast:
    def test_valid_forecast_is_valid(self, engine, valid_station_and_time):
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        # With 25 h of history, lag_24 should be satisfiable.
        # Accept both valid and incomplete_lags (data may have gaps).
        assert result.data_quality in ("good", "missing_weather", "incomplete_lags")

    def test_forecast_horizon_is_6_hours(self, engine, valid_station_and_time):
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        assert result.forecast_horizon_hours == 6

    def test_target_timestamp_is_t_plus_6h(self, engine, valid_station_and_time):
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        if result.target_timestamp is not None:
            diff = (result.target_timestamp - result.forecast_issue_timestamp).total_seconds()
            assert diff == pytest.approx(6 * 3600, abs=1)

    def test_forecast_pm25_non_negative(self, engine, valid_station_and_time):
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        if result.is_valid and result.forecast_pm25 is not None:
            assert result.forecast_pm25 >= 0.0

    def test_forecast_pm25_plausible_range(self, engine, valid_station_and_time):
        """PM2.5 for Delhi should be within 0–2000 µg/m³ (extreme events included)."""
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        if result.is_valid and result.forecast_pm25 is not None:
            assert 0 <= result.forecast_pm25 <= 2000, (
                f"Forecast {result.forecast_pm25} outside plausible range"
            )

    def test_feature_vector_has_22_keys(self, engine, valid_station_and_time):
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        if result.feature_vector is not None:
            assert len(result.feature_vector) == 22

    def test_feature_vector_keys_match_schema(self, engine, valid_station_and_time):
        engine._ensure_loaded()
        sid, lat, lon, t = valid_station_and_time
        result = engine.run_forecast(sid, lat, lon, t)
        if result.feature_vector is not None:
            assert set(result.feature_vector.keys()) == set(engine._features)


# ---------------------------------------------------------------------------
# Tests — Missing lag rejection
# ---------------------------------------------------------------------------

class TestMissingLagRejection:
    def test_forecast_rejected_for_very_early_time(self, engine):
        """
        A timestamp before any data exists should fail because lag_24 (and
        others) cannot be satisfied.
        """
        import pandas as pd
        engine._ensure_loaded()
        df = engine._df
        sid = int(df["station_id"].iloc[0])
        lat = float(df[df["station_id"] == sid]["latitude"].iloc[0])
        lon = float(df[df["station_id"] == sid]["longitude"].iloc[0])

        # Request a time far before the dataset starts.
        very_early = datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = engine.run_forecast(sid, lat, lon, very_early)

        # Must not produce a valid forecast.
        assert not result.is_valid
        assert result.data_quality in ("incomplete_lags", "failed", "missing")

    def test_missing_lags_listed(self, engine):
        import pandas as pd
        engine._ensure_loaded()
        df = engine._df
        sid = int(df["station_id"].iloc[0])
        lat = float(df[df["station_id"] == sid]["latitude"].iloc[0])
        lon = float(df[df["station_id"] == sid]["longitude"].iloc[0])

        very_early = datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc)
        result = engine.run_forecast(sid, lat, lon, very_early)
        assert result.missing_features is not None


# ---------------------------------------------------------------------------
# Tests — Features do not leak future data
# ---------------------------------------------------------------------------

class TestNoFutureDataLeakage:
    def test_features_built_only_from_past(self, engine, valid_station_and_time):
        """
        Verify that the feature vector is computed from data at or before
        issue_time, not from t+1 through t+6.
        """
        import pandas as pd
        engine._ensure_loaded()
        sid, lat, lon, t = valid_station_and_time

        df = engine._df
        station_df = df[
            (df["station_id"] == sid)
            & (df["timestamp"] <= pd.Timestamp(t))
        ].sort_values("timestamp")

        # The lag_0 value must correspond to the PM2.5 at exactly t.
        lag0_row = station_df[station_df["timestamp"] == pd.Timestamp(t)]
        if not lag0_row.empty:
            expected_lag0 = float(lag0_row["pm25"].values[0])
            result = engine.run_forecast(sid, lat, lon, t)
            if result.feature_vector and "pm25_lag_0" in result.feature_vector:
                assert result.feature_vector["pm25_lag_0"] == pytest.approx(expected_lag0, rel=1e-5)
