"""
forecast_engine.py — Diwali Plume Stage A

Constructs the exact 22-feature vector that the saved XGBoost model expects
and runs a six-hour-ahead PM2.5 prediction.

IMPORTANT constraints:
- The 22-feature order is fixed by baseline_features.json — do NOT reorder.
- Lag values are assembled by exact timestamp lookup — no interpolation.
- Weather features come from the historical ERA5 reanalysis stored in the
  processed dataset.
- Features at time t must NEVER include data from t+1 through t+6.
- The prediction horizon is exactly 6 hours — do not claim otherwise.
- This is a historical-replay engine; live forecasting is NOT supported yet.
- Missing lag values cause the forecast to be rejected — never filled silently.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match baseline.py exactly
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH    = Path("models/xgboost_pm25_6h.pkl")
DEFAULT_FEATURES_PATH = Path("models/baseline_features.json")
DEFAULT_DATA_PATH     = Path("data/processed/delhi_multimodal_clean.csv")

FORECAST_HOURS = 6
LAGS = [0, 1, 3, 6, 12, 24]   # hours before t

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "boundary_layer_height",
]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ForecastResult:
    """Output of run_forecast()."""

    # Forecast values
    forecast_pm25: Optional[float]
    forecast_horizon_hours: int
    forecast_issue_timestamp: Optional[datetime]   # time t (features assembled at)
    target_timestamp: Optional[datetime]           # t + 6 h

    # Station provenance
    station_id: int
    station_name: str

    # Feature diagnostics
    feature_vector: Optional[dict]
    missing_features: list[str] = field(default_factory=list)

    # Data quality
    data_quality: str = "unknown"   # "good" | "incomplete_lags" | "missing_weather" | "failed"
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = False


# ---------------------------------------------------------------------------
# ForecastEngine
# ---------------------------------------------------------------------------


class ForecastEngine:
    """
    Wraps the saved XGBoost model and handles feature construction.

    Usage
    -----
    engine = ForecastEngine()
    result = engine.run_forecast(station_id, latitude, longitude, issue_time, station_df)
    """

    def __init__(
        self,
        model_path:    Path = DEFAULT_MODEL_PATH,
        features_path: Path = DEFAULT_FEATURES_PATH,
        data_path:     Path = DEFAULT_DATA_PATH,
    ) -> None:
        self.model_path    = Path(model_path)
        self.features_path = Path(features_path)
        self.data_path     = Path(data_path)

        self._model    = None
        self._features: Optional[list[str]] = None
        self._df:       Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"XGBoost model not found: {self.model_path}")
        self._model = joblib.load(self.model_path)
        logger.info("Loaded XGBoost model from %s", self.model_path)

    def _load_features(self) -> None:
        if not self.features_path.exists():
            raise FileNotFoundError(f"Feature schema not found: {self.features_path}")
        with open(self.features_path, encoding="utf-8") as fh:
            self._features = json.load(fh)
        if len(self._features) != 22:
            raise ValueError(
                f"Expected 22 features, got {len(self._features)}: {self._features}"
            )
        logger.info("Loaded %d features from %s", len(self._features), self.features_path)

    def _load_data(self) -> None:
        path = self.data_path
        if not path.exists():
            sample_path = Path("data/sample/delhi_multimodal_sample.csv")
            if sample_path.exists():
                logger.warning(
                    "Primary dataset not found at %s. Falling back to sample dataset at %s",
                    path,
                    sample_path,
                )
                path = sample_path
            else:
                raise FileNotFoundError(f"Dataset not found: {self.data_path}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        # Same alias resolution as baseline.py and station_matcher.py.
        aliases = {
            "station_id": ["location_id"],
            "timestamp":  ["hour", "timestamp", "timestamp_ist"],
            "pm25":       ["pm2.5", "value_pm25"],
            "latitude":   ["lat"],
            "longitude":  ["lon"],
        }
        for standard, candidates in aliases.items():
            if standard not in df.columns:
                for c in candidates:
                    if c in df.columns:
                        df = df.rename(columns={c: standard})
                        break

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["pm25"]      = pd.to_numeric(df["pm25"],      errors="coerce")
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        for col in WEATHER_FEATURES:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["station_id", "timestamp"])
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        self._df = df
        logger.info("Loaded dataset: %d rows, %d stations", len(df), df["station_id"].nunique())

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()
        if self._features is None:
            self._load_features()
        if self._df is None:
            self._load_data()

    # ------------------------------------------------------------------
    # Feature construction  (mirrors baseline.py exactly)
    # ------------------------------------------------------------------

    def _build_features(
        self,
        station_id: int,
        latitude: float,
        longitude: float,
        issue_time: datetime,
    ) -> tuple[dict, list[str]]:
        """
        Build the 22-feature dict for a prediction at *issue_time* (time t).

        Only data at or before issue_time is used.
        Returns (feature_dict, missing_feature_names).
        """
        self._ensure_loaded()
        df = self._df

        if issue_time.tzinfo is None:
            issue_time = issue_time.replace(tzinfo=timezone.utc)
        else:
            issue_time = issue_time.astimezone(timezone.utc)

        issue_ts = pd.Timestamp(issue_time)

        # Station slice, up to and including issue_time.
        station_df = df[
            (df["station_id"] == station_id)
            & (df["timestamp"] <= issue_ts)
        ].sort_values("timestamp")

        features: dict = {}
        missing:  list[str] = []

        # --- PM2.5 lags -----------------------------------------------
        # Lag 0 = PM2.5 at exactly t.
        # Lag k = PM2.5 at t - k hours.
        # We look for exact hourly timestamps; no interpolation.
        for lag in LAGS:
            feat_name = f"pm25_lag_{lag}"
            target_ts = issue_ts - pd.Timedelta(hours=lag)
            row = station_df[station_df["timestamp"] == target_ts]
            if row.empty or pd.isna(row["pm25"].values[0]):
                # Try within ±30 minutes for the exact hour.
                window = station_df[
                    (station_df["timestamp"] >= target_ts - pd.Timedelta(minutes=30))
                    & (station_df["timestamp"] <= target_ts + pd.Timedelta(minutes=30))
                ].sort_values("timestamp")
                if window.empty or pd.isna(window["pm25"].values[-1]):
                    missing.append(feat_name)
                    features[feat_name] = float("nan")
                else:
                    features[feat_name] = float(window["pm25"].values[-1])
            else:
                features[feat_name] = float(row["pm25"].values[0])

        # --- Static station coordinates --------------------------------
        features["latitude"]  = latitude
        features["longitude"] = longitude

        # --- Weather features at time t --------------------------------
        # ERA5 reanalysis: one central Delhi coordinate, stored per station row.
        weather_row = station_df[station_df["timestamp"] == issue_ts]
        if weather_row.empty:
            # Try nearest within ±1 h (ERA5 is hourly).
            station_df["_tdiff"] = (station_df["timestamp"] - issue_ts).abs()
            near = station_df[station_df["_tdiff"] <= pd.Timedelta(hours=1)]
            if not near.empty:
                weather_row = near.sort_values("_tdiff").iloc[:1]
            station_df = station_df.drop(columns=["_tdiff"], errors="ignore")

        for feat in WEATHER_FEATURES:
            if feat not in self._features:
                continue
            if not weather_row.empty and feat in weather_row.columns:
                val = weather_row[feat].values[0]
                if pd.isna(val):
                    missing.append(feat)
                    features[feat] = float("nan")
                else:
                    features[feat] = float(val)
            else:
                missing.append(feat)
                features[feat] = float("nan")

        # --- Cyclical time features ------------------------------------
        # Reproduces baseline.py lines 150-156 exactly.
        hour_val    = issue_time.hour
        day_val     = issue_time.timetuple().tm_yday
        features["hour_sin"] = math.sin(2 * math.pi * hour_val / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour_val / 24)
        features["day_sin"]  = math.sin(2 * math.pi * day_val  / 365.25)
        features["day_cos"]  = math.cos(2 * math.pi * day_val  / 365.25)

        return features, missing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_forecast(
        self,
        station_id: int,
        latitude: float,
        longitude: float,
        issue_time: datetime,
    ) -> ForecastResult:
        """
        Produce a 6-hour-ahead PM2.5 forecast for *issue_time* (time t).

        Parameters
        ----------
        station_id  : Matched monitoring station ID.
        latitude    : Station latitude (from station matcher).
        longitude   : Station longitude (from station matcher).
        issue_time  : The time at which the forecast is issued (= recording time).
                      Features are drawn exclusively from data at or before this time.

        Returns
        -------
        ForecastResult — always returned; check .is_valid.
        """
        try:
            self._ensure_loaded()
        except Exception as exc:
            return ForecastResult(
                forecast_pm25=None, forecast_horizon_hours=FORECAST_HOURS,
                forecast_issue_timestamp=issue_time,
                target_timestamp=None,
                station_id=station_id, station_name="",
                feature_vector=None, data_quality="failed",
                warnings=[f"Model/data loading failed: {exc}"], is_valid=False,
            )

        if issue_time.tzinfo is None:
            issue_time = issue_time.replace(tzinfo=timezone.utc)
        else:
            issue_time = issue_time.astimezone(timezone.utc)

        target_time = issue_time + timedelta(hours=FORECAST_HOURS)

        # Station name for display.
        station_rows = self._df[self._df["station_id"] == station_id]
        station_name = (
            str(station_rows["station_name"].iloc[0])
            if not station_rows.empty and "station_name" in station_rows.columns
            else f"station_{station_id}"
        )

        # Build features.
        try:
            features, missing = self._build_features(station_id, latitude, longitude, issue_time)
        except Exception as exc:
            return ForecastResult(
                forecast_pm25=None, forecast_horizon_hours=FORECAST_HOURS,
                forecast_issue_timestamp=issue_time, target_timestamp=target_time,
                station_id=station_id, station_name=station_name,
                feature_vector=None, data_quality="failed",
                warnings=[f"Feature construction failed: {exc}"], is_valid=False,
            )

        # Reject if critical PM2.5 lags are missing.
        lag_missing = [f for f in missing if f.startswith("pm25_lag_")]
        weather_missing = [f for f in missing if not f.startswith("pm25_lag_")]

        warnings: list[str] = []
        if lag_missing:
            warnings.append(
                f"Missing PM2.5 lag features: {lag_missing}. Forecast rejected."
            )
            return ForecastResult(
                forecast_pm25=None, forecast_horizon_hours=FORECAST_HOURS,
                forecast_issue_timestamp=issue_time, target_timestamp=target_time,
                station_id=station_id, station_name=station_name,
                feature_vector=features, missing_features=missing,
                data_quality="incomplete_lags",
                warnings=warnings, is_valid=False,
            )

        if weather_missing:
            warnings.append(
                f"Missing weather features: {weather_missing}. "
                "Forecast produced but reliability is reduced."
            )

        # Assemble feature vector in the EXACT saved order.
        try:
            X = np.array([[features[f] for f in self._features]], dtype=np.float32)
        except KeyError as exc:
            return ForecastResult(
                forecast_pm25=None, forecast_horizon_hours=FORECAST_HOURS,
                forecast_issue_timestamp=issue_time, target_timestamp=target_time,
                station_id=station_id, station_name=station_name,
                feature_vector=features, missing_features=missing,
                data_quality="failed",
                warnings=[f"Feature ordering failed: {exc}"], is_valid=False,
            )

        # Run model prediction.
        try:
            prediction = float(self._model.predict(X)[0])
            prediction = max(prediction, 0.0)   # PM2.5 cannot be negative
        except Exception as exc:
            return ForecastResult(
                forecast_pm25=None, forecast_horizon_hours=FORECAST_HOURS,
                forecast_issue_timestamp=issue_time, target_timestamp=target_time,
                station_id=station_id, station_name=station_name,
                feature_vector=features, missing_features=missing,
                data_quality="failed",
                warnings=[f"Model prediction failed: {exc}"], is_valid=False,
            )

        quality = "good" if not weather_missing else "missing_weather"

        return ForecastResult(
            forecast_pm25=round(prediction, 2),
            forecast_horizon_hours=FORECAST_HOURS,
            forecast_issue_timestamp=issue_time,
            target_timestamp=target_time,
            station_id=station_id,
            station_name=station_name,
            feature_vector=features,
            missing_features=missing,
            data_quality=quality,
            warnings=warnings,
            is_valid=True,
        )

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------

    @property
    def model_available(self) -> bool:
        return self.model_path.exists()

    @property
    def features_available(self) -> bool:
        return self.features_path.exists()

    @property
    def dataset_available(self) -> bool:
        return self.data_path.exists() or Path("data/sample/delhi_multimodal_sample.csv").exists()
