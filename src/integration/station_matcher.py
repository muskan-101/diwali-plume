"""
station_matcher.py — Diwali Plume Stage A

Finds the nearest monitoring station to user-supplied coordinates and
retrieves the latest PM2.5 reading at or before the requested time.

Design constraints:
- Uses Haversine formula for geographic distance.
- Never silently fills missing PM2.5 observations.
- Never selects a station purely on distance if its data is invalid or stale.
- Returns explicit data-quality flags alongside every result.
- Reading staleness is configurable; default maximum is 3 hours.
- Maximum search radius is configurable; default is 50 km.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = Path("data/processed/delhi_multimodal_clean.csv")

# Columns expected in the processed dataset.
COL_STATION_ID = "location_id"
COL_STATION_NAME = "station_name"
COL_LATITUDE = "latitude"
COL_LONGITUDE = "longitude"
COL_TIMESTAMP = "hour"          # raw name in CSV
COL_PM25 = "pm25"
COL_PM25_COUNT = "pm25_sensor_count"

LAGS = [0, 1, 3, 6, 12, 24]

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StationMatch:
    """Result returned by find_nearest_station()."""

    # Station identity
    station_id: int
    station_name: str
    station_latitude: float
    station_longitude: float
    distance_km: float

    # PM2.5 reading
    pm25: Optional[float]
    reading_timestamp: Optional[datetime]
    reading_age_hours: Optional[float]
    pm25_sensor_count: Optional[float]

    # Data quality
    data_quality: str           # "good" | "stale" | "missing" | "no_station"
    warnings: list[str] = field(default_factory=list)
    is_valid: bool = True       # False if no usable station was found


@dataclass
class StationMatcherConfig:
    max_distance_km: float = 50.0
    max_reading_age_hours: float = 3.0
    data_path: Path = DEFAULT_DATA_PATH


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# StationMatcher
# ---------------------------------------------------------------------------

class StationMatcher:
    """Loads station data once and answers nearest-station queries."""

    def __init__(self, config: Optional[StationMatcherConfig] = None) -> None:
        self.config = config or StationMatcherConfig()
        self._df: Optional[pd.DataFrame] = None
        self._station_meta: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        logger.info("Loading station dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip().str.lower()

        # Normalise column names to what the rest of the system expects.
        aliases = {
            "station_id": ["location_id"],
            "timestamp":  ["hour", "timestamp", "timestamp_ist"],
            "latitude":   ["lat"],
            "longitude":  ["lon"],
            "pm25":       ["pm2.5", "value_pm25"],
        }
        for standard, candidates in aliases.items():
            if standard not in df.columns:
                for c in candidates:
                    if c in df.columns:
                        df = df.rename(columns={c: standard})
                        break

        required = ["station_id", "station_name", "timestamp", "latitude", "longitude", "pm25"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Keep only rows with valid timestamps (pm25 may be NaN — that's real data).
        df = df.dropna(subset=["station_id", "timestamp", "latitude", "longitude"])
        df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

        self._df = df

        # Build a lightweight station metadata table (one row per station).
        meta = (
            df.groupby("station_id", as_index=False)
            .agg(
                station_name=("station_name", "first"),
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
            )
        )
        # Drop stations with invalid coordinates.
        meta = meta.dropna(subset=["latitude", "longitude"])
        self._station_meta = meta
        logger.info(
            "Loaded %d rows, %d stations",
            len(df),
            meta["station_id"].nunique(),
        )

    def _ensure_loaded(self) -> None:
        if self._df is None:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_nearest_station(
        self,
        latitude: float,
        longitude: float,
        recording_datetime: datetime,
        *,
        max_distance_km: Optional[float] = None,
        max_reading_age_hours: Optional[float] = None,
    ) -> StationMatch:
        """
        Find the nearest monitoring station and retrieve its latest PM2.5
        reading at or before *recording_datetime*.

        Parameters
        ----------
        latitude, longitude :
            User-supplied location (decimal degrees).
        recording_datetime :
            The video recording time (timezone-aware recommended; naive
            datetimes are assumed UTC).
        max_distance_km :
            Override config value.
        max_reading_age_hours :
            Override config value.

        Returns
        -------
        StationMatch
            Always returned; check .is_valid and .data_quality.
        """
        # --- validate inputs -------------------------------------------
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            return StationMatch(
                station_id=-1, station_name="", station_latitude=float("nan"),
                station_longitude=float("nan"), distance_km=float("nan"),
                pm25=None, reading_timestamp=None, reading_age_hours=None,
                pm25_sensor_count=None, data_quality="no_station", is_valid=False,
                warnings=["Invalid coordinates supplied."],
            )

        # Normalise recording_datetime to UTC-aware.
        if recording_datetime.tzinfo is None:
            logger.warning("recording_datetime is naive; assuming UTC.")
            recording_datetime = recording_datetime.replace(tzinfo=timezone.utc)
        else:
            recording_datetime = recording_datetime.astimezone(timezone.utc)

        max_dist = max_distance_km if max_distance_km is not None else self.config.max_distance_km
        max_age  = max_reading_age_hours if max_reading_age_hours is not None else self.config.max_reading_age_hours

        self._ensure_loaded()

        # --- find nearest station within radius ------------------------
        meta = self._station_meta.copy()
        meta["distance_km"] = meta.apply(
            lambda r: _haversine_km(latitude, longitude, r["latitude"], r["longitude"]),
            axis=1,
        )
        meta = meta.sort_values("distance_km")

        within_radius = meta[meta["distance_km"] <= max_dist]
        if within_radius.empty:
            closest = meta.iloc[0]
            return StationMatch(
                station_id=int(closest["station_id"]),
                station_name=str(closest["station_name"]),
                station_latitude=float(closest["latitude"]),
                station_longitude=float(closest["longitude"]),
                distance_km=float(closest["distance_km"]),
                pm25=None, reading_timestamp=None, reading_age_hours=None,
                pm25_sensor_count=None, data_quality="no_station", is_valid=False,
                warnings=[
                    f"No station within {max_dist} km. "
                    f"Closest is '{closest['station_name']}' at "
                    f"{closest['distance_km']:.1f} km."
                ],
            )

        # Iterate candidates in order of increasing distance, pick the first
        # one that has a valid (non-stale) PM2.5 reading.
        df = self._df

        for _, row in within_radius.iterrows():
            sid = row["station_id"]
            sname = str(row["station_name"])
            slat = float(row["latitude"])
            slon = float(row["longitude"])
            dist = float(row["distance_km"])

            station_df = df[df["station_id"] == sid].copy()
            station_df = station_df[station_df["timestamp"] <= pd.Timestamp(recording_datetime)]

            match = self._extract_reading(
                station_df, sid, sname, slat, slon, dist, recording_datetime, max_age
            )
            if match.data_quality == "good":
                return match
            # else continue to next candidate

        # No candidate had a good reading — return the nearest with its
        # actual quality status so the caller can decide.
        nearest = within_radius.iloc[0]
        sid = nearest["station_id"]
        station_df = df[df["station_id"] == sid].copy()
        station_df = station_df[station_df["timestamp"] <= pd.Timestamp(recording_datetime)]
        return self._extract_reading(
            station_df, int(sid), str(nearest["station_name"]),
            float(nearest["latitude"]), float(nearest["longitude"]),
            float(nearest["distance_km"]), recording_datetime, max_age
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_reading(
        self,
        station_df: pd.DataFrame,
        station_id: int,
        station_name: str,
        lat: float,
        lon: float,
        dist: float,
        ref_time: datetime,
        max_age_hours: float,
    ) -> StationMatch:
        """Extract the latest PM2.5 reading at or before ref_time."""

        warnings: list[str] = []

        if station_df.empty:
            return StationMatch(
                station_id=station_id, station_name=station_name,
                station_latitude=lat, station_longitude=lon, distance_km=dist,
                pm25=None, reading_timestamp=None, reading_age_hours=None,
                pm25_sensor_count=None, data_quality="missing", is_valid=False,
                warnings=[
                    f"Station '{station_name}' has no data at or before the requested time."
                ],
            )

        # Latest row (already filtered to <= ref_time, sorted ascending).
        latest = station_df.sort_values("timestamp").iloc[-1]
        ts = latest["timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        ref_aware = ref_time if ref_time.tzinfo else ref_time.replace(tzinfo=timezone.utc)
        age_hours = (ref_aware - ts).total_seconds() / 3600.0

        pm25_val = latest["pm25"]
        pm25_count = latest.get("pm25_sensor_count", None)

        if pd.isna(pm25_val):
            # Try to find the most recent non-null reading within allowed age.
            non_null = station_df[station_df["pm25"].notna()].sort_values("timestamp")
            if non_null.empty:
                return StationMatch(
                    station_id=station_id, station_name=station_name,
                    station_latitude=lat, station_longitude=lon, distance_km=dist,
                    pm25=None, reading_timestamp=ts.to_pydatetime(),
                    reading_age_hours=round(age_hours, 2),
                    pm25_sensor_count=None, data_quality="missing", is_valid=False,
                    warnings=[
                        f"Station '{station_name}' has no valid PM2.5 readings before the requested time."
                    ],
                )
            latest = non_null.iloc[-1]
            ts = latest["timestamp"]
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            age_hours = (ref_aware - ts).total_seconds() / 3600.0
            pm25_val = latest["pm25"]
            pm25_count = latest.get("pm25_sensor_count", None)
            warnings.append(
                f"Most recent non-null PM2.5 reading is {age_hours:.1f} h old "
                f"(latest row had NaN)."
            )

        # Staleness check.
        if age_hours > max_age_hours:
            warnings.append(
                f"PM2.5 reading is {age_hours:.1f} h old "
                f"(threshold: {max_age_hours} h). Flagged as stale."
            )
            return StationMatch(
                station_id=station_id, station_name=station_name,
                station_latitude=lat, station_longitude=lon, distance_km=dist,
                pm25=float(pm25_val),
                reading_timestamp=ts.to_pydatetime(),
                reading_age_hours=round(age_hours, 2),
                pm25_sensor_count=float(pm25_count) if pm25_count is not None and not pd.isna(pm25_count) else None,
                data_quality="stale", is_valid=True,
                warnings=warnings,
            )

        return StationMatch(
            station_id=station_id, station_name=station_name,
            station_latitude=lat, station_longitude=lon, distance_km=dist,
            pm25=float(pm25_val),
            reading_timestamp=ts.to_pydatetime(),
            reading_age_hours=round(age_hours, 2),
            pm25_sensor_count=float(pm25_count) if pm25_count is not None and not pd.isna(pm25_count) else None,
            data_quality="good",
            is_valid=True,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_station_pm25_history(
        self, station_id: int, up_to: datetime, hours_back: int = 48
    ) -> pd.DataFrame:
        """
        Return a station's PM2.5 time-series for lag construction.

        Only rows *strictly at or before* `up_to` are returned.
        No interpolation is performed.

        Parameters
        ----------
        station_id : int
        up_to      : datetime  (reference / prediction time t)
        hours_back : int       How many hours of history to fetch.
        """
        self._ensure_loaded()
        df = self._df

        if up_to.tzinfo is None:
            up_to = up_to.replace(tzinfo=timezone.utc)
        else:
            up_to = up_to.astimezone(timezone.utc)

        earliest = up_to - timedelta(hours=hours_back)
        mask = (
            (df["station_id"] == station_id)
            & (df["timestamp"] <= pd.Timestamp(up_to))
            & (df["timestamp"] >= pd.Timestamp(earliest))
        )
        return df[mask].sort_values("timestamp").copy()

    def get_weather_at(
        self, station_id: int, timestamp: datetime
    ) -> Optional[pd.Series]:
        """
        Return the weather row nearest to *timestamp* (exact match preferred).
        Returns None if no matching row exists.
        Weather in this dataset is ERA5 reanalysis — not station-specific.
        """
        self._ensure_loaded()

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)

        df = self._df
        mask = df["station_id"] == station_id
        sub = df[mask].copy()

        # Exact match first.
        exact = sub[sub["timestamp"] == pd.Timestamp(timestamp)]
        if not exact.empty:
            return exact.iloc[0]

        # Nearest within ±1 h.
        sub["_tdiff"] = (sub["timestamp"] - pd.Timestamp(timestamp)).abs()
        near = sub[sub["_tdiff"] <= pd.Timedelta(hours=1)]
        if near.empty:
            return None
        return near.sort_values("_tdiff").iloc[0]

    def check_station_lags_complete(
        self,
        station_id: int,
        issue_time: datetime,
        required_lags: tuple[int, ...] = (0, 1, 3, 6, 12, 24),
    ) -> tuple[bool, list[str]]:
        """Check if a specific station has all required historical PM2.5 lags."""
        self._ensure_loaded()
        df = self._df

        if issue_time.tzinfo is None:
            issue_time = issue_time.replace(tzinfo=timezone.utc)
        else:
            issue_time = issue_time.astimezone(timezone.utc)

        issue_ts = pd.Timestamp(issue_time)
        station_df = df[
            (df["station_id"] == station_id)
            & (df["timestamp"] <= issue_ts)
        ]
        if station_df.empty:
            return False, [f"pm25_lag_{lag}" for lag in required_lags]

        missing = []
        for lag in required_lags:
            feat = f"pm25_lag_{lag}"
            target_ts = issue_ts - pd.Timedelta(hours=lag)
            row = station_df[station_df["timestamp"] == target_ts]
            if row.empty or pd.isna(row["pm25"].values[0]):
                win = station_df[
                    (station_df["timestamp"] >= target_ts - pd.Timedelta(minutes=30))
                    & (station_df["timestamp"] <= target_ts + pd.Timedelta(minutes=30))
                ]
                if win.empty or pd.isna(win["pm25"].values[-1]):
                    missing.append(feat)

        return (len(missing) == 0), missing

    def find_forecast_station(
        self,
        latitude: float,
        longitude: float,
        issue_time: datetime,
        max_fallback_km: float = 10.0,
        required_lags: tuple[int, ...] = (0, 1, 3, 6, 12, 24),
    ) -> tuple[Optional[StationMatch], bool, Optional[str], list[str]]:
        """
        Finds an eligible station with complete historical lags for forecasting.
        Maintains clear distinction between nearest measured station and forecast station.
        """
        self._ensure_loaded()
        nearest = self.find_nearest_station(latitude, longitude, issue_time)
        if not nearest.is_valid or nearest.station_id < 0:
            return None, False, "No valid monitoring station found.", []

        # 1. Test nearest station
        nearest_complete, nearest_missing = self.check_station_lags_complete(
            nearest.station_id, issue_time, required_lags
        )
        if nearest_complete:
            return nearest, False, None, []

        # 2. Nearest station lacks complete lags -> Search candidates within max_fallback_km
        meta = self._station_meta.copy()
        meta["distance_km"] = meta.apply(
            lambda r: _haversine_km(latitude, longitude, r["latitude"], r["longitude"]),
            axis=1,
        )
        candidates = meta[meta["distance_km"] <= max_fallback_km].sort_values("distance_km")

        for _, row in candidates.iterrows():
            sid = int(row["station_id"])
            if sid == nearest.station_id:
                continue

            complete, _ = self.check_station_lags_complete(sid, issue_time, required_lags)
            if complete:
                # Build station match for this fallback
                s_df = self._df[self._df["station_id"] == sid]
                match = self._extract_reading(
                    s_df[s_df["timestamp"] <= pd.Timestamp(issue_time)],
                    sid,
                    str(row["station_name"]),
                    float(row["latitude"]),
                    float(row["longitude"]),
                    float(row["distance_km"]),
                    issue_time,
                    max_age_hours=72.0,
                )
                note = (
                    f"Controlled Fallback: Nearest station '{nearest.station_name}' ({nearest.distance_km:.2f} km) "
                    f"lacks complete historical lags ({nearest_missing}). Forecast station '{match.station_name}' "
                    f"({match.distance_km:.2f} km) was selected within {max_fallback_km} km radius. "
                    "Station identities and coordinates are preserved; lag data is NOT mixed."
                )
                return match, True, note, nearest_missing

        # No candidate station within fallback radius has complete lags
        note = (
            f"Nearest station '{nearest.station_name}' ({nearest.distance_km:.2f} km) lacks lags {nearest_missing}. "
            f"No other station within {max_fallback_km} km radius possessed complete historical lags for {issue_time}."
        )
        return None, False, note, nearest_missing

    @property
    def dataset(self) -> pd.DataFrame:
        """Access the full loaded dataset (read-only use intended)."""
        self._ensure_loaded()
        return self._df
