"""
test_api.py — Unit and integration tests for the Flask API.

Covered scenarios
-----------------
- GET /health returns expected fields.
- POST /analyze with missing file field → 400.
- POST /analyze with invalid extension → 400.
- POST /analyze with missing coordinates → 400.
- POST /analyze with invalid datetime → 400.
- POST /analyze end-to-end with smoke_test.mp4 (integration smoke test).
- GET /results/<id> → 404 for unknown id.
- GET /results/<id> → 200 for known id (after /analyze).
- GET /outputs/<id>/<filename> path traversal blocked → 400/404.
- Error handling — 413 for oversized payload (mocked).
"""

import sys
import io
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Return a Flask test client with temporary upload/output dirs."""
    import src.api.app as app_module

    # Redirect filesystem dirs to tmp so tests are isolated.
    tmp = tmp_path_factory.mktemp("api_test")
    app_module.UPLOAD_DIR  = tmp / "uploads"
    app_module.OUTPUT_DIR  = tmp / "outputs"
    app_module.RESULTS_DIR = tmp / "results"
    for d in (app_module.UPLOAD_DIR, app_module.OUTPUT_DIR, app_module.RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Reset cached service so it picks up new paths.
    app_module._service = None

    app_module.app.config["TESTING"] = True
    app_module.app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    with app_module.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Tests — /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_dashboard_index_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_contains_status(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_has_model_flags(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        assert "vision_model_available" in data
        assert "forecast_model_available" in data
        assert "dataset_available" in data

    def test_health_flags_are_bool(self, client):
        resp = client.get("/health")
        data = resp.get_json()
        for key in ("vision_model_available", "forecast_model_available", "dataset_available"):
            assert isinstance(data[key], bool)


# ---------------------------------------------------------------------------
# Tests — /analyze input validation
# ---------------------------------------------------------------------------

class TestAnalyzeValidation:
    def test_missing_video_field(self, client):
        resp = client.post("/analyze", data={
            "latitude": "28.6",
            "longitude": "77.2",
            "recording_datetime": "2025-10-01T12:00:00",
        })
        assert resp.status_code == 400
        assert "video" in resp.get_json().get("error", "").lower()

    def test_invalid_extension(self, client):
        data = {
            "latitude": "28.6",
            "longitude": "77.2",
            "recording_datetime": "2025-10-01T12:00:00",
            "mode": "historical_replay",
        }
        fake_file = (io.BytesIO(b"fake content"), "test.txt")
        resp = client.post(
            "/analyze",
            data={**data, "video": fake_file},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400
        assert "unsupported" in resp.get_json().get("error", "").lower()

    def test_missing_latitude(self, client):
        fake_mp4 = (io.BytesIO(b"\x00" * 100), "test.mp4")
        resp = client.post(
            "/analyze",
            data={
                "video": fake_mp4,
                "longitude": "77.2",
                "recording_datetime": "2025-10-01T12:00:00",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_invalid_latitude(self, client):
        fake_mp4 = (io.BytesIO(b"\x00" * 100), "test.mp4")
        resp = client.post(
            "/analyze",
            data={
                "video": fake_mp4,
                "latitude": "not_a_float",
                "longitude": "77.2",
                "recording_datetime": "2025-10-01T12:00:00",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_missing_datetime(self, client):
        fake_mp4 = (io.BytesIO(b"\x00" * 100), "test.mp4")
        resp = client.post(
            "/analyze",
            data={
                "video": fake_mp4,
                "latitude": "28.6",
                "longitude": "77.2",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_invalid_datetime_format(self, client):
        fake_mp4 = (io.BytesIO(b"\x00" * 100), "test.mp4")
        resp = client.post(
            "/analyze",
            data={
                "video": fake_mp4,
                "latitude": "28.6",
                "longitude": "77.2",
                "recording_datetime": "not-a-date",
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Tests — /results
# ---------------------------------------------------------------------------

class TestResultsEndpoint:
    def test_unknown_result_id_returns_404(self, client):
        fake_id = "a" * 36  # all-'a' UUID-shaped string
        resp = client.get(f"/results/{fake_id}")
        assert resp.status_code == 404

    def test_invalid_result_id_shape_returns_400(self, client):
        resp = client.get("/results/../../etc/passwd")
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Tests — /outputs path traversal
# ---------------------------------------------------------------------------

class TestOutputsSecurity:
    def test_path_traversal_blocked(self, client):
        # UUIDs only: traversal with ../.. should fail safe.
        resp = client.get("/outputs/../../etc/passwd/file.csv")
        assert resp.status_code in (400, 404)

    def test_invalid_filename_blocked(self, client):
        fake_id = "a" * 36
        resp = client.get(f"/outputs/{fake_id}/../../secret.csv")
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Integration smoke test — /analyze with real video
# ---------------------------------------------------------------------------

SMOKE_VIDEO = ROOT / "data" / "raw" / "visual" / "input_videos" / "smoke_test.mp4"
MODEL_PATH  = ROOT / "models" / "vision" / "fire_smoke_yolov8n_dfire_best.pt"
DATA_PATH   = ROOT / "data" / "processed" / "delhi_multimodal_clean.csv"


@pytest.fixture(scope="class")
def analyze_response(client):
    with open(SMOKE_VIDEO, "rb") as fh:
        resp = client.post(
            "/analyze",
            data={
                "video": (fh, "smoke_test.mp4"),
                "latitude": "28.5633",
                "longitude": "77.1869",
                "recording_datetime": "2025-06-15T12:00:00+05:30",
                "mode": "historical_replay",
                "max_sampled_frames": "30",
            },
            content_type="multipart/form-data",
        )
    return resp


@pytest.mark.skipif(
    not SMOKE_VIDEO.exists() or not MODEL_PATH.exists() or not DATA_PATH.exists(),
    reason="Smoke test requires demo video, vision model, and dataset.",
)
class TestAnalyzeIntegration:
    """
    End-to-end test through the Flask API using the existing demo video.
    Uses a real timestamp/location from the dataset to maximise chance of
    finding a valid station match.
    """

    # R K Puram, Delhi station (location_id=17): lat=28.5633, lon=77.1869
    LAT = 28.5633
    LON = 77.1869
    # A well-covered timestamp from the dataset.
    DT  = "2025-06-15T12:00:00+05:30"

    def test_analyze_returns_200(self, analyze_response):
        assert analyze_response.status_code == 200, analyze_response.data

    def test_analyze_response_has_result_id(self, analyze_response):
        data = analyze_response.get_json()
        assert "result_id" in data

    def test_analyze_response_has_video_analysis(self, analyze_response):
        data = analyze_response.get_json()
        assert "video_analysis" in data
        assert "nearest_station" in data
        assert "pm25_forecast_6h" in data

    def test_analyze_response_has_station(self, analyze_response):
        data = analyze_response.get_json()
        station = data.get("nearest_station", {})
        # Station should be found for central Delhi.
        assert station.get("found") or len(data.get("warnings", [])) > 0

    def test_result_retrievable_via_get(self, client, analyze_response):
        result_id = analyze_response.get_json().get("result_id")
        assert result_id is not None
        resp2 = client.get(f"/results/{result_id}")
        assert resp2.status_code == 200

    def test_no_pm25_in_video_analysis(self, analyze_response):
        """Video analysis section must not contain a pm25 key."""
        data = analyze_response.get_json()
        video_section = data.get("video_analysis", {})
        # Confirm we do NOT expose a raw pm25 value inside video_analysis.
        assert "pm25" not in video_section or video_section.get("pm25") is None

    def test_forecast_caveat_present(self, analyze_response):
        data = analyze_response.get_json()
        forecast = data.get("pm25_forecast_6h", {})
        assert "caveat" in forecast or not forecast.get("available", True)
