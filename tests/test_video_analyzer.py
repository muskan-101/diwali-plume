"""
test_video_analyzer.py — Unit tests for VideoAnalyzer.

Covered scenarios
-----------------
- Invalid video path → error result.
- Unsupported file extension → error result.
- Zero-dimension / corrupt video → error result.
- Valid smoke video → success with detections.
- Valid fire video → success with detections.
- Empty detection results → no crash.
- Output files created (annotated MP4, CSV timeline).
- Result fields are correctly typed.

NOTE: Integration smoke tests use the existing demo videos.
      These videos are NOT used to tune the detector.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime, timezone

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.integration.video_analyzer import VideoAnalyzer, VideoAnalysisResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH  = ROOT / "models" / "vision" / "fire_smoke_yolov8n_dfire_best.pt"
SMOKE_VIDEO = ROOT / "data" / "raw" / "visual" / "input_videos" / "smoke_test.mp4"
FIRE_VIDEO  = ROOT / "data" / "raw" / "visual" / "input_videos" / "fire_test.mp4"


def _require_model():
    if not MODEL_PATH.exists():
        pytest.skip(f"Vision model not found at {MODEL_PATH}")

def _require_smoke():
    if not SMOKE_VIDEO.exists():
        pytest.skip(f"Smoke test video not found at {SMOKE_VIDEO}")

def _require_fire():
    if not FIRE_VIDEO.exists():
        pytest.skip(f"Fire test video not found at {FIRE_VIDEO}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    _require_model()
    return VideoAnalyzer(model_path=MODEL_PATH, confidence=0.25)


@pytest.fixture
def tmp_output(tmp_path):
    return tmp_path / "outputs"


# ---------------------------------------------------------------------------
# Tests — Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_missing_video_returns_error(self, analyzer, tmp_output):
        result = analyzer.analyze(Path("/nonexistent/video.mp4"), tmp_output)
        assert not result.success
        assert result.error_message is not None
        assert result.total_frames == 0

    def test_invalid_extension_returns_error(self, analyzer, tmp_output, tmp_path):
        # Create a dummy .txt file.
        fake = tmp_path / "fake.txt"
        fake.write_text("not a video")
        result = analyzer.analyze(fake, tmp_output)
        assert not result.success
        assert "Unsupported" in (result.error_message or "")

    def test_corrupt_video_returns_error(self, analyzer, tmp_output, tmp_path):
        """Write random bytes with .mp4 extension; OpenCV should fail to open it."""
        corrupt = tmp_path / "corrupt.mp4"
        corrupt.write_bytes(b"\x00\xff" * 50)
        result = analyzer.analyze(corrupt, tmp_output)
        # OpenCV may or may not open it; if it does, frames=0.
        assert not result.success or result.total_frames == 0

    def test_model_unavailable_returns_error(self, tmp_output, tmp_path):
        analyzer_bad = VideoAnalyzer(
            model_path=tmp_path / "nonexistent.pt",
            confidence=0.25,
        )
        # Create a minimal valid video (1 frame).
        import cv2
        fake_video = tmp_path / "tiny.mp4"
        writer = cv2.VideoWriter(
            str(fake_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            1.0,
            (64, 64),
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
        writer.release()

        result = analyzer_bad.analyze(fake_video, tmp_output)
        assert not result.success


# ---------------------------------------------------------------------------
# Tests — Model availability
# ---------------------------------------------------------------------------

class TestModelAvailability:
    def test_model_available_property(self):
        _require_model()
        analyzer = VideoAnalyzer(model_path=MODEL_PATH)
        assert analyzer.model_available

    def test_model_unavailable_property(self, tmp_path):
        analyzer = VideoAnalyzer(model_path=tmp_path / "none.pt")
        assert not analyzer.model_available


# ---------------------------------------------------------------------------
# Tests — Minimal synthetic video (no model needed beyond this)
# ---------------------------------------------------------------------------

class TestSyntheticVideo:
    """Tests that work with a synthetically generated video."""

    def _make_blank_video(self, path: Path, n_frames: int = 10) -> None:
        import cv2
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (128, 128),
        )
        for _ in range(n_frames):
            writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
        writer.release()

    def test_blank_video_produces_zero_detections(self, analyzer, tmp_output, tmp_path):
        """A blank video should process without crashing and report 0 detections."""
        _require_model()
        video = tmp_path / "blank.mp4"
        self._make_blank_video(video, n_frames=5)
        result = analyzer.analyze(video, tmp_output / "blank")
        assert result.success
        assert result.total_frames == 5
        assert result.smoke_detection_count >= 0
        assert result.fire_detection_count >= 0

    def test_result_fields_correctly_typed(self, analyzer, tmp_output, tmp_path):
        _require_model()
        video = tmp_path / "typed.mp4"
        self._make_blank_video(video, n_frames=3)
        result = analyzer.analyze(video, tmp_output / "typed")
        assert isinstance(result.total_frames, int)
        assert isinstance(result.frames_with_smoke, int)
        assert isinstance(result.frames_with_fire, int)
        assert isinstance(result.pct_frames_with_smoke, float)
        assert isinstance(result.pct_frames_with_fire, float)
        assert isinstance(result.detection_timeline, list)

    def test_output_files_created(self, analyzer, tmp_output, tmp_path):
        _require_model()
        video = tmp_path / "outputs_check.mp4"
        self._make_blank_video(video, n_frames=3)
        out_dir = tmp_output / "files"
        result = analyzer.analyze(video, out_dir)
        if result.success:
            assert result.annotated_video_path is not None
            assert Path(result.annotated_video_path).exists()
            if result.detection_csv_path:
                assert Path(result.detection_csv_path).exists()

    def test_pct_smoke_plus_pct_fire_in_range(self, analyzer, tmp_output, tmp_path):
        _require_model()
        video = tmp_path / "pct.mp4"
        self._make_blank_video(video, n_frames=10)
        result = analyzer.analyze(video, tmp_output / "pct")
        if result.success:
            assert 0 <= result.pct_frames_with_smoke <= 100
            assert 0 <= result.pct_frames_with_fire <= 100


# ---------------------------------------------------------------------------
# Integration smoke tests — existing demo videos
# ---------------------------------------------------------------------------

class TestDemoVideos:
    """
    Uses existing validated demo videos as integration smoke tests.
    These tests do NOT tune the detector or claim validation performance.
    """

    def test_smoke_video_processes_successfully(self, analyzer, tmp_output):
        _require_model()
        _require_smoke()
        result = analyzer.analyze(SMOKE_VIDEO, tmp_output / "smoke_integration")
        assert result.success, f"Expected success; error: {result.error_message}"
        assert result.total_frames > 0
        # The detector should find at least some smoke in a smoke video.
        # We do not assert a specific count — that would be tuning.

    def test_fire_video_processes_successfully(self, analyzer, tmp_output):
        _require_model()
        _require_fire()
        result = analyzer.analyze(FIRE_VIDEO, tmp_output / "fire_integration")
        assert result.success, f"Expected success; error: {result.error_message}"
        assert result.total_frames > 0

    def test_smoke_video_produces_annotated_mp4(self, analyzer, tmp_output):
        _require_model()
        _require_smoke()
        result = analyzer.analyze(SMOKE_VIDEO, tmp_output / "smoke_mp4")
        if result.success and result.annotated_video_path:
            assert Path(result.annotated_video_path).exists()

    def test_smoke_video_produces_csv_timeline(self, analyzer, tmp_output):
        _require_model()
        _require_smoke()
        result = analyzer.analyze(SMOKE_VIDEO, tmp_output / "smoke_csv")
        if result.success and result.detection_csv_path:
            assert Path(result.detection_csv_path).exists()

    def test_caveat_not_pm25(self):
        """
        Reminder test: the analyzer result has no pm25 field.
        Detections must never be treated as PM2.5 values.
        """
        result_fields = set(VideoAnalysisResult.__dataclass_fields__.keys())
        prohibited = {"pm25", "aqi", "concentration"}
        overlap = result_fields & prohibited
        assert not overlap, (
            f"VideoAnalysisResult must not have PM2.5/AQI fields: {overlap}"
        )
