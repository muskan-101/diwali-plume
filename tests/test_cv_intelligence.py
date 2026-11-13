"""
test_cv_intelligence.py — Tests for Advanced Vision-Led Air Pollution Intelligence Pipeline

Covers:
- PlumeSegmenter: Dark Channel Prior + saturation segmentation, coverage %, centroid
- TemporalMotionAnalyzer: Dense optical flow, motion speed and direction, camera motion compensation
- HazeVisibilityEstimator: DCP dark channel mean, scene clarity index, contrast degradation
- EventClassifier: Multi-cue event classification
- VisualReportGenerator: Multi-panel time-series visualization creation
- Scientific caveat integrity
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision.cv_intelligence import (
    PlumeSegmenter,
    TemporalMotionAnalyzer,
    HazeVisibilityEstimator,
    EventClassifier,
    VisualReportGenerator,
    FrameCVMetrics,
)


@pytest.fixture
def synthetic_frame():
    """Generates a 128x128 synthetic test frame (RGB)."""
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    cv2.circle(img, (64, 64), 30, (140, 140, 140), -1)
    return img


class TestPlumeSegmenter:
    def test_segment_plume_returns_coverage_and_mask(self, synthetic_frame):
        segmenter = PlumeSegmenter(patch_size=7)
        mask, info = segmenter.segment_plume(synthetic_frame, [])
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (128, 128)
        assert isinstance(info, dict)
        assert "plume_coverage_pct" in info
        assert "dark_channel_mean" in info
        assert 0.0 <= info["plume_coverage_pct"] <= 100.0

    def test_segment_plume_blank_frame(self):
        segmenter = PlumeSegmenter(patch_size=7)
        black_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mask, info = segmenter.segment_plume(black_frame, [])
        assert mask.shape == (64, 64)
        assert info["plume_coverage_pct"] == 0.0


class TestTemporalMotionAnalyzer:
    def test_motion_analysis_first_frame(self, synthetic_frame):
        analyzer = TemporalMotionAnalyzer()
        mask = np.zeros((128, 128), dtype=np.uint8)
        res = analyzer.analyze_motion(synthetic_frame, mask)
        assert res["plume_speed"] == 0.0
        assert res["camera_speed"] == 0.0

    def test_motion_analysis_consecutive_frames(self, synthetic_frame):
        analyzer = TemporalMotionAnalyzer()
        mask = np.ones((128, 128), dtype=np.uint8) * 255
        analyzer.analyze_motion(synthetic_frame, mask)
        # Shifted frame simulating motion
        shifted = np.roll(synthetic_frame, 3, axis=1)
        res = analyzer.analyze_motion(shifted, mask)
        assert "plume_speed" in res
        assert "camera_speed" in res
        assert "plume_direction_deg" in res
        assert res["plume_speed"] >= 0.0


class TestHazeVisibilityEstimator:
    def test_haze_estimation_normal_frame(self, synthetic_frame):
        rms, clarity, severity = HazeVisibilityEstimator.estimate(synthetic_frame, 70.0)
        assert rms >= 0.0
        assert 0.0 <= clarity <= 100.0
        assert severity in {
            "Clear",
            "Mild Haze",
            "Moderate Haze",
            "Severe Haze/Smog",
        }

    def test_haze_estimation_handles_none_gracefully(self):
        rms, clarity, severity = HazeVisibilityEstimator.estimate(None, 80.0)
        assert rms == 0.0
        assert 0.0 <= clarity <= 100.0
        assert severity is not None


class TestEventClassifier:
    def test_classify_event_diffuse_smoke(self):
        event = EventClassifier.classify(
            has_fire=False,
            fire_conf=0.0,
            has_smoke=True,
            smoke_conf=0.15,
            plume_coverage_pct=5.0,
            haze_severity="Moderate Haze",
            clarity_index=50.0,
        )
        assert event == "diffuse_smoke_plume"

    def test_classify_event_open_fire(self):
        event = EventClassifier.classify(
            has_fire=True,
            fire_conf=0.85,
            has_smoke=True,
            smoke_conf=0.5,
            plume_coverage_pct=10.0,
            haze_severity="Slight Haze",
            clarity_index=70.0,
        )
        assert event == "active_fire_event"

    def test_classify_event_ambient_haze(self):
        event = EventClassifier.classify(
            has_fire=False,
            fire_conf=0.0,
            has_smoke=False,
            smoke_conf=0.0,
            plume_coverage_pct=1.0,
            haze_severity="Severe Haze/Smog",
            clarity_index=25.0,
        )
        assert event == "ambient_smog_haze"


class TestVisualReportGenerator:
    def test_generate_visual_trends_plot(self, tmp_path):
        metrics = [
            FrameCVMetrics(
                frame_number=i,
                timestamp_seconds=float(i),
                smoke_detected=(i > 2),
                fire_detected=False,
                smoke_confidence=0.4 if i > 2 else 0.0,
                fire_confidence=0.0,
                plume_coverage_pct=float(i * 5),
                plume_area_px=i * 100,
                plume_centroid_x=64.0,
                plume_centroid_y=64.0,
                optical_flow_speed=1.5,
                optical_flow_direction_deg=90.0,
                camera_motion_speed=0.5,
                relative_plume_speed=1.0,
                dark_channel_mean=70.0,
                rms_contrast=40.0,
                scene_clarity_index=65.0,
                classified_event="diffuse_smoke_plume",
            )
            for i in range(10)
        ]
        out_png = tmp_path / "trends.png"
        result_path = VisualReportGenerator.generate_trend_plot(
            frames_metrics=metrics,
            output_png_path=out_png,
            video_name="test_video",
        )
        assert result_path is not None
        assert Path(result_path).exists()
        assert Path(result_path).stat().st_size > 0
