"""
video_analyzer.py — Diwali Plume Stage A

Wraps the existing smoke_fire_detector.py logic into an importable,
integration-friendly class that:
  - accepts an uploaded video path and an output directory
  - runs YOLOv8n inference on every frame (CPU)
  - produces an annotated MP4 and a detection-timeline CSV
  - returns structured video-level summary statistics

IMPORTANT caveats (must be communicated to callers):
  - Detection confidence is NOT a PM2.5 measurement.
  - Smoke/fire bounding boxes are NOT direct pollution readings.
  - This model identifies visual patterns; it does NOT quantify air quality.
"""

from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = Path("models/vision/fire_smoke_yolov8n_dfire_best.pt")
DEFAULT_CONFIDENCE = 0.25

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CLASS_SMOKE = "smoke"
CLASS_FIRE  = "fire"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VideoAnalysisResult:
    """Summary statistics for one analysed video."""

    # Paths
    input_video_path:    str
    annotated_video_path: Optional[str]
    detection_csv_path:  Optional[str]

    # Frame stats
    total_frames: int
    frames_with_smoke: int
    frames_with_fire: int

    # Detection counts (one entry per bounding box)
    smoke_detection_count: int
    fire_detection_count: int

    # Confidence
    avg_smoke_confidence: Optional[float]
    avg_fire_confidence:  Optional[float]
    max_confidence:       Optional[float]

    # Percentage of frames
    pct_frames_with_smoke: float
    pct_frames_with_fire:  float

    # Detection timeline (list of dicts with frame / timestamp / class / confidence)
    detection_timeline: list[dict] = field(default_factory=list)

    # Status
    success: bool = True
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# VideoAnalyzer
# ---------------------------------------------------------------------------


class VideoAnalyzer:
    """
    Runs YOLOv8n smoke/fire inference on an uploaded video.

    Parameters
    ----------
    model_path : Path to the YOLOv8n .pt weight file.
    confidence : Detection confidence threshold (0–1).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self._model = None   # Lazy-load on first call.

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        from ultralytics import YOLO   # import here to keep startup fast
        if not self.model_path.exists():
            raise FileNotFoundError(f"Vision model not found: {self.model_path}")
        self._model = YOLO(str(self.model_path))
        logger.info("Loaded YOLO model from %s", self.model_path)

    def _ensure_model(self) -> None:
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        video_path: Path,
        output_dir: Path,
        output_stem: Optional[str] = None,
    ) -> VideoAnalysisResult:
        """
        Process a video and return a VideoAnalysisResult.

        Parameters
        ----------
        video_path  : Path to the input video.
        output_dir  : Directory where annotated video and CSV are saved.
        output_stem : Base filename (without extension) for outputs.
                      Defaults to the input video stem.

        Returns
        -------
        VideoAnalysisResult — always returned; check .success.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)

        # --- validation ---
        if not video_path.exists():
            return _error_result(str(video_path), f"Video not found: {video_path}")

        if video_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return _error_result(
                str(video_path),
                f"Unsupported file extension '{video_path.suffix}'. "
                f"Allowed: {ALLOWED_EXTENSIONS}",
            )

        # --- open video ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return _error_result(str(video_path), f"Cannot open video: {video_path}")

        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width == 0 or height == 0:
            cap.release()
            return _error_result(str(video_path), "Video has zero dimensions — file may be corrupt.")

        # --- setup outputs ---
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_stem or video_path.stem
        annotated_path = output_dir / f"{stem}_annotated.mp4"
        csv_path       = output_dir / f"{stem}_timeline.csv"

        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            return _error_result(
                str(video_path), "Could not create output video writer."
            )

        # --- load model (once) ---
        try:
            self._ensure_model()
        except Exception as exc:
            cap.release()
            writer.release()
            return _error_result(str(video_path), f"Model load error: {exc}")

        # --- inference loop ---
        rows: list[dict] = []
        smoke_confs: list[float] = []
        fire_confs:  list[float] = []
        smoke_frame_set: set[int] = set()
        fire_frame_set:  set[int] = set()
        smoke_det_count = 0
        fire_det_count  = 0
        frame_number    = 0
        warnings:  list[str] = []

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_number += 1
                timestamp_s  = (frame_number - 1) / fps

                try:
                    result = self._model.predict(
                        source=frame,
                        conf=self.confidence,
                        device="cpu",
                        verbose=False,
                    )[0]
                except Exception as exc:
                    warnings.append(f"Frame {frame_number}: inference error — {exc}")
                    writer.write(frame)
                    continue

                annotated = result.plot()
                writer.write(annotated)

                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls.item())
                        conf   = float(box.conf.item())
                        label  = self._model.names[cls_id]

                        rows.append({
                            "frame":             frame_number,
                            "timestamp_seconds": round(timestamp_s, 3),
                            "class":             label,
                            "confidence":        round(conf, 4),
                        })

                        if label == CLASS_SMOKE:
                            smoke_det_count += 1
                            smoke_confs.append(conf)
                            smoke_frame_set.add(frame_number)
                        elif label == CLASS_FIRE:
                            fire_det_count  += 1
                            fire_confs.append(conf)
                            fire_frame_set.add(frame_number)

                if frame_number % 100 == 0:
                    logger.debug("Processed %d frames", frame_number)

        except Exception as exc:
            warnings.append(f"Processing interrupted at frame {frame_number}: {exc}")
        finally:
            cap.release()
            writer.release()

        if frame_number == 0:
            return _error_result(str(video_path), "Video contains no readable frames.")

        # --- write timeline CSV ---
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer_csv = csv.DictWriter(
                    fh, fieldnames=["frame", "timestamp_seconds", "class", "confidence"]
                )
                writer_csv.writeheader()
                writer_csv.writerows(rows)
        except Exception as exc:
            warnings.append(f"Could not write CSV timeline: {exc}")
            csv_path = None

        # --- aggregate statistics ---
        total = frame_number
        avg_smoke = round(sum(smoke_confs) / len(smoke_confs), 4) if smoke_confs else None
        avg_fire  = round(sum(fire_confs)  / len(fire_confs),  4) if fire_confs  else None
        all_confs = smoke_confs + fire_confs
        max_conf  = round(max(all_confs), 4) if all_confs else None

        pct_smoke = round(100.0 * len(smoke_frame_set) / total, 2) if total else 0.0
        pct_fire  = round(100.0 * len(fire_frame_set)  / total, 2) if total else 0.0

        return VideoAnalysisResult(
            input_video_path=str(video_path),
            annotated_video_path=str(annotated_path),
            detection_csv_path=str(csv_path) if csv_path else None,
            total_frames=total,
            frames_with_smoke=len(smoke_frame_set),
            frames_with_fire=len(fire_frame_set),
            smoke_detection_count=smoke_det_count,
            fire_detection_count=fire_det_count,
            avg_smoke_confidence=avg_smoke,
            avg_fire_confidence=avg_fire,
            max_confidence=max_conf,
            pct_frames_with_smoke=pct_smoke,
            pct_frames_with_fire=pct_fire,
            detection_timeline=rows,
            success=True,
            error_message=None,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def model_available(self) -> bool:
        return self.model_path.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(video_path: str, message: str) -> VideoAnalysisResult:
    logger.error("VideoAnalyzer error for %s: %s", video_path, message)
    return VideoAnalysisResult(
        input_video_path=video_path,
        annotated_video_path=None,
        detection_csv_path=None,
        total_frames=0,
        frames_with_smoke=0,
        frames_with_fire=0,
        smoke_detection_count=0,
        fire_detection_count=0,
        avg_smoke_confidence=None,
        avg_fire_confidence=None,
        max_confidence=None,
        pct_frames_with_smoke=0.0,
        pct_frames_with_fire=0.0,
        detection_timeline=[],
        success=False,
        error_message=message,
        warnings=[],
    )
