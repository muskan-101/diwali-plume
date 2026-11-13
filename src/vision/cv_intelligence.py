"""
cv_intelligence.py — Advanced Vision-Led Air Pollution Intelligence Pipeline

Components:
  1. SmokeFireDetector: YOLOv8n inference with confidence tracking, duplicate handling,
     and temporal IoU-based box persistence.
  2. PlumeSegmenter: Physics-grounded plume & smoke segmentation using Dark Channel Prior,
     chromatic saturation, and morphological contours.
  3. TemporalMotionAnalyzer: Farneback dense optical flow separating camera movement
     from local plume dispersion and velocity.
  4. HazeVisibilityEstimator: Scene clarity, Dark Channel Prior transmission, and contrast degradation.
  5. EventClassifier: Multi-cue event classification (fire, plume, fireworks, ambient smog, uncertain).
  6. VisualReportGenerator: Multi-panel time-series visualization plots.
  7. CVIntelligencePipeline: Unified end-to-end analyzer producing video annotations,
     detection timeline CSV, frame-level feature CSV, visual trend PNG, and structured JSON.

Scientific Integrity Notice:
  Visual metrics represent optical and photometric evidence from camera frames.
  They do NOT constitute direct atmospheric PM2.5 or PM10 mass concentration measurements.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

# Ensure matplotlib cache directory is writable and headless
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "data", "outputs", ".matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/vision/fire_smoke_yolov8n_dfire_best.pt")
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class FrameCVMetrics:
    frame_number: int
    timestamp_seconds: float
    smoke_detected: bool
    fire_detected: bool
    smoke_confidence: float
    fire_confidence: float
    plume_coverage_pct: float
    plume_area_px: int
    plume_centroid_x: Optional[float]
    plume_centroid_y: Optional[float]
    optical_flow_speed: float
    optical_flow_direction_deg: float
    camera_motion_speed: float
    relative_plume_speed: float
    dark_channel_mean: float
    rms_contrast: float
    scene_clarity_index: float  # 0 (severe smog) to 100 (crystal clear)
    classified_event: str


@dataclass
class CVAnalysisSummary:
    video_path: str
    annotated_video_path: Optional[str]
    timeline_csv_path: Optional[str]
    features_csv_path: Optional[str]
    trend_graph_path: Optional[str]
    total_frames: int
    sampled_frames: int
    sampling_stride: int
    duration_seconds: float
    fps: float
    frame_width: int
    frame_height: int
    
    # Detection counts & frequencies
    frames_with_smoke: int
    frames_with_fire: int
    pct_frames_with_smoke: float
    pct_frames_with_fire: float
    total_smoke_detections: int
    total_fire_detections: int
    avg_smoke_confidence: Optional[float]
    avg_fire_confidence: Optional[float]
    max_confidence: Optional[float]
    
    # Plume morphological stats
    max_plume_coverage_pct: float
    avg_plume_coverage_pct: float
    plume_expansion_trend: str  # "expanding", "contracting", "stable", "negligible"
    
    # Motion & dynamics
    dominant_plume_direction: Optional[str]
    avg_apparent_motion_speed: float
    camera_motion_assessment: str  # "stable", "panning_or_jitter", "insufficient_motion"
    
    # Haze & visibility
    avg_dark_channel: float
    avg_scene_clarity_index: float
    haze_severity: str  # "Severe Haze/Smog", "Moderate Haze", "Mild Haze", "Clear"
    
    # Event classification
    primary_event_classification: str
    event_confidence_score: float
    
    # Quality & Integrity
    detection_timeline: List[Dict[str, Any]] = field(default_factory=list)
    confidence_uncertainty_notes: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Smoke & Fire Detector (YOLOv8n wrapper with duplicate suppression)
# ---------------------------------------------------------------------------

class SmokeFireDetector:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, conf_threshold: float = 0.15, imgsz: int = 320):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            torch.set_num_threads(4)
            from ultralytics import YOLO
            if not self.model_path.exists():
                raise FileNotFoundError(f"Vision weights not found at {self.model_path}")
            self._model = YOLO(str(self.model_path))
            # Warm up model
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._model.predict(dummy, imgsz=self.imgsz, device="cpu", verbose=False)

    def detect(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        self._ensure_model()
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device="cpu",
            verbose=False,
        )[0]
        
        detections = []
        if results.boxes is not None:
            for b in results.boxes:
                cls_id = int(b.cls.item())
                score = float(b.conf.item())
                label = self._model.names[cls_id]
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                detections.append({
                    "class": label,
                    "confidence": round(score, 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                })
        
        # Suppress near-identical duplicate boxes (IoU > 0.8)
        filtered = self._suppress_duplicate_boxes(detections)
        return filtered, results.plot()

    @staticmethod
    def _suppress_duplicate_boxes(boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(boxes) <= 1:
            return boxes
        kept = []
        for b in sorted(boxes, key=lambda x: x["confidence"], reverse=True):
            overlap = False
            for k in kept:
                if b["class"] == k["class"]:
                    iou = _compute_iou(b["box"], k["box"])
                    if iou > 0.80:
                        overlap = True
                        break
            if not overlap:
                kept.append(b)
        return kept


def _compute_iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 2. Plume Segmenter (Physics-grounded Dark Channel + Chromatic Analysis)
# ---------------------------------------------------------------------------

class PlumeSegmenter:
    """
    Estimates smoke and plume coverage per frame using dark channel prior and
    color saturation degradation.
    Labelled as experimental research estimation.
    """
    def __init__(self, patch_size: int = 15, dcp_threshold: float = 75.0):
        self.patch_size = patch_size
        self.dcp_threshold = dcp_threshold
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))

    def segment_plume(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]
        total_pixels = h * w
        
        # 1. Dark Channel Prior
        min_channel = np.min(frame, axis=2)
        dark_channel = cv2.erode(min_channel, self.kernel)
        
        # 2. Saturation & Brightness
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        # Smoke characteristic: elevated dark channel, moderate-to-high brightness, low color saturation
        mask_dcp = (dark_channel >= self.dcp_threshold)
        mask_sat = (sat < 90)
        mask_val = (val > 60)
        combined_mask = (mask_dcp & mask_sat & mask_val).astype(np.uint8) * 255
        
        # If smoke detections exist from YOLO, heavily weight regions in or around bounding boxes
        if detections:
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            for d in detections:
                if d["class"] == "smoke":
                    bx = [int(v) for v in d["box"]]
                    cv2.rectangle(bbox_mask, (bx[0], bx[1]), (bx[2], bx[3]), 255, -1)
            # Dilate bbox to capture surrounding dispersing plume
            bbox_mask = cv2.dilate(bbox_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
            combined_mask = cv2.bitwise_or(combined_mask, cv2.bitwise_and(combined_mask, bbox_mask))

        # Morphological opening and closing to clean noise
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        
        plume_px = int(np.count_nonzero(cleaned_mask))
        coverage_pct = round((plume_px / total_pixels) * 100.0, 2)
        
        # Centroid calculation
        moments = cv2.moments(cleaned_mask)
        if moments["m00"] > 0:
            cx = round(moments["m10"] / moments["m00"], 1)
            cy = round(moments["m01"] / moments["m00"], 1)
        else:
            cx, cy = None, None

        return cleaned_mask, {
            "plume_area_px": plume_px,
            "plume_coverage_pct": coverage_pct,
            "centroid_x": cx,
            "centroid_y": cy,
            "dark_channel_mean": round(float(np.mean(dark_channel)), 2),
        }


# ---------------------------------------------------------------------------
# 3. Temporal Motion Analyzer (Dense Optical Flow)
# ---------------------------------------------------------------------------

class TemporalMotionAnalyzer:
    """
    Computes Farneback dense optical flow to capture plume movement velocity,
    direction, and distinguishes global camera movement.
    """
    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset previous frame memory across video files."""
        self.prev_gray = None

    def analyze_motion(self, frame: np.ndarray, plume_mask: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return {
                "plume_speed": 0.0,
                "plume_direction_deg": 0.0,
                "camera_speed": 0.0,
                "relative_speed": 0.0,
                "flow_u": 0.0,
                "flow_v": 0.0,
            }
        
        # Farneback Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        self.prev_gray = gray
        
        u = flow[..., 0]
        v = flow[..., 1]
        mag, ang = cv2.cartToPolar(u, v, angleInDegrees=True)
        
        # Global camera motion: robust median across the frame
        cam_speed = float(np.median(mag))
        
        # Local plume motion: inside the plume mask if plume exists
        if np.count_nonzero(plume_mask) > 100:
            mask_bool = (plume_mask > 0)
            plume_mag = float(np.mean(mag[mask_bool]))
            plume_u = float(np.mean(u[mask_bool]))
            plume_v = float(np.mean(v[mask_bool]))
            plume_ang = float(np.mean(ang[mask_bool]))
        else:
            plume_mag = cam_speed
            plume_u = float(np.median(u))
            plume_v = float(np.median(v))
            plume_ang = float(np.median(ang))
            
        rel_speed = max(0.0, plume_mag - cam_speed)

        return {
            "plume_speed": round(plume_mag, 2),
            "plume_direction_deg": round(plume_ang, 1),
            "camera_speed": round(cam_speed, 2),
            "relative_speed": round(rel_speed, 2),
            "flow_u": round(plume_u, 2),
            "flow_v": round(plume_v, 2),
        }


# ---------------------------------------------------------------------------
# 4. Haze & Visibility Estimator
# ---------------------------------------------------------------------------

class HazeVisibilityEstimator:
    @staticmethod
    def estimate(frame: Optional[np.ndarray], dcp_mean: float) -> Tuple[float, float, str]:
        if frame is None or frame.size == 0:
            if dcp_mean > 95.0:
                sev = "Severe Haze/Smog"
            elif dcp_mean > 65.0:
                sev = "Moderate Haze"
            elif dcp_mean > 40.0:
                sev = "Mild Haze"
            else:
                sev = "Clear"
            cl = round(max(0.0, 100.0 - (dcp_mean / 255.0 * 85.0)), 1)
            return 0.0, cl, sev

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rms_contrast = float(np.std(gray))
        
        # Clarity index: high contrast + low dark channel -> high clarity (0-100)
        # Clear sky DCP is ~15-30, heavy smog DCP is 100-200.
        clarity = 100.0 - (dcp_mean / 255.0 * 85.0) - (max(0.0, 50.0 - rms_contrast) * 0.4)
        clarity = round(max(0.0, min(100.0, clarity)), 1)
        
        if dcp_mean > 95.0 or clarity < 35.0:
            severity = "Severe Haze/Smog"
        elif dcp_mean > 65.0 or clarity < 55.0:
            severity = "Moderate Haze"
        elif dcp_mean > 40.0:
            severity = "Mild Haze"
        else:
            severity = "Clear"
            
        return round(rms_contrast, 2), clarity, severity


# ---------------------------------------------------------------------------
# 5. Multi-Cue Event Classifier
# ---------------------------------------------------------------------------

class EventClassifier:
    @staticmethod
    def classify(
        has_fire: bool,
        fire_conf: float,
        has_smoke: bool,
        smoke_conf: float,
        plume_coverage_pct: float,
        haze_severity: str,
        clarity_index: float,
    ) -> str:
        if has_fire and fire_conf >= 0.25:
            return "active_fire_event"
        if has_smoke and (smoke_conf >= 0.20 or plume_coverage_pct >= 8.0):
            return "dense_smoke_plume"
        if plume_coverage_pct >= 4.0:
            return "diffuse_smoke_plume"
        if haze_severity in ("Severe Haze/Smog", "Moderate Haze") or clarity_index < 45.0:
            return "ambient_smog_haze"
        if smoke_conf > 0.05:
            return "low_confidence_smoke_signal"
        return "normal_ambient_scene"


# ---------------------------------------------------------------------------
# 6. Visual Trend Report Generator
# ---------------------------------------------------------------------------

class VisualReportGenerator:
    @staticmethod
    def generate_trend_plot(
        frames_metrics: List[FrameCVMetrics],
        output_png_path: Path,
        video_name: str,
    ) -> Optional[str]:
        if not frames_metrics:
            return None
        
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        times = [m.timestamp_seconds for m in frames_metrics]
        smoke_c = [m.smoke_confidence for m in frames_metrics]
        fire_c = [m.fire_confidence for m in frames_metrics]
        coverage = [m.plume_coverage_pct for m in frames_metrics]
        speed = [m.optical_flow_speed for m in frames_metrics]
        dcp = [m.dark_channel_mean for m in frames_metrics]
        clarity = [m.scene_clarity_index for m in frames_metrics]

        fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        fig.suptitle(f"Diwali Plume — CV Intelligence Multi-Signal Timeline\nSource: {video_name}", fontsize=13, weight="bold")
        
        # Subplot 1: Confidence
        axs[0].plot(times, smoke_c, label="Smoke Confidence", color="#ff7f0e", linewidth=1.8)
        axs[0].plot(times, fire_c, label="Fire Confidence", color="#d62728", linewidth=1.8)
        axs[0].set_ylabel("Detection Conf")
        axs[0].set_ylim(0, 1.0)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")

        # Subplot 2: Plume Coverage %
        axs[1].fill_between(times, coverage, color="#7f7f7f", alpha=0.4)
        axs[1].plot(times, coverage, color="#4d4d4d", linewidth=1.8, label="Plume Coverage (%)")
        axs[1].set_ylabel("Coverage (%)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")

        # Subplot 3: Apparent Optical Flow Motion
        axs[2].plot(times, speed, color="#1f77b4", linewidth=1.5, label="Apparent Flow Speed (px/f)")
        axs[2].set_ylabel("Motion (px/f)")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="upper right")

        # Subplot 4: Dark Channel Haze & Scene Clarity Index
        ax4_2 = axs[3].twinx()
        axs[3].plot(times, dcp, color="#8c564b", linewidth=1.5, label="Dark Channel Haze (DCP)")
        ax4_2.plot(times, clarity, color="#2ca02c", linewidth=1.5, linestyle="--", label="Clarity Index (0-100)")
        axs[3].set_ylabel("DCP Mean")
        ax4_2.set_ylabel("Clarity (0-100)")
        axs[3].set_xlabel("Time (seconds)")
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc="upper left")
        ax4_2.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(str(output_png_path), dpi=150)
        plt.close(fig)
        return str(output_png_path)


def _ensure_h264_web_compatible(mp4_path: Path) -> None:
    """Re-encodes an mp4v video to web-standard H.264 (yuv420p) so browsers can play it natively."""
    import shutil
    import subprocess
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe or not mp4_path.exists():
        return
    tmp_path = mp4_path.with_name(f"web_{mp4_path.name}")
    try:
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", str(mp4_path),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            shutil.move(str(tmp_path), str(mp4_path))
    except Exception as exc:
        logger.warning("ffmpeg web transcoding skipped: %s", exc)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 7. Unified CV Intelligence Pipeline
# ---------------------------------------------------------------------------

class CVIntelligencePipeline:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.10,
    ):
        self.detector = SmokeFireDetector(model_path, conf_threshold)
        self.segmenter = PlumeSegmenter()
        self.motion_analyzer = TemporalMotionAnalyzer()
        self.event_classifier = EventClassifier()

    def analyze_video(
        self,
        video_path: Path,
        output_dir: Path,
        output_stem: Optional[str] = None,
        sample_stride: Optional[int] = None,
        max_sampled_frames: int = 300,
    ) -> CVAnalysisSummary:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = output_stem or video_path.stem

        if not video_path.exists():
            return self._fail(str(video_path), f"Video file not found: {video_path}")

        if video_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return self._fail(str(video_path), f"Unsupported video extension '{video_path.suffix}'")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self._fail(str(video_path), f"OpenCV cannot open video stream: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_cap_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_cap_frames <= 0:
            total_cap_frames = 1

        if sample_stride is None:
            if total_cap_frames > max_sampled_frames:
                effective_stride = max(1, total_cap_frames // max_sampled_frames)
            else:
                effective_stride = 1
        else:
            effective_stride = max(1, sample_stride)

        if width <= 0 or height <= 0:
            cap.release()
            return self._fail(str(video_path), "Video has zero spatial dimensions")

        # Reset optical flow state for fresh video
        self.motion_analyzer.reset()

        annotated_path = output_dir / f"{stem}_annotated.mp4"
        timeline_csv_path = output_dir / f"{stem}_timeline.csv"
        features_csv_path = output_dir / f"{stem}_features.csv"
        trend_png_path = output_dir / f"{stem}_visual_trends.png"

        writer: Optional[cv2.VideoWriter] = None

        frame_metrics_list: List[FrameCVMetrics] = []
        timeline_rows: List[Dict[str, Any]] = []
        smoke_confs: List[float] = []
        fire_confs: List[float] = []
        coverages: List[float] = []
        dcp_means: List[float] = []
        clarities: List[float] = []
        speeds: List[float] = []
        directions: List[float] = []
        events: List[str] = []

        smoke_frame_count = 0
        fire_frame_count = 0
        total_smoke_boxes = 0
        total_fire_boxes = 0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if effective_stride > 1 and (frame_idx % effective_stride != 0):
                    continue

                # Ensure writer is matched to real frame dimensions
                if writer is None:
                    real_h, real_w = frame.shape[:2]
                    width, height = real_w, real_h
                    writer = cv2.VideoWriter(
                        str(annotated_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        max(1.0, fps / effective_stride),
                        (width, height),
                    )

                timestamp_s = round((frame_idx - 1) / fps, 3)

                # 1. Detection
                detections, annotated_frame = self.detector.detect(frame)
                
                cur_smoke_conf = 0.0
                cur_fire_conf = 0.0
                has_smoke = False
                has_fire = False

                for d in detections:
                    c = d["confidence"]
                    lbl = d["class"]
                    timeline_rows.append({
                        "frame": frame_idx,
                        "timestamp_seconds": timestamp_s,
                        "class": lbl,
                        "confidence": c,
                        "box": d["box"]
                    })
                    if lbl == "smoke":
                        has_smoke = True
                        total_smoke_boxes += 1
                        smoke_confs.append(c)
                        if c > cur_smoke_conf:
                            cur_smoke_conf = c
                    elif lbl == "fire":
                        has_fire = True
                        total_fire_boxes += 1
                        fire_confs.append(c)
                        if c > cur_fire_conf:
                            cur_fire_conf = c

                if has_smoke:
                    smoke_frame_count += 1
                if has_fire:
                    fire_frame_count += 1

                # 2. Plume Segmentation
                plume_mask, seg_stats = self.segmenter.segment_plume(frame, detections)
                coverage_pct = seg_stats["plume_coverage_pct"]
                coverages.append(coverage_pct)
                dcp_means.append(seg_stats["dark_channel_mean"])

                # 3. Motion & Flow
                motion_stats = self.motion_analyzer.analyze_motion(frame, plume_mask)
                speeds.append(motion_stats["plume_speed"])
                if motion_stats["plume_speed"] > 0.5:
                    directions.append(motion_stats["plume_direction_deg"])

                # 4. Haze & Visibility
                rms_cont, clarity, severity = HazeVisibilityEstimator.estimate(
                    frame, seg_stats["dark_channel_mean"]
                )
                clarities.append(clarity)

                # 5. Event Classification
                event_lbl = self.event_classifier.classify(
                    has_fire, cur_fire_conf, has_smoke, cur_smoke_conf,
                    coverage_pct, severity, clarity
                )
                events.append(event_lbl)

                # 6. Overlay HUD & Contours onto Annotated Video
                # Contour overlay for plume
                contours, _ = cv2.findContours(plume_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated_frame, contours, -1, (0, 165, 255), 1)

                # Telemetry HUD banner
                hud_text = (
                    f"F:{frame_idx} | T:{timestamp_s:.1f}s | "
                    f"Plume:{coverage_pct:.1f}% | "
                    f"Haze:{severity} | Clarity:{clarity:.0f}"
                )
                cv2.rectangle(annotated_frame, (10, 10), (width - 10, 42), (0, 0, 0), -1)
                cv2.putText(annotated_frame, hud_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
                
                writer.write(annotated_frame)

                if frame_idx % 20 == 0 or frame_idx == 1:
                    logger.info("Processed frame %d / %d", frame_idx, total_cap_frames)

                # Store frame metrics
                frame_metrics_list.append(FrameCVMetrics(
                    frame_number=frame_idx,
                    timestamp_seconds=timestamp_s,
                    smoke_detected=has_smoke,
                    fire_detected=has_fire,
                    smoke_confidence=round(cur_smoke_conf, 4),
                    fire_confidence=round(cur_fire_conf, 4),
                    plume_coverage_pct=coverage_pct,
                    plume_area_px=seg_stats["plume_area_px"],
                    plume_centroid_x=seg_stats["centroid_x"],
                    plume_centroid_y=seg_stats["centroid_y"],
                    optical_flow_speed=motion_stats["plume_speed"],
                    optical_flow_direction_deg=motion_stats["plume_direction_deg"],
                    camera_motion_speed=motion_stats["camera_speed"],
                    relative_plume_speed=motion_stats["relative_speed"],
                    dark_channel_mean=seg_stats["dark_channel_mean"],
                    rms_contrast=rms_cont,
                    scene_clarity_index=clarity,
                    classified_event=event_lbl,
                ))

        finally:
            cap.release()
            if writer is not None:
                writer.release()
                _ensure_h264_web_compatible(annotated_path)

        if not frame_metrics_list:
            return self._fail(str(video_path), "No readable video frames found.")

        # --- Write Timeline CSV ---
        with open(timeline_csv_path, "w", newline="", encoding="utf-8") as f:
            w_csv = csv.DictWriter(f, fieldnames=["frame", "timestamp_seconds", "class", "confidence", "box"])
            w_csv.writeheader()
            w_csv.writerows(timeline_rows)

        # --- Write Frame-Level Features CSV ---
        with open(features_csv_path, "w", newline="", encoding="utf-8") as f:
            w_csv = csv.DictWriter(f, fieldnames=list(asdict(frame_metrics_list[0]).keys()))
            w_csv.writeheader()
            for fm in frame_metrics_list:
                w_csv.writerow(asdict(fm))

        # --- Generate Multi-Panel Trend Plot ---
        VisualReportGenerator.generate_trend_plot(
            frame_metrics_list, trend_png_path, video_path.name
        )

        # Aggregate Statistics
        n_frames = len(frame_metrics_list)
        pct_smoke = round((smoke_frame_count / n_frames) * 100.0, 2)
        pct_fire = round((fire_frame_count / n_frames) * 100.0, 2)
        avg_smoke_c = round(float(np.mean(smoke_confs)), 3) if smoke_confs else None
        avg_fire_c = round(float(np.mean(fire_confs)), 3) if fire_confs else None
        all_c = smoke_confs + fire_confs
        max_c = round(float(np.max(all_c)), 3) if all_c else None

        avg_cov = round(float(np.mean(coverages)), 2)
        max_cov = round(float(np.max(coverages)), 2)
        
        # Area growth trend
        if len(coverages) >= 4:
            cov_start = np.mean(coverages[:len(coverages)//3])
            cov_end = np.mean(coverages[-len(coverages)//3:])
            if cov_end - cov_start > 2.0:
                expansion_trend = "expanding"
            elif cov_start - cov_end > 2.0:
                expansion_trend = "contracting"
            else:
                expansion_trend = "stable"
        else:
            expansion_trend = "negligible"

        # Motion & Direction
        avg_speed = round(float(np.mean(speeds)), 2)
        dominant_dir = _cardinal_direction(float(np.median(directions))) if directions else None
        median_cam = float(np.median([m.camera_motion_speed for m in frame_metrics_list]))
        cam_assessment = "panning_or_jitter" if median_cam > 3.0 else ("stable" if median_cam < 1.0 else "mild_camera_drift")

        # Haze
        mean_dcp = round(float(np.mean(dcp_means)), 1)
        mean_clarity = round(float(np.mean(clarities)), 1)
        _, _, overall_haze = HazeVisibilityEstimator.estimate(frame, mean_dcp)

        # Primary Event Classification
        event_counts = Counter(events)
        primary_event = event_counts.most_common(1)[0][0]
        event_conf = round(event_counts[primary_event] / n_frames, 2)

        return CVAnalysisSummary(
            video_path=str(video_path),
            annotated_video_path=str(annotated_path),
            timeline_csv_path=str(timeline_csv_path),
            features_csv_path=str(features_csv_path),
            trend_graph_path=str(trend_png_path),
            total_frames=total_cap_frames,
            sampled_frames=n_frames,
            sampling_stride=effective_stride,
            duration_seconds=round(total_cap_frames / fps, 2),
            fps=round(fps, 2),
            frame_width=width,
            frame_height=height,
            frames_with_smoke=smoke_frame_count,
            frames_with_fire=fire_frame_count,
            pct_frames_with_smoke=pct_smoke,
            pct_frames_with_fire=pct_fire,
            total_smoke_detections=total_smoke_boxes,
            total_fire_detections=total_fire_boxes,
            avg_smoke_confidence=avg_smoke_c,
            avg_fire_confidence=avg_fire_c,
            max_confidence=max_c,
            max_plume_coverage_pct=max_cov,
            avg_plume_coverage_pct=avg_cov,
            plume_expansion_trend=expansion_trend,
            dominant_plume_direction=dominant_dir,
            avg_apparent_motion_speed=avg_speed,
            camera_motion_assessment=cam_assessment,
            avg_dark_channel=mean_dcp,
            avg_scene_clarity_index=mean_clarity,
            haze_severity=overall_haze,
            primary_event_classification=primary_event,
            event_confidence_score=event_conf,
            detection_timeline=timeline_rows,
            confidence_uncertainty_notes=[
                "Smoke bounding boxes and plume coverage represent visual optical properties.",
                "Lighting, sunlight angle, vehicle headlights, and atmospheric fog can influence Dark Channel Prior and clarity indices.",
                "Optical flow measures apparent pixel velocity on the camera projection plane, not absolute meteorological wind velocity."
            ],
            limitations=[
                "YOLO detector was trained on daytime combustion scenes; nocturnal performance is reduced.",
                "Dark Channel Prior assumes non-sky pixels; large uniform white or gray surfaces may mimic haze.",
                "No causal link between visual smoke and ambient PM2.5 can be established without temporal air quality registration."
            ],
            success=True,
        )

    def _fail(self, video_path: str, msg: str) -> CVAnalysisSummary:
        logger.error("CV pipeline failed: %s", msg)
        return CVAnalysisSummary(
            video_path=video_path,
            annotated_video_path=None,
            timeline_csv_path=None,
            features_csv_path=None,
            trend_graph_path=None,
            total_frames=0,
            sampled_frames=0,
            sampling_stride=1,
            duration_seconds=0.0,
            fps=0.0,
            frame_width=0,
            frame_height=0,
            frames_with_smoke=0,
            frames_with_fire=0,
            pct_frames_with_smoke=0.0,
            pct_frames_with_fire=0.0,
            total_smoke_detections=0,
            total_fire_detections=0,
            avg_smoke_confidence=None,
            avg_fire_confidence=None,
            max_confidence=None,
            max_plume_coverage_pct=0.0,
            avg_plume_coverage_pct=0.0,
            plume_expansion_trend="negligible",
            dominant_plume_direction=None,
            avg_apparent_motion_speed=0.0,
            camera_motion_assessment="unknown",
            avg_dark_channel=0.0,
            avg_scene_clarity_index=0.0,
            haze_severity="unknown",
            primary_event_classification="uncertain_event",
            event_confidence_score=0.0,
            detection_timeline=[],
            confidence_uncertainty_notes=[],
            limitations=[],
            success=False,
            error_message=msg,
        )


def _cardinal_direction(deg: float) -> str:
    # 0 is East (+x), 90 is South (+y in image coords), 180 is West, 270 is North
    val = (deg % 360)
    if 337.5 <= val or val < 22.5:
        return "East"
    elif 22.5 <= val < 67.5:
        return "South-East"
    elif 67.5 <= val < 112.5:
        return "South"
    elif 112.5 <= val < 157.5:
        return "South-West"
    elif 157.5 <= val < 202.5:
        return "West"
    elif 202.5 <= val < 247.5:
        return "North-West"
    elif 247.5 <= val < 292.5:
        return "North"
    else:
        return "North-East"
