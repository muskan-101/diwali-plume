"""
app.py — Diwali Plume Stage A Flask API

Endpoints
---------
GET  /health                          Model/dataset availability check.
POST /analyze                         Submit a video + location + datetime.
GET  /results/<result_id>             Retrieve a saved analysis result.
GET  /outputs/<result_id>/<filename>  Serve annotated video or CSV safely.

Security
--------
- File extensions validated against an allowlist.
- Upload size capped (default 500 MB).
- Unique filenames generated server-side; user filenames are not trusted.
- Path traversal is prevented in /outputs by resolving and anchoring paths.
- Arbitrary filesystem paths are never exposed in responses; only result_id
  and filename components are surfaced.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so src.* imports work.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from flask import Flask, jsonify, request, send_from_directory, abort, render_template

from src.integration.integration_service import IntegrationService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths  (all relative to project root)
# ---------------------------------------------------------------------------

UPLOAD_DIR  = _ROOT / "data" / "uploads"
OUTPUT_DIR  = _ROOT / "data" / "outputs"
RESULTS_DIR = _ROOT / "data" / "results"
TEMPLATE_DIR = _ROOT / "src" / "templates"
STATIC_DIR = _ROOT / "src" / "static"
FRONTEND_DIST = _ROOT / "frontend" / "dist"

for _d in (UPLOAD_DIR, OUTPUT_DIR, RESULTS_DIR, TEMPLATE_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "assets").exists():
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIST / "assets"),
        static_url_path="/assets",
        template_folder=str(TEMPLATE_DIR),
    )
else:
    app = Flask(
        __name__,
        template_folder=str(TEMPLATE_DIR),
        static_folder=str(STATIC_DIR),
    )
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Lazy-initialised; shared across requests.
_service: IntegrationService | None = None


def _get_service() -> IntegrationService:
    global _service
    if _service is None:
        _service = IntegrationService(
            output_dir=OUTPUT_DIR,
            model_path=_ROOT / "models" / "vision" / "fire_smoke_yolov8n_dfire_best.pt",
            xgb_path=_ROOT / "models" / "xgboost_pm25_6h.pkl",
            features_path=_ROOT / "models" / "baseline_features.json",
            data_path=_ROOT / "data" / "processed" / "delhi_multimodal_clean.csv",
            max_fallback_km=10.0,
            vision_confidence=0.15,
        )
    return _service


# ---------------------------------------------------------------------------
# GET / (Web Dashboard)
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """Render the Diwali Plume React intelligence dashboard."""
    if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
        return send_from_directory(str(FRONTEND_DIST), "index.html")
    try:
        return render_template("dashboard.html")
    except Exception as exc:
        logger.exception("Failed to render dashboard")
        return jsonify({
            "status": "online",
            "message": "Diwali Plume Multimodal Intelligence API",
            "dashboard_error": str(exc),
        }), 200


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Return API and model availability status."""
    try:
        svc = _get_service()
        h   = svc.health()
    except Exception as exc:
        logger.exception("Health check failed")
        h = {
            "vision_model_available":   False,
            "forecast_model_available": False,
            "feature_schema_available": False,
            "dataset_available":        False,
            "cpcb_engine_available":    False,
            "episode_engine_available": False,
            "precaution_engine_available": False,
            "error":                    str(exc),
        }

    return jsonify({
        "status":  "ok",
        "api":     "Diwali Plume Multimodal Intelligence",
        "mode":    "historical_replay",
        **h,
    })


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accept multipart/form-data with:
      - video             : MP4/AVI/MOV file
      - latitude          : float
      - longitude         : float
      - recording_datetime: ISO-8601 string
      - mode              : "historical_replay" (only supported value)
    """
    # --- file ---
    if "video" not in request.files:
        return jsonify({"error": "Missing 'video' file field."}), 400

    file = request.files["video"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        }), 400

    # Generate a UUID-based filename — do NOT trust the user-supplied name.
    result_id    = str(uuid.uuid4())
    safe_filename = f"{result_id}{ext}"
    upload_path   = UPLOAD_DIR / safe_filename

    try:
        file.save(str(upload_path))
    except Exception as exc:
        logger.exception("Failed to save upload")
        return jsonify({"error": f"File save failed: {exc}"}), 500

    # --- form fields ---
    try:
        latitude  = float(request.form.get("latitude",  ""))
        longitude = float(request.form.get("longitude", ""))
    except (TypeError, ValueError):
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": "'latitude' and 'longitude' must be numeric."}), 400

    raw_dt = request.form.get("recording_datetime", "")
    if not raw_dt:
        upload_path.unlink(missing_ok=True)
        return jsonify({"error": "'recording_datetime' is required (ISO-8601)."}), 400

    try:
        recording_datetime = datetime.fromisoformat(raw_dt)
    except ValueError:
        upload_path.unlink(missing_ok=True)
        return jsonify({
            "error": f"Cannot parse 'recording_datetime': '{raw_dt}'. Use ISO-8601 format."
        }), 400

    mode = request.form.get("mode", "historical_replay")
    try:
        max_sampled_frames = int(request.form.get("max_sampled_frames", 250))
    except (TypeError, ValueError):
        max_sampled_frames = 250

    # --- run pipeline ---
    try:
        svc    = _get_service()
        result = svc.analyze(
            video_path=upload_path,
            latitude=latitude,
            longitude=longitude,
            recording_datetime=recording_datetime,
            mode=mode,
            result_id=result_id,
            max_sampled_frames=max_sampled_frames,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed")
        return jsonify({"error": f"Internal error: {exc}"}), 500

    # Persist result for /results/<id> lookup.
    _save_result(result_id, result)

    # Strip absolute filesystem paths from the response; expose only IDs.
    result = _sanitise_paths(result, result_id)

    return jsonify(result), 200


# ---------------------------------------------------------------------------
# GET /results/<result_id>
# ---------------------------------------------------------------------------

@app.route("/results/<result_id>", methods=["GET"])
def get_result(result_id: str):
    """Retrieve a previously generated analysis result."""
    # Sanitise result_id: allow only alphanumeric and hyphens.
    if not _is_safe_id(result_id):
        abort(400)

    result_file = RESULTS_DIR / f"{result_id}.json"
    if not result_file.exists():
        return jsonify({"error": f"Result '{result_id}' not found."}), 404

    try:
        with open(result_file, encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify(data), 200
    except Exception as exc:
        logger.exception("Failed to load result %s", result_id)
        return jsonify({"error": f"Could not load result: {exc}"}), 500


# ---------------------------------------------------------------------------
# GET /outputs/<result_id>/<filename>
# ---------------------------------------------------------------------------

@app.route("/outputs/<result_id>/<filename>", methods=["GET"])
def serve_output(result_id: str, filename: str):
    """
    Serve generated annotated videos and CSV files.
    Path traversal is prevented by anchoring to OUTPUT_DIR/<result_id>.
    """
    if not _is_safe_id(result_id) or not _is_safe_filename(filename):
        abort(400)

    # Resolve and anchor.
    result_subdir = (OUTPUT_DIR / result_id).resolve()
    if not result_subdir.is_dir():
        abort(404)

    target = (result_subdir / filename).resolve()
    if not str(target).startswith(str(result_subdir)):
        # Path traversal attempt.
        abort(403)

    if not target.exists():
        abort(404)

    return send_from_directory(str(result_subdir), filename)


# ---------------------------------------------------------------------------
# Client-side SPA routes (React Router support)
# ---------------------------------------------------------------------------

@app.route("/<path:path>", methods=["GET"])
def serve_spa_routes(path: str):
    """Fallback route for React Router client-side paths."""
    if any(path.startswith(prefix) for prefix in ("health", "analyze", "results", "outputs")):
        abort(404)
    if FRONTEND_DIST.exists():
        target = (FRONTEND_DIST / path).resolve()
        if target.exists() and not target.is_dir() and str(target).startswith(str(FRONTEND_DIST.resolve())):
            return send_from_directory(str(FRONTEND_DIST), path)
        return send_from_directory(str(FRONTEND_DIST), "index.html")
    abort(404)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request.", "detail": str(e)}), 400

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"error": "Forbidden."}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found.", "detail": str(e)}), 404

@app.errorhandler(413)
def request_too_large(e):
    return jsonify({"error": "Upload too large. Maximum is 500 MB."}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error.", "detail": str(e)}), 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_result(result_id: str, result: dict) -> None:
    try:
        path = RESULTS_DIR / f"{result_id}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not persist result %s: %s", result_id, exc)


def _sanitise_paths(result: dict, result_id: str) -> dict:
    """
    Replace absolute filesystem paths with safe API-relative URLs.
    E.g. 'data/outputs/<id>/annotated.mp4' -> '/outputs/<id>/annotated.mp4'
    """
    def _replace(obj):
        if isinstance(obj, dict):
            return {k: _replace(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_replace(i) for i in obj]
        if isinstance(obj, str):
            p = Path(obj)
            try:
                if result_id in str(p) and p.suffix.lower() in {".mp4", ".csv", ".png", ".jpg"}:
                    return f"/outputs/{result_id}/{p.name}"
                rel = p.resolve().relative_to((OUTPUT_DIR / result_id).resolve())
                return f"/outputs/{result_id}/{rel.name}"
            except (ValueError, Exception):
                pass
        return obj
    return _replace(result)


def _is_safe_id(value: str) -> bool:
    """Allow only UUID-shaped strings: hex + hyphens, 32–36 chars."""
    import re
    return bool(re.fullmatch(r"[0-9a-f\-]{32,36}", value))


def _is_safe_filename(value: str) -> bool:
    """Allow only simple filenames with extensions; no path separators."""
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]{1,120}\.[a-z0-9]{1,8}", value))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diwali Plume API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logger.info("Starting Diwali Plume API on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)
