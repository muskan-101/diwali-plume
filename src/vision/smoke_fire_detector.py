
import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = "models/vision/fire_smoke_yolov8n_dfire_best.pt"
OUTPUT_DIR = Path("runs/vision/smoke_fire")


def analyze_video(video_path, confidence=0.25):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video = OUTPUT_DIR / f"{video_path.stem}_annotated.mp4"
    output_csv = OUTPUT_DIR / f"{video_path.stem}_timeline.csv"

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    counts = Counter()
    rows = []
    frame_number = 0
    total_detections = 0

    print(f"\nAnalyzing: {video_path.name}")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: CPU\n")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            timestamp = (frame_number - 1) / fps

            result = model.predict(
                source=frame,
                conf=confidence,
                device="cpu",
                verbose=False
            )[0]

            annotated = result.plot()

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    score = float(box.conf.item())
                    label = model.names[class_id]

                    counts[label] += 1
                    total_detections += 1

                    rows.append({
                        "frame": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "class": label,
                        "confidence": round(score, 4)
                    })

            writer.write(annotated)

            if frame_number % 30 == 0:
                print(f"Processed {frame_number} frames")

    finally:
        cap.release()
        writer.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        columns = [
            "frame",
            "timestamp_seconds",
            "class",
            "confidence"
        ]
        csv_writer = csv.DictWriter(file, fieldnames=columns)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    print("\n" + "=" * 50)
    print("SMOKE/FIRE DETECTION SUMMARY")
    print("=" * 50)
    print("Frames processed:", frame_number)
    print("Total detections:", total_detections)

    for label in model.names.values():
        print(f"{label}: {counts[label]}")

    print("\nAnnotated video:", output_video)
    print("Detection timeline:", output_csv)
    print("Experiment completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--conf", type=float, default=0.25)

    args = parser.parse_args()
    analyze_video(args.source, args.conf)
