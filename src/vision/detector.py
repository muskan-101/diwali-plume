
import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


# --------------------------------------------------
# DIWALI PLUME - FIRST CV EXPERIMENT
# Pretrained YOLOv8n object detection
# --------------------------------------------------

MODEL_PATH = "yolov8n.pt"

OUTPUT_DIR = "runs/vision"

def run_detection(source):

    print("\n" + "=" * 55)
    print("DIWALI PLUME - COMPUTER VISION")
    print("=" * 55)

    print("\nLoading pretrained YOLOv8n model...")

    model = YOLO(MODEL_PATH)

    print("Model loaded successfully.")
    print("Device: CPU")
    print("Input:", source)

    print("\nStarting inference...")

    class_counts = Counter()
    frame_count = 0
    detection_count = 0

    results = model.predict(
        source=source,
        device="cpu",
        imgsz=640,
        conf=0.25,
        save=True,
        stream=True,
        project=OUTPUT_DIR,
        name="baseline",
        exist_ok=True,
        verbose=False
    )

    for result in results:

        frame_count += 1

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            class_name = model.names[class_id]

            class_counts[class_name] += 1
            detection_count += 1

            print(
                f"Frame {frame_count}: "
                f"{class_name} "
                f"(confidence: {confidence:.2f})"
            )

    print("\n" + "=" * 55)
    print("DETECTION SUMMARY")
    print("=" * 55)

    print("Frames processed:", frame_count)
    print("Total detections:", detection_count)

    if class_counts:
        print("\nDetected objects:")

        for name, count in class_counts.most_common():
            print(f"{name}: {count}")
    else:
        print("No objects detected.")

    print("\nAnnotated output saved under:")
    print(Path(OUTPUT_DIR) / "baseline")

    print("\nExperiment completed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Diwali Plume video/image detector"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image or video"
    )

    args = parser.parse_args()

    if not Path(args.source).exists():
        raise FileNotFoundError(
            f"Input file not found: {args.source}"
        )

    run_detection(args.source)
