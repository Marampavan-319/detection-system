"""Smoke-test the Session 4 YOLO detection pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection import YOLODetector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"Image not found: {args.image}")
        return 1

    detector = YOLODetector(args.model, args.confidence, args.iou)
    try:
        detections = detector.detect(args.image)
    except Exception as exc:
        print(f"Detection validation failed: {exc}")
        return 1

    print(f"Model: {args.model}")
    print(f"Detections: {len(detections)}")
    for d in detections:
        print(
            f"- {d.class_name}: {d.confidence:.3f} "
            f"bbox={tuple(round(v, 1) for v in d.bbox)}"
        )

    print("Session 4 YOLO smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
