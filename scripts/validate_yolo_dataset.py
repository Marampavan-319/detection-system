"""
Validate the local YOLO dataset before training.

Usage:
    python scripts/validate_yolo_dataset.py
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "yolo"
NUM_CLASSES = 12


def main():
    total = 0
    bad_images = 0
    bad_labels = 0
    missing_labels = 0

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = [
            p for p in image_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]

        for image in images:
            total += 1
            try:
                with Image.open(image) as img:
                    width, height = img.size
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid dimensions")
            except Exception as exc:
                bad_images += 1
                print(f"[BAD IMAGE] {image}: {exc}")
                continue

            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing_labels += 1
                continue

            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    bad_labels += 1
                    print(f"[BAD LABEL] {label}:{line_no} -> expected 5 values")
                    continue

                try:
                    cls = int(parts[0])
                    values = [float(v) for v in parts[1:]]
                except ValueError:
                    bad_labels += 1
                    print(f"[BAD LABEL] {label}:{line_no} -> non-numeric value")
                    continue

                if not 0 <= cls < NUM_CLASSES or any(v < 0 or v > 1 for v in values):
                    bad_labels += 1
                    print(f"[BAD LABEL] {label}:{line_no} -> class/coordinate out of range")

        print(f"{split}: {len(images)} images")

    print("\nValidation summary")
    print(f"Images checked: {total}")
    print(f"Bad images: {bad_images}")
    print(f"Bad label rows: {bad_labels}")
    print(f"Images without labels: {missing_labels}")

    if bad_images or bad_labels:
        raise SystemExit(1)

    print("Dataset structure is valid for YOLO.")
    

if __name__ == "__main__":
    main()
