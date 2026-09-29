"""
Convert a locally downloaded DeepPCB dataset to this project's YOLO format.

No network access is used. No other GitHub repository is required.

DeepPCB annotations in this Kaggle copy use:
    x1 y1 x2 y2 type

DeepPCB type IDs:
    1 open
    2 short
    3 mousebite
    4 spur
    5 spurious copper
    6 pin-hole

Project YOLO IDs:
    6 open_circuit
    7 short_circuit
    8 mousebite
    9 spur
    11 spurious_copper
    10 pin_hole
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "yolo"

TYPE_MAP = {
    1: 6,
    2: 7,
    3: 8,
    4: 9,
    5: 11,
    6: 10,
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def find_files(source: Path):
    images = sorted(
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    labels = {}
    for p in source.rglob("*.txt"):
        labels.setdefault(p.stem.lower(), []).append(p)
    return images, labels


def split(items, seed):
    items = list(items)
    random.Random(seed).shuffle(items)
    n = len(items)
    a = int(n * 0.8)
    b = a + int(n * 0.1)
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def find_label(image: Path, labels_by_stem):
    matches = labels_by_stem.get(image.stem.lower(), [])
    if not matches:
        return None

    for candidate in matches:
        if candidate.parent == image.parent:
            return candidate

    # Prefer a labels directory with the same relative dataset branch.
    for candidate in matches:
        if candidate.parent.name.lower() == "labels":
            return candidate

    return matches[0]


def convert_box(line, width, height):
    # Kaggle DeepPCB annotation files use whitespace-separated values:
    # x1 y1 x2 y2 class_id
    parts = line.strip().replace(",", " ").split()
    if len(parts) != 5:
        return None

    try:
        x1, y1, x2, y2, cls = map(int, parts)
    except ValueError:
        return None

    if cls not in TYPE_MAP:
        return None

    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height

    return f"{TYPE_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main():
    a = args()
    source = a.source if a.source.is_absolute() else ROOT / a.source

    images, labels_by_stem = find_files(source)

    print(f"Images discovered: {len(images)}")
    print(f"Annotation files discovered: {sum(len(v) for v in labels_by_stem.values())}")

    if not images:
        print(f"No supported images found under {source}")
        return

    labeled_images = []
    for image in images:
        label = find_label(image, labels_by_stem)
        if label:
            labeled_images.append((image, label))

    print(f"Images with matching annotations: {len(labeled_images)}")

    if not labeled_images:
        print("No image/annotation pairs were found.")
        return

    splits = split(labeled_images, a.seed)

    for split_name, items in splits.items():
        image_dir = OUT / "images" / split_name
        label_dir = OUT / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        skipped = 0

        for src, label_src in items:
            try:
                with Image.open(src) as img:
                    width, height = img.size
            except Exception:
                skipped += 1
                continue

            labels = []
            for line in label_src.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                result = convert_box(line, width, height)
                if result:
                    labels.append(result)

            if not labels:
                skipped += 1
                continue

            relative = src.relative_to(source)
            safe_stem = "_".join(relative.with_suffix("").parts)

            destination_image = image_dir / f"{safe_stem}{src.suffix.lower()}"
            destination_label = label_dir / f"{safe_stem}.txt"

            shutil.copy2(src, destination_image)
            destination_label.write_text(
                "\n".join(labels) + "\n", encoding="utf-8"
            )
            converted += 1

        print(f"{split_name}: {converted} images converted, {skipped} skipped")

    print("DeepPCB conversion complete.")
    print(f"YOLO dataset: {OUT}")


if __name__ == "__main__":
    main()
