"""
ElectroDiagnose - local dataset preparation utility.

This script does NOT download anything and does NOT depend on another
GitHub repository. It works only with files already placed under
datasets/raw/.

Usage from the repository root:
    python scripts/prepare_dataset.py

Optional:
    python scripts/prepare_dataset.py --source datasets/raw/pcb --device pcb
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "datasets" / "raw"
YOLO_ROOT = ROOT / "datasets" / "yolo"
SUMMARY_FILE = ROOT / "datasets" / "processed" / "dataset_summary.csv"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local images for YOLO training.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--copy-existing-labels",
        action="store_true",
        help="Copy a matching YOLO .txt file when it exists beside an image.",
    )
    return parser.parse_args()


def image_key(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def discover_images(source: Path, device: str | None):
    roots = [source / device] if device else [p for p in source.iterdir() if p.is_dir()]
    images = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(path)
    return sorted(images)


def is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def split_items(items):
    items = list(items)
    random.Random(SEED).shuffle(items)
    n = len(items)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])
    return {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }


def ensure_dirs():
    for split in SPLIT_RATIOS:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    source = args.source if args.source.is_absolute() else ROOT / args.source
    ensure_dirs()

    images = discover_images(source, args.device)
    readable = [p for p in images if is_readable(p)]

    if not readable:
        print(f"No readable images found in: {source}")
        print("Place images under datasets/raw/<device>/ and run again.")
        return

    groups = split_items(readable)
    summary_rows = []

    for split, paths in groups.items():
        for src in paths:
            device = src.parent.name.lower()
            new_stem = image_key(src)
            destination = YOLO_ROOT / "images" / split / f"{new_stem}{src.suffix.lower()}"
            shutil.copy2(src, destination)

            label_src = src.with_suffix(".txt")
            label_dst = YOLO_ROOT / "labels" / split / f"{new_stem}.txt"
            label_status = "missing"

            if args.copy_existing_labels and label_src.exists():
                shutil.copy2(label_src, label_dst)
                label_status = "copied"

            summary_rows.append({
                "device": device,
                "source_file": str(src.relative_to(ROOT)),
                "split": split,
                "output_file": str(destination.relative_to(ROOT)),
                "label_status": label_status,
            })

    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Dataset preparation complete.")
    for split, paths in groups.items():
        print(f"  {split}: {len(paths)} images")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
