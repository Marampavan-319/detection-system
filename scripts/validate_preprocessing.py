"""Validate the Session 3 image preprocessing pipeline.

Usage:
    python scripts/validate_preprocessing.py path/to/image.jpg
    python scripts/validate_preprocessing.py --source datasets/yolo/images/test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import preprocess_image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate image preprocessing.")
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--size", type=int, default=640)
    return parser.parse_args()


def iter_images(source: Path):
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def main():
    args = parse_args()
    if args.image is None and args.source is None:
        raise SystemExit("Provide an image path or --source directory.")

    source = args.image or args.source
    if not source.exists():
        raise SystemExit(f"Path does not exist: {source}")

    checked = 0
    failed = 0

    for path in iter_images(source):
        try:
            result = preprocess_image(path, size=args.size)
            if result.image.shape != (args.size, args.size, 3):
                raise ValueError(f"unexpected shape: {result.image.shape}")
            if not (0.0 <= float(result.tensor.min()) <= 1.0):
                raise ValueError("normalized minimum is outside [0, 1]")
            if not (0.0 <= float(result.tensor.max()) <= 1.0):
                raise ValueError("normalized maximum is outside [0, 1]")
            checked += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path}: {exc}")

    print(f"Images checked: {checked}")
    print(f"Images failed: {failed}")

    if failed:
        raise SystemExit(1)

    print("Session 3 preprocessing validation passed.")


if __name__ == "__main__":
    main()
