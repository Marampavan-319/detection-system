"""Image preprocessing utilities for ElectroDiagnose.

Session 3 implements the deterministic image-processing stage used before
computer-vision inference:

1. Load and validate the image.
2. Apply EXIF orientation.
3. Convert to RGB.
4. Letterbox-resize to the model input size while preserving aspect ratio.
5. Convert pixels to float32 in the [0, 1] range for model-ready tensors.

The saved preview image remains uint8. Normalization is applied only to the
returned NumPy array so source image quality is not unnecessarily changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class PreprocessResult:
    """Result of deterministic image preprocessing."""

    image: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    scale: float
    pad: tuple[int, int, int, int]

    @property
    def tensor(self) -> np.ndarray:
        """Return CHW float32 data in the [0, 1] range."""
        return np.transpose(self.image, (2, 0, 1)).astype(np.float32) / 255.0


def load_image(path: str | Path) -> Image.Image:
    """Load an image, apply EXIF orientation, and convert it to RGB."""
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to read image: {image_path}") from exc


def letterbox(
    image: Image.Image,
    size: int | tuple[int, int] = 640,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, float, tuple[int, int, int, int]]:
    """Resize while preserving aspect ratio and pad to the requested size."""
    if isinstance(size, int):
        target_w = target_h = size
    else:
        target_w, target_h = size

    if target_w <= 0 or target_h <= 0:
        raise ValueError("Target dimensions must be positive")

    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)

    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2
    canvas.paste(resized, (left, top))

    right = target_w - new_w - left
    bottom = target_h - new_h - top
    return canvas, scale, (left, top, right, bottom)


def preprocess_image(
    path: str | Path,
    size: int | tuple[int, int] = 640,
) -> PreprocessResult:
    """Return a validated, letterboxed, normalized-ready image."""
    image = load_image(path)
    original_size = image.size
    processed, scale, pad = letterbox(image, size=size)
    array = np.asarray(processed, dtype=np.uint8)

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Preprocessed image must have shape HxWx3")

    return PreprocessResult(
        image=array,
        original_size=original_size,
        processed_size=(processed.width, processed.height),
        scale=scale,
        pad=pad,
    )


def save_processed(
    path: str | Path,
    output_path: str | Path,
    size: int | tuple[int, int] = 640,
) -> PreprocessResult:
    """Preprocess an image and save the uint8 RGB result."""
    result = preprocess_image(path, size=size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image, mode="RGB").save(output)
    return result
