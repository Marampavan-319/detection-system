"""YOLO-based computer-vision detection for ElectroDiagnose.

Session 4 adds a testable wrapper around Ultralytics YOLO with lazy model loading.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Detection:
    """One detected object in image coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


class YOLODetector:
    """Lazy-loading YOLO detector with a stable project-level output schema."""

    def __init__(
        self,
        model_path: str | Path = "yolo11n.pt",
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        self.model_path = str(model_path)
        self.confidence = confidence
        self.iou = iou
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required. Install with pip install -r requirements.txt."
                ) from exc
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: str | Path | Image.Image) -> list[Detection]:
        """Run inference and return project detections."""
        if isinstance(image, (str, Path)):
            source: Any = str(image)
        elif isinstance(image, Image.Image):
            source = image
        else:
            raise TypeError("image must be a file path or PIL.Image.Image")

        results = self._load_model().predict(
            source=source,
            conf=self.confidence,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=str(result.names[class_id]),
                        confidence=confidence,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections


def detect_objects(
    image: str | Path | Image.Image,
    model_path: str | Path = "yolo11n.pt",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Convenience API returning JSON-friendly detection dictionaries."""
    detector = YOLODetector(model_path, confidence, iou, device)
    return [d.to_dict() for d in detector.detect(image)]
