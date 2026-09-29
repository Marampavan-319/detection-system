"""Defect localization utilities.

Session 6 converts component matches into structured, image-space defect
locations. It keeps localization deterministic and reports only what is
supported by the visible bounding box.
"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DefectLocation:
    device: str
    component: str
    subsystem: str
    defect_class: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    area_ratio: float
    normalized_bbox: tuple[float, float, float, float]
    region: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "component": self.component,
            "subsystem": self.subsystem,
            "defect_class": self.defect_class,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "center": list(self.center),
            "area_ratio": self.area_ratio,
            "normalized_bbox": list(self.normalized_bbox),
            "region": self.region,
        }


def _region_from_center(cx: float, cy: float) -> str:
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    vertical = "top" if cy < 1 / 3 else "bottom" if cy > 2 / 3 else "middle"
    return f"{vertical}-{horizontal}"


class DefectLocalizer:
    """Create normalized localization records from component matches."""

    def localize(
        self,
        matches: Iterable[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> list[DefectLocation]:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        locations: list[DefectLocation] = []

        for match in matches:
            x1, y1, x2, y2 = (
                float(v) for v in match.get("bbox", (0, 0, 0, 0))
            )

            x1 = max(0.0, min(x1, float(image_width)))
            x2 = max(0.0, min(x2, float(image_width)))
            y1 = max(0.0, min(y1, float(image_height)))
            y2 = max(0.0, min(y2, float(image_height)))

            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            width = x2 - x1
            height = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            locations.append(
                DefectLocation(
                    device=str(match.get("device", "")).strip().lower(),
                    component=str(match.get("component", "unknown component")),
                    subsystem=str(match.get("subsystem", "unknown subsystem")),
                    defect_class=str(match.get("detector_class", "unknown defect")),
                    confidence=float(match.get("confidence", 0.0)),
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    area_ratio=(width * height) / (image_width * image_height),
                    normalized_bbox=(
                        x1 / image_width,
                        y1 / image_height,
                        x2 / image_width,
                        y2 / image_height,
                    ),
                    region=_region_from_center(
                        cx / image_width,
                        cy / image_height,
                    ),
                )
            )

        return locations


def localize_defects(
    matches: Iterable[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    """JSON-friendly convenience API."""
    localizer = DefectLocalizer()
    return [
        location.to_dict()
        for location in localizer.localize(matches, image_width, image_height)
    ]
