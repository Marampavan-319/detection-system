"""Component and subsystem identification from YOLO detections.

Session 5 converts generic detector class names into device-aware component
and subsystem labels. It is deliberately rule-based so it remains deterministic
until category-specific trained models are available.
"""

from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_COMPONENT_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "pcb": {
        "open": ("circuit trace", "PCB interconnect"),
        "short": ("shorted region", "PCB interconnect"),
        "mousebite": ("board edge", "PCB fabrication"),
        "spur": ("copper spur", "PCB interconnect"),
        "copper": ("copper region", "PCB interconnect"),
        "pin-hole": ("pin-hole region", "PCB fabrication"),
    },
    "laptop": {
        "screen": ("display", "display subsystem"),
        "keyboard": ("keyboard", "input subsystem"),
        "hinge": ("hinge", "mechanical subsystem"),
        "port": ("I/O port", "I/O subsystem"),
        "chassis": ("chassis", "mechanical subsystem"),
    },
    "smartphone": {
        "screen": ("display", "display subsystem"),
        "screen crack": ("display", "display subsystem"),
        "scratch": ("display", "display subsystem"),
        "dead pixel": ("display", "display subsystem"),
        "camera": ("camera module", "camera subsystem"),
        "port": ("charging port", "power/I-O subsystem"),
        "battery": ("battery", "power subsystem"),
    },
    "router": {
        "lan connected": ("LAN port", "network interface"),
        "lan not connected": ("LAN port", "network interface"),
        "led on": ("status LED", "indicator subsystem"),
        "led off": ("status LED", "indicator subsystem"),
    },
}


@dataclass(frozen=True)
class ComponentMatch:
    device: str
    detector_class: str
    component: str
    subsystem: str
    confidence: float
    bbox: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "detector_class": self.detector_class,
            "component": self.component,
            "subsystem": self.subsystem,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
        }


class ComponentIdentifier:
    """Map detector outputs to components using an explicit category map."""

    def __init__(
        self,
        mappings: dict[str, dict[str, tuple[str, str]]] | None = None,
    ) -> None:
        self.mappings = mappings or DEFAULT_COMPONENT_MAP

    def identify(
        self,
        device: str,
        detections: Iterable[dict[str, Any]],
    ) -> list[ComponentMatch]:
        category = device.strip().lower()
        category_map = self.mappings.get(category, {})
        matches: list[ComponentMatch] = []

        for detection in detections:
            detector_class = str(detection.get("class_name", "")).strip()
            key = detector_class.lower()
            mapped = category_map.get(key)
            if mapped is None:
                component = detector_class or "unknown component"
                subsystem = "unknown subsystem"
            else:
                component, subsystem = mapped

            bbox = tuple(float(v) for v in detection.get("bbox", (0, 0, 0, 0)))
            confidence = float(detection.get("confidence", 0.0))
            matches.append(
                ComponentMatch(
                    device=category,
                    detector_class=detector_class,
                    component=component,
                    subsystem=subsystem,
                    confidence=confidence,
                    bbox=bbox,
                )
            )

        return matches


def identify_components(
    device: str,
    detections: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """JSON-friendly convenience API."""
    identifier = ComponentIdentifier()
    return [match.to_dict() for match in identifier.identify(device, detections)]
