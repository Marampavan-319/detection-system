"""Severity and confidence scoring for ElectroDiagnose.

Session 8 converts detector evidence into explainable confidence and severity
scores. The scores describe the supplied evidence and visible risk signals;
they do not prove an internal electrical fault.
"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SeverityResult:
    confidence: float
    severity: str
    rationale: list[str]
    priority: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "severity": self.severity,
            "rationale": list(self.rationale),
            "priority": self.priority,
        }


class SeverityScorer:
    """Produce deterministic, explainable confidence and severity values."""

    def score(
        self,
        locations: Iterable[dict[str, Any]],
        reasoning: dict[str, Any] | None = None,
    ) -> SeverityResult:
        items = list(locations)
        if not items:
            return SeverityResult(
                confidence=0.0,
                severity="UNKNOWN",
                rationale=["No localized visual evidence was supplied."],
                priority="LOW",
            )

        confidences = [
            max(0.0, min(1.0, float(item.get("confidence", 0.0))))
            for item in items
        ]
        base_confidence = sum(confidences) / len(confidences)
        rationale: list[str] = [
            f"Average visual detection confidence is {base_confidence:.2f}."
        ]

        clean_reasoning = reasoning or {}
        if len(clean_reasoning.get("evidence", [])):
            confidence = min(1.0, base_confidence + 0.02)
            rationale.append("Additional structured evidence supports the visual finding.")
        else:
            confidence = base_confidence

        defect_classes = {
            str(item.get("defect_class", "")).strip().lower() for item in items
        }
        device = str(items[0].get("device", "")).strip().lower()

        high_risk = {"short", "battery", "burn", "smoke", "fire", "spark"}
        medium_risk = {
            "open", "spur", "pin-hole", "screen", "hinge", "port",
            "keyboard", "chassis", "copper", "stain", "scratch", "mousebite",
        }

        if defect_classes & high_risk:
            severity = "HIGH"
            priority = "HIGH"
            rationale.append("The finding includes a defect class associated with elevated inspection risk.")
        elif defect_classes & medium_risk:
            severity = "MEDIUM"
            priority = "MEDIUM"
            rationale.append("The finding indicates a visible defect requiring inspection.")
        else:
            severity = "LOW"
            priority = "LOW"
            rationale.append("No elevated-risk defect class was supplied.")

        if device == "router" and "lan not connected" in defect_classes:
            severity = "LOW"
            priority = "MEDIUM"
            rationale.append("A LAN-not-connected state is treated as a connection check rather than a confirmed hardware fault.")

        if device == "router" and "led off" in defect_classes:
            severity = "MEDIUM"
            priority = "MEDIUM"
            rationale.append("An inactive status LED warrants checking power and device state.")

        return SeverityResult(
            confidence=round(confidence, 3),
            severity=severity,
            rationale=rationale,
            priority=priority,
        )


def score_severity(
    locations: Iterable[dict[str, Any]],
    reasoning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-friendly convenience API."""
    return SeverityScorer().score(locations, reasoning).to_dict()
