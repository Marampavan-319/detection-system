"""Final diagnostic result engine for ElectroDiagnose.

Session 9 combines localization, reasoning, and severity into one stable,
JSON-friendly diagnostic response. The result describes supplied evidence and
does not certify an internal electrical fault.
"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class DiagnosticResult:
    status: str
    device: str
    findings: list[dict[str, Any]]
    confidence: float
    severity: str
    priority: str
    summary: str
    possible_causes: list[str]
    next_actions: list[str]
    limitations: list[str]
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "device": self.device,
            "findings": [dict(item) for item in self.findings],
            "confidence": self.confidence,
            "severity": self.severity,
            "priority": self.priority,
            "summary": self.summary,
            "possible_causes": list(self.possible_causes),
            "next_actions": list(self.next_actions),
            "limitations": list(self.limitations),
            "review_required": self.review_required,
        }


class DiagnosticEngine:
    """Assemble a consistent final diagnostic response from prior pipeline stages."""

    def build(
        self,
        device: str,
        locations: Iterable[dict[str, Any]],
        reasoning: dict[str, Any] | None,
        severity: dict[str, Any] | None,
    ) -> DiagnosticResult:
        category = device.strip().lower()
        location_list = list(locations)
        reasoning_data = reasoning or {}
        severity_data = severity or {}

        findings = [
            {
                "defect": str(item.get("defect_class", "unknown defect")),
                "component": str(item.get("component", "unknown component")),
                "subsystem": str(item.get("subsystem", "unknown subsystem")),
                "region": str(item.get("region", "unknown region")),
                "confidence": round(float(item.get("confidence", 0.0)), 3),
                "bbox": list(item.get("bbox", [])),
            }
            for item in location_list
        ]

        if location_list:
            status = "DEFECT_DETECTED"
            summary = str(reasoning_data.get("summary") or self._fallback_summary(category, findings))
        else:
            status = "INSUFFICIENT_EVIDENCE"
            summary = "No visible defect was localized; additional evidence is required."

        confidence = max(0.0, min(1.0, float(severity_data.get("confidence", 0.0))))
        severity_name = str(severity_data.get("severity", "UNKNOWN"))
        priority = str(severity_data.get("priority", "LOW"))

        possible_causes = list(reasoning_data.get("possible_causes", []))
        next_actions = list(reasoning_data.get("next_actions", []))
        limitations = list(reasoning_data.get(
            "limitations",
            [
                "This is a visual/evidence-based assessment, not proof of an internal electrical fault.",
                "Physical inspection and electrical measurements may be required before repair.",
            ],
        ))

        if not next_actions:
            next_actions = ["Provide clearer visual evidence or an error screenshot for further analysis."]

        review_required = (
            status != "DEFECT_DETECTED"
            or confidence < 0.60
            or severity_name == "HIGH"
        )

        return DiagnosticResult(
            status=status,
            device=category,
            findings=findings,
            confidence=round(confidence, 3),
            severity=severity_name,
            priority=priority,
            summary=summary,
            possible_causes=possible_causes,
            next_actions=next_actions,
            limitations=limitations,
            review_required=review_required,
        )

    @staticmethod
    def _fallback_summary(device: str, findings: list[dict[str, Any]]) -> str:
        count = len(findings)
        noun = "finding" if count == 1 else "findings"
        return f"{count} visible defect {noun} identified for {device or 'device'}."


def diagnose(
    device: str,
    locations: Iterable[dict[str, Any]],
    reasoning: dict[str, Any] | None = None,
    severity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-friendly convenience API for the final diagnostic response."""
    return DiagnosticEngine().build(device, locations, reasoning, severity).to_dict()
