"""Evidence-based multimodal reasoning for ElectroDiagnose.

Session 7 combines visual detections/localizations, OCR/error text, and optional
user symptoms into a structured evidence summary. It does not claim to prove
an internal electrical fault; conclusions are limited to the supplied evidence.
"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ReasoningResult:
    device: str
    summary: str
    evidence: list[str]
    observations: list[str]
    possible_causes: list[str]
    next_actions: list[str]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "summary": self.summary,
            "evidence": list(self.evidence),
            "observations": list(self.observations),
            "possible_causes": list(self.possible_causes),
            "next_actions": list(self.next_actions),
            "limitations": list(self.limitations),
        }


class EvidenceReasoner:
    """Combine structured visual, OCR, and symptom evidence deterministically."""

    def reason(
        self,
        device: str,
        locations: Iterable[dict[str, Any]],
        ocr_text: str = "",
        symptoms: str = "",
    ) -> ReasoningResult:
        category = device.strip().lower()
        location_list = list(locations)
        evidence: list[str] = []
        observations: list[str] = []
        possible_causes: list[str] = []
        next_actions: list[str] = []
        limitations = [
            "This is a visual/evidence-based assessment, not proof of an internal electrical fault.",
            "Physical inspection and electrical measurements may be required before repair.",
        ]

        if location_list:
            for item in location_list:
                defect = str(item.get("defect_class", "unknown defect"))
                component = str(item.get("component", "unknown component"))
                subsystem = str(item.get("subsystem", "unknown subsystem"))
                region = str(item.get("region", "unknown region"))
                confidence = float(item.get("confidence", 0.0))
                evidence.append(
                    f"Visual detection: {defect} at {component} ({region}), "
                    f"confidence {confidence:.2f}."
                )
                observations.append(
                    f"{defect} is visibly localized to the {component} in the "
                    f"{subsystem}."
                )
            evidence_count = len(location_list)
            summary = (
                f"{evidence_count} visible defect finding"
                f"{'s' if evidence_count != 1 else ''} identified for {category or 'device'}."
            )
        else:
            summary = f"No visual defect localization was supplied for {category or 'device'}."
            evidence.append("No structured visual detections were provided.")

        clean_ocr = " ".join(ocr_text.split())
        if clean_ocr:
            evidence.append(f"OCR/error evidence: {clean_ocr[:500]}")
            observations.append("The supplied OCR/error text should be compared with the visible findings.")

        clean_symptoms = " ".join(symptoms.split())
        if clean_symptoms:
            evidence.append(f"User symptom: {clean_symptoms[:500]}")
            observations.append("The reported symptom is contextual evidence and is not independently verified.")

        if location_list:
            defect_classes = {str(item.get("defect_class", "")).lower() for item in location_list}
            if category == "pcb" and "short" in defect_classes:
                possible_causes.append("Possible PCB interconnect short or conductive bridge near the localized region.")
                next_actions.append("Inspect the localized PCB region under magnification and verify continuity before repair.")
            elif category == "laptop" and any(
                str(item.get("defect_class", "")).lower() in {"screen", "hinge", "keyboard", "port", "chassis"}
                for item in location_list
            ):
                possible_causes.append("Visible physical damage or wear affecting the identified laptop component.")
                next_actions.append("Inspect the localized component for physical damage, loose connections, or mechanical obstruction.")
            elif category == "smartphone":
                possible_causes.append("Visible damage or abnormal condition affecting the identified smartphone component.")
                next_actions.append("Inspect the localized component and compare with the reported symptom before repair.")
            elif category == "router":
                possible_causes.append("Visible component state or connection condition affecting the identified router area.")
                next_actions.append("Check the localized indicator/port and verify the corresponding network or power condition.")
            else:
                possible_causes.append("The visible finding may have multiple underlying causes; additional inspection is needed.")
                next_actions.append("Inspect the localized region and collect additional evidence before repair.")

        if clean_ocr:
            next_actions.append("Compare OCR/error text with the physical finding and reproduce the reported error if safe.")
        if clean_symptoms:
            next_actions.append("Use the reported symptom as supporting context and verify it independently.")

        if not next_actions:
            next_actions.append("Provide a clear device image or error screenshot for further analysis.")

        return ReasoningResult(
            device=category,
            summary=summary,
            evidence=evidence,
            observations=observations,
            possible_causes=possible_causes,
            next_actions=next_actions,
            limitations=limitations,
        )


def reason_about_evidence(
    device: str,
    locations: Iterable[dict[str, Any]],
    ocr_text: str = "",
    symptoms: str = "",
) -> dict[str, Any]:
    """JSON-friendly convenience API."""
    return EvidenceReasoner().reason(
        device=device,
        locations=locations,
        ocr_text=ocr_text,
        symptoms=symptoms,
    ).to_dict()
