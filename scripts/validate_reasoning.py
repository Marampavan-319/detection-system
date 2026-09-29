"""Deterministic smoke test for Session 7 reasoning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reasoning import EvidenceReasoner


def main() -> None:
    locations = [
        {
            "device": "pcb",
            "component": "shorted region",
            "subsystem": "PCB interconnect",
            "defect_class": "short",
            "confidence": 0.91,
            "region": "middle-center",
        },
        {
            "device": "pcb",
            "component": "copper region",
            "subsystem": "PCB interconnect",
            "defect_class": "copper",
            "confidence": 0.84,
            "region": "middle-right",
        },
    ]

    result = EvidenceReasoner().reason(
        device="PCB",
        locations=locations,
        ocr_text="ERROR: continuity check failed",
        symptoms="Board fails electrical test.",
    )

    assert result.device == "pcb"
    assert len(result.evidence) == 4
    assert "shorted region" in result.observations[0]
    assert any("PCB interconnect short" in item for item in result.possible_causes)
    assert any("magnification" in item for item in result.next_actions)
    assert any("OCR/error evidence" in item for item in result.evidence)
    assert any("User symptom" in item for item in result.evidence)
    assert result.limitations

    empty = EvidenceReasoner().reason("router", [])
    assert empty.summary.startswith("No visual defect localization")
    assert empty.next_actions

    print("Multimodal reasoning validated.")
    print(f"Evidence items: {len(result.evidence)}")
    print(f"Observations: {len(result.observations)}")
    print(f"Possible causes: {len(result.possible_causes)}")
    print(f"Next actions: {len(result.next_actions)}")
    print("Session 7 multimodal reasoning validation passed.")


if __name__ == "__main__":
    main()
