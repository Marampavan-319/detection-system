"""Validation smoke test for Session 9 diagnostic result engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diagnosis import DiagnosticEngine


def main() -> None:
    locations = [
        {
            "device": "pcb",
            "component": "shorted region",
            "subsystem": "PCB interconnect",
            "defect_class": "short",
            "confidence": 0.91,
            "bbox": [350.0, 350.0, 650.0, 650.0],
            "region": "middle-center",
        }
    ]

    reasoning = {
        "summary": "1 visible defect finding identified for pcb.",
        "possible_causes": [
            "Possible PCB interconnect short or conductive bridge near the localized region."
        ],
        "next_actions": [
            "Inspect the localized PCB region under magnification and verify continuity before repair."
        ],
        "limitations": [
            "This is a visual/evidence-based assessment, not proof of an internal electrical fault."
        ],
        "evidence": ["Visual detection: short at shorted region (middle-center), confidence 0.91."],
    }

    severity = {
        "confidence": 0.93,
        "severity": "HIGH",
        "priority": "HIGH",
    }

    result = DiagnosticEngine().build("PCB", locations, reasoning, severity).to_dict()

    assert result["status"] == "DEFECT_DETECTED"
    assert result["device"] == "pcb"
    assert result["findings"][0]["defect"] == "short"
    assert result["findings"][0]["component"] == "shorted region"
    assert result["findings"][0]["region"] == "middle-center"
    assert result["confidence"] == 0.93
    assert result["severity"] == "HIGH"
    assert result["priority"] == "HIGH"
    assert result["review_required"] is True
    assert result["possible_causes"]
    assert result["next_actions"]
    assert result["limitations"]

    empty = DiagnosticEngine().build(
        "laptop",
        [],
        {"evidence": []},
        {"confidence": 0.0, "severity": "UNKNOWN", "priority": "LOW"},
    ).to_dict()

    assert empty["status"] == "INSUFFICIENT_EVIDENCE"
    assert empty["confidence"] == 0.0
    assert empty["review_required"] is True

    print("Diagnostic result validated.")
    print(
        f"Defect case: status={result['status']}, confidence={result['confidence']:.2f}, "
        f"severity={result['severity']}, priority={result['priority']}"
    )
    print(
        f"Empty case: status={empty['status']}, confidence={empty['confidence']:.2f}, "
        f"severity={empty['severity']}"
    )
    print("Session 9 diagnostic result validation passed.")


if __name__ == "__main__":
    main()
