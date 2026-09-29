"""Smoke test for the Session 10 application analysis pipeline."""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import analyze_evidence, _demo_detections


def main() -> None:
    image = Image.new("RGB", (1000, 1000), "white")
    detections = _demo_detections("PCB", image)

    result = analyze_evidence(
        device="PCB",
        image=image,
        detections=detections,
        ocr_text="ERROR: continuity check failed",
        symptoms="Board fails electrical test.",
    )

    diagnosis = result["diagnosis"]

    assert len(result["detections"]) == 1
    assert result["components"][0]["component"] == "shorted region"
    assert result["locations"][0]["region"] == "middle-center"
    assert diagnosis["status"] == "DEFECT_DETECTED"
    assert diagnosis["severity"] == "HIGH"
    assert diagnosis["priority"] == "HIGH"
    assert diagnosis["confidence"] == 0.93
    assert diagnosis["review_required"] is True
    assert diagnosis["possible_causes"]
    assert diagnosis["next_actions"]

    print("Session 10 application pipeline validated.")
    print(
        f"Status={diagnosis['status']}, confidence={diagnosis['confidence']:.2f}, "
        f"severity={diagnosis['severity']}, priority={diagnosis['priority']}"
    )


if __name__ == "__main__":
    main()
