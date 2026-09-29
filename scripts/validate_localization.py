"""Session 6 localization smoke test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.localization import localize_defects


def main() -> None:
    matches = [
        {
            "device": "pcb",
            "detector_class": "short",
            "component": "shorted region",
            "subsystem": "PCB interconnect",
            "confidence": 0.91,
            "bbox": [100, 50, 300, 250],
        },
        {
            "device": "pcb",
            "detector_class": "copper",
            "component": "copper region",
            "subsystem": "PCB interconnect",
            "confidence": 0.84,
            "bbox": [700, 500, 900, 700],
        },
    ]

    results = localize_defects(matches, image_width=1000, image_height=1000)

    assert len(results) == 2
    assert results[0]["normalized_bbox"] == [0.1, 0.05, 0.3, 0.25]
    assert results[0]["center"] == [200.0, 150.0]
    assert results[0]["region"] == "middle-center"
    assert abs(results[0]["area_ratio"] - 0.04) < 1e-9
    assert results[1]["region"] == "middle-right"

    print("Defect localization validated.")
    print(f"Locations: {len(results)}")
    for result in results:
        print(
            f"- {result['defect_class']} -> {result['component']} "
            f"@ {result['region']} | area={result['area_ratio']:.3f}"
        )
    print("Session 6 defect localization validation passed.")


if __name__ == "__main__":
    main()
