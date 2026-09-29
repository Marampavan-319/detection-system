"""Validate Session 5 component identification without model inference."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.components import identify_components


def main() -> int:
    detections = [
        {"class_name": "short", "confidence": 0.91, "bbox": [10, 20, 40, 60]},
        {"class_name": "copper", "confidence": 0.84, "bbox": [50, 60, 90, 100]},
    ]
    matches = identify_components("pcb", detections)

    if len(matches) != 2:
        print("Component validation failed: unexpected match count.")
        return 1
    if matches[0]["component"] != "shorted region":
        print("Component validation failed: PCB mapping is incorrect.")
        return 1
    if matches[0]["subsystem"] != "PCB interconnect":
        print("Component validation failed: subsystem mapping is incorrect.")
        return 1

    print("Component identification mappings validated.")
    print(f"Matches: {len(matches)}")
    for match in matches:
        print(f"- {match['component']} -> {match['subsystem']}")
    print("Session 5 component identification validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
