"""Deterministic smoke test for Session 8 severity/confidence scoring."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.severity import SeverityScorer


def main() -> None:
    scorer = SeverityScorer()

    high = scorer.score(
        [{"device": "pcb", "defect_class": "short", "confidence": 0.91}],
        {"evidence": ["Visual detection", "OCR/error evidence"]},
    )
    assert high.confidence == 0.93
    assert high.severity == "HIGH"
    assert high.priority == "HIGH"

    medium = scorer.score(
        [{"device": "laptop", "defect_class": "screen", "confidence": 0.84}]
    )
    assert medium.confidence == 0.84
    assert medium.severity == "MEDIUM"
    assert medium.priority == "MEDIUM"

    low = scorer.score([])
    assert low.confidence == 0.0
    assert low.severity == "UNKNOWN"

    router = scorer.score(
        [{"device": "router", "defect_class": "lan not connected", "confidence": 0.88}]
    )
    assert router.severity == "LOW"
    assert router.priority == "MEDIUM"

    print("Severity and confidence validated.")
    print(f"High case: confidence={high.confidence:.2f}, severity={high.severity}, priority={high.priority}")
    print(f"Medium case: confidence={medium.confidence:.2f}, severity={medium.severity}, priority={medium.priority}")
    print(f"Empty case: confidence={low.confidence:.2f}, severity={low.severity}")
    print(f"Router case: confidence={router.confidence:.2f}, severity={router.severity}, priority={router.priority}")
    print("Session 8 severity/confidence validation passed.")


if __name__ == "__main__":
    main()
