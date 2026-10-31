"""Confidence gate: accept retrieved evidence only above a per-model threshold.

Cosine scores are not comparable across embedding models, so each model gets
its own threshold, calibrated on the stage A labels (each model's top-1
passage for the internal_dev calibration queries) and then frozen in
``configs/retrieval/gate_thresholds.json``. It is never retuned on validation.

Calibration tries every observed top-1 score as the threshold (accept when
``score >= threshold``) and keeps the best F1; ties go to higher precision,
then to the higher (more cautious) threshold.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

THRESHOLDS_PATH = Path("configs/retrieval/gate_thresholds.json")


def confusion_at(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict[str, float]:
    """Precision, recall and F1 when accepting ``score >= threshold``."""
    tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and y == 1)
    fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and y == 0)
    fn = sum(1 for s, y in zip(scores, labels, strict=True) if s < threshold and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "accepted": tp + fp, "tp": tp, "fp": fp, "fn": fn}


def calibrate_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Pick the threshold with the best F1 over the observed scores.

    With no relevant examples nothing should ever be accepted, so the
    threshold is set just above the highest score and flagged.
    """
    if len(scores) != len(labels) or not scores:
        raise ValueError("scores and labels must be non-empty and the same length.")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 or 1.")
    curve = [confusion_at(scores, labels, t) for t in sorted(set(scores))]
    if not any(labels):
        threshold = math.nextafter(max(scores), math.inf)
        return {"chosen": confusion_at(scores, labels, threshold), "curve": curve, "no_positive_examples": True,
                "n": len(scores), "n_positive": 0}
    best = max(curve, key=lambda point: (point["f1"], point["precision"], point["threshold"]))
    return {"chosen": best, "curve": curve, "no_positive_examples": False, "n": len(scores),
            "n_positive": int(sum(labels))}


class Gate:
    """Runtime accept/reject decision for one model's top-1 score."""

    def __init__(self, method_id: str, threshold: float) -> None:
        self.method_id = method_id
        self.threshold = float(threshold)

    def accept(self, score: float) -> bool:
        return float(score) >= self.threshold

    @classmethod
    def load(cls, method_id: str, path: Path = THRESHOLDS_PATH) -> Gate:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        if method_id not in values.get("thresholds", {}):
            raise KeyError(f"No frozen threshold for {method_id!r} in {path}.")
        return cls(method_id, values["thresholds"][method_id]["threshold"])
