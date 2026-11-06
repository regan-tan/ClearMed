import json

import pytest

from clearmed.retrieval.gate import Gate, calibrate_threshold, confusion_at

# Toy calibration table: top-1 score and whether that passage was relevant.
SCORES = [0.90, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
LABELS = [1, 1, 0, 1, 0, 0, 1]


def test_toy_calibration_table_picks_070():
    result = calibrate_threshold(SCORES, LABELS)
    chosen = result["chosen"]
    assert chosen["threshold"] == 0.70
    assert (chosen["tp"], chosen["fp"], chosen["fn"]) == (3, 1, 1)
    assert chosen["precision"] == chosen["recall"] == chosen["f1"] == 0.75
    assert len(result["curve"]) == len(SCORES) and not result["no_positive_examples"]


def test_f1_ties_go_to_higher_precision():
    # Threshold 0.9 -> P 1.0 R 0.5 F1 .667; threshold 0.6 -> P .5 R 1.0 F1 .667.
    result = calibrate_threshold([0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1])
    assert result["chosen"]["threshold"] == 0.9 and result["chosen"]["precision"] == 1.0


def test_all_negative_set_rejects_everything():
    result = calibrate_threshold([0.4, 0.9, 0.6], [0, 0, 0])
    assert result["no_positive_examples"]
    assert result["chosen"]["threshold"] > 0.9 and result["chosen"]["accepted"] == 0


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        calibrate_threshold([], [])
    with pytest.raises(ValueError):
        calibrate_threshold([0.5], [2])
    with pytest.raises(ValueError):
        confusion_at([0.5, 0.6], [1], 0.5)


def test_gate_accepts_at_or_above_threshold_and_loads_frozen_value(tmp_path):
    gate = Gate("minilm", 0.7)
    assert gate.accept(0.7) and gate.accept(0.71) and not gate.accept(0.69)
    path = tmp_path / "gate_thresholds.json"
    path.write_text(json.dumps({"thresholds": {"minilm": {"threshold": 0.66}}}), encoding="utf-8")
    assert Gate.load("minilm", path).threshold == 0.66
    with pytest.raises(KeyError, match="No frozen threshold"):
        Gate.load("e5_base_v2", path)
