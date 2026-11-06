import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from test_pipeline import FakeRetriever

from clearmed.conversation.pipeline import StubGenerator, TutorPipeline
from clearmed.retrieval.gate import Gate

SPEC = spec_from_file_location("run_conversation_pilot_for_test",
                               Path(__file__).parents[1] / "scripts" / "run_conversation_pilot.py")
assert SPEC is not None and SPEC.loader is not None
PILOT = module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)

TYPES = ("why_correct", "why_not_option", "key_clue", "mechanism", "off_topic")


def records(count=20):
    return [
        {"query_id": f"calibration-{i:03d}", "question_id": f"q{i}", "follow_up_type": TYPES[i % 5],
         "follow_up": f"follow-up {i}", "question": f"Question {i}?", "gold_label": "A", "second_turn": None,
         "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}}
        for i in range(count)
    ]


class ScoredRetriever(FakeRetriever):
    """Top-1 score depends on the query: follow-ups ending in 3 score low."""

    def retrieve(self, query_text, k):
        low = query_text.splitlines()[0].endswith("3")
        self.scores = [0.4] if low else [0.9, 0.8]
        return super().retrieve(query_text, k)


def tutor():
    return TutorPipeline(StubGenerator(), stage="calibration", retriever=ScoredRetriever([]),
                         gate=Gate("minilm", 0.5), method_id="minilm")


def test_candidates_are_on_topic_balanced_and_seeded():
    order = PILOT.candidate_order(records())
    assert len(order) == 16 and all(r["follow_up_type"] != "off_topic" for r in order)
    assert sorted(r["follow_up_type"] for r in order[:4]) == sorted(TYPES[:4])
    assert [r["query_id"] for r in order] == [r["query_id"] for r in PILOT.candidate_order(records())]


def test_pilot_counts_fallbacks_and_malformed_and_writes_sheet(tmp_path):
    def predict(rows):
        assert all("gold_label" not in row and "cop" not in row for row in rows)
        return [None if row["id"] == "q1" else "B" for row in rows]

    summary = PILOT.run(records=records(), tutor=tutor(), predict=predict, output_dir=tmp_path / "pilot", n=5)
    assert summary["n_graded"] == 5 and summary["malformed_prediction"] <= 1
    assert summary["attempted"] == 5 + summary["fallback"] + summary["malformed_prediction"]
    with (tmp_path / "pilot" / "pilot_sheet.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and all(r["evidence"].startswith("[1] passage 1") for r in rows)
    assert all(r["reply"].startswith("Answer: B") and r["addresses_follow_up"] == "" for r in rows)
    turns = (tmp_path / "pilot" / "turns.jsonl").read_text("utf-8").splitlines()
    assert len(turns) == summary["attempted"]
    with pytest.raises(FileExistsError):
        PILOT.run(records=records(), tutor=tutor(), predict=predict, output_dir=tmp_path / "pilot", n=5)


def test_too_few_accepted_replies_writes_nothing(tmp_path):
    with pytest.raises(RuntimeError, match="passed the gate"):
        PILOT.run(records=records(4), tutor=tutor(), predict=lambda rows: ["A"] * len(rows),
                  output_dir=tmp_path / "pilot", n=10)
    assert not (tmp_path / "pilot").exists()


def grade(output_dir, grades):
    sheet = output_dir / "pilot_sheet.csv"
    with sheet.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row, (addresses, uses) in zip(rows, grades, strict=True):
        row["addresses_follow_up"], row["uses_new_evidence_fact"] = addresses, uses
    with sheet.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_score_needs_complete_grades_and_applies_pass_rule(tmp_path):
    PILOT.run(records=records(), tutor=tutor(), predict=lambda rows: ["A"] * len(rows),
              output_dir=tmp_path / "pilot", n=5)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        PILOT.score(tmp_path / "pilot")
    grade(tmp_path / "pilot", [("1", "1"), ("1", "0"), ("1", "1"), ("0", "1"), ("0", "0")])
    result = PILOT.score(tmp_path / "pilot")
    assert result["passed"] == 2 and result["n"] == 5 and result["pass_rate"] == 0.4
    assert result["pass_mark"] == 3 and not result["pilot_passed"]  # 60% of 5, rounded up.
    low, high = result["pass_rate_ci95"]
    assert low <= 0.4 <= high
    assert json.loads((tmp_path / "pilot" / "pilot_result.json").read_text("utf-8"))["passed"] == 2


def test_pass_mark_stays_six_of_ten():
    assert __import__("math").ceil(PILOT.PASS_RATE * 10) == 6


def gate_off_tutor():
    return TutorPipeline(StubGenerator(), stage="calibration", retriever=ScoredRetriever([]),
                         gate_off=True, method_id="minilm_gate_off")


def test_gate_off_pilot_answers_every_follow_up_in_the_same_order(tmp_path):
    first = PILOT.run(records=records(), tutor=gate_off_tutor(), predict=lambda rows: ["B"] * len(rows),
                      output_dir=tmp_path / "a", n=8)
    PILOT.run(records=records(), tutor=gate_off_tutor(), predict=lambda rows: ["B"] * len(rows),
              output_dir=tmp_path / "b", n=8)
    assert first["fallback"] == 0 and first["attempted"] == 8 and first["gate_off"] is True
    assert first["threshold"] is None
    ids = [(tmp_path / d / "pilot_sheet.csv").read_text("utf-8").splitlines() for d in ("a", "b")]
    assert [line.split(",")[0] for line in ids[0]] == [line.split(",")[0] for line in ids[1]]


def test_compare_pairs_pilots_and_rejects_different_follow_ups(tmp_path):
    for name in ("a", "b"):
        PILOT.run(records=records(), tutor=gate_off_tutor(), predict=lambda rows: ["B"] * len(rows),
                  output_dir=tmp_path / name, n=4)
    grade(tmp_path / "a", [("1", "1")] * 3 + [("0", "0")])
    grade(tmp_path / "b", [("1", "1")] + [("1", "0")] * 3)
    result = PILOT.compare(tmp_path / "a", tmp_path / "b")
    assert (result["passed_a"], result["passed_b"], result["only_a_passed"], result["only_b_passed"]) == (3, 1, 2, 0)
    assert result["a_minus_b"]["mean_difference"] == pytest.approx(0.5)
    PILOT.run(records=records(), tutor=gate_off_tutor(), predict=lambda rows: ["B"] * len(rows),
              output_dir=tmp_path / "c", n=3)
    grade(tmp_path / "c", [("1", "1")] * 3)
    with pytest.raises(ValueError, match="different follow-ups"):
        PILOT.compare(tmp_path / "a", tmp_path / "c")


@pytest.mark.parametrize("text, expected", [
    ("Ans. (c) Hypothyroidism", True), ("Answer is B because", True), ("ans: d", True),
    ("Ans. is 'a' i.e., Clozapine", True), ("Ans. Dhatura", False), ("the answer is a hormone", False),
    ("The answer depends on cardiac output", False), ("Answers vary", False),
])
def test_stray_answer_letter_pattern(text, expected):
    assert bool(PILOT.STRAY_ANSWER_LETTER.search(text)) is expected
