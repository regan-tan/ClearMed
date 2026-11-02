import csv
import io
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from random import Random

import pytest

from clearmed.conversation.pipeline import NO_RETRIEVAL, StubGenerator, TutorPipeline
from clearmed.conversation.rubric import (
    COLUMNS,
    assign_labellers,
    gradable_replies,
    read_scores,
)
from clearmed.evaluation.conversation import classify_fallbacks, kappa, reply_scores
from clearmed.retrieval.gate import Gate


def load_script(name):
    spec = spec_from_file_location(f"{name}_for_test", Path(__file__).parents[1] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RUN = load_script("run_conversation_eval")
EXPORT = load_script("export_rubric_sheets")
EVALUATE = load_script("evaluate_conversation")
METHODS = ("minilm", "e5_base_v2", "biolord_2023")


def records():
    rows = []
    for i in range(6):
        rows.append({
            "query_id": f"evaluation-{i:03d}", "question_id": f"v{i}", "follow_up_type": "why_correct",
            "follow_up": f"Why {i}?", "question": f"Question {i}?", "gold_label": "A", "gold_answer_text": "alpha",
            "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
            "rq4": i < 5, "second_turn": "Summarise?" if i in (0, 1) else None,
            "exp": "SECRET",
        })
    return rows


class Retriever:
    """Doc IDs depend on the query number; query 4 scores below every gate."""

    def __init__(self, method):
        self.method = method

    def retrieve(self, query_text, k):
        number = int(query_text.splitlines()[0].removeprefix("Follow-up: Why ").rstrip("?")) \
            if "Why " in query_text.splitlines()[0] else 0
        score = 0.1 if number == 4 else 0.9
        return [{"rank": r, "doc_id": f"{self.method}-q{number}-d{r}", "score": score, "text": f"evidence {r}"}
                for r in range(1, k + 1)]


def tutors(stage="rq4"):
    built = {NO_RETRIEVAL: TutorPipeline(StubGenerator(), stage=stage)}
    for method in METHODS:
        built[method] = TutorPipeline(StubGenerator(), stage=stage, retriever=Retriever(method),
                                      gate=Gate(method, 0.5), method_id=method)
    return built


def predict(rows):
    assert all(set(row) == {"id", "question", "opa", "opb", "opc", "opd"} for row in rows)
    return [None if row["id"] == "v3" else "B" for row in rows]


def write_pool(root):
    pool_dir = root / "judging" / "pool-1"
    pool_dir.mkdir(parents=True)
    (pool_dir / "pool.json").write_text("{}", "utf-8")
    judgments = [
        {"stage": "B", "query_id": "evaluation-000", "doc_id": "minilm-q0-d1", "relevant": 1},
        {"stage": "B", "query_id": "evaluation-001", "doc_id": "minilm-q1-d1", "relevant": 0},
        {"stage": "B", "query_id": "evaluation-004", "doc_id": "x", "relevant": 0},
    ]
    (pool_dir / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments), "utf-8")


def fill(text, sheet, value):
    rows = list(csv.DictReader(io.StringIO(text)))
    for row in rows:
        for criterion in ("correct", "no_unsupported_claims", "grounded"):
            if criterion in row:
                row[criterion] = value(row)
        if "consistent" in row and row["consistent"] == "":
            row["consistent"] = "2"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS[sheet], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def test_full_rq4_flow(tmp_path):
    run_dir = tmp_path / "eval" / "rq4-test"
    summary = RUN.run(records=records(), tutors=tutors(), predict=predict, output_dir=run_dir)
    # 5 RQ4 queries, v3 malformed -> 4 used; 2 second turns -> 6 turns per system.
    assert summary["malformed_predictions"] == 1 and summary["n_rq4_queries"] == 5
    assert {s: v["turns"] for s, v in summary["systems"].items()} == dict.fromkeys(RUN.SYSTEMS, 6)
    assert summary["systems"]["minilm"]["fallbacks"] == 1 and summary["systems"][NO_RETRIEVAL]["fallbacks"] == 0
    minilm = [json.loads(line) for line in (run_dir / "minilm" / "predictions.jsonl").read_text("utf-8").splitlines()]
    assert all("SECRET" not in (t["prompt"] or "") and "Answer: A" not in (t["query_text"] or "") for t in minilm)
    assert all(t["predicted_label"] == "B" for t in minilm)

    records_by_id = {r["query_id"]: r for r in records() if r["rq4"]}
    exported = EXPORT.run(run_dir=run_dir, records=records_by_id, explanations={"v0": "Reference text."})
    # Blind: 4 systems x 6 turns - 3 fallbacks = 21 replies; grounding: 18 - 3 = 15.
    assert exported["blind"]["replies"] == 21 and exported["grounding"]["replies"] == 15
    assert exported["blind"]["shared_replies"] == round(21 * 0.2)
    assert exported["fallbacks_not_exported"]["e5_base_v2"] == 1
    rubric = run_dir / "rubric"
    blind = (rubric / "blind_sheet.csv").read_text("utf-8")
    for name in (*METHODS, NO_RETRIEVAL, "minilm-q"):
        assert name not in blind
    assert "Reference text." in blind and "(none in dataset)" in blind
    grounding = (rubric / "grounding_sheet.csv").read_text("utf-8")
    assert "[1] evidence 1" in grounding
    with pytest.raises(FileExistsError):
        EXPORT.run(run_dir=run_dir, records=records_by_id, explanations={})

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "blind_sheet.csv").write_text(fill(blind, "blind", lambda row: "2"), "utf-8")
    (labels_dir / "grounding_sheet.csv").write_text(fill(grounding, "grounding", lambda row: "1"), "utf-8")
    write_pool(tmp_path)
    metrics = EVALUATE.run(run_dir=run_dir, labels_dir=labels_dir, pool_id="pool-1",
                           judging_root=tmp_path / "judging")
    assert metrics["all_replies"]["minilm"]["correct"]["mean"] == 2.0
    assert metrics["all_replies"]["minilm"]["grounded"]["mean"] == 1.0
    assert metrics["all_replies"][NO_RETRIEVAL]["grounded"]["n"] == 0
    assert metrics["all_replies"]["minilm"]["consistent"]["n"] == 2
    # evaluation-004 fell back for every retrieval system; stage B had no relevant passage -> correct.
    assert metrics["fallbacks"]["minilm"] == {"replies": 6, "fallbacks": 1, "correct": 1, "unnecessary": 0,
                                              "unjudged": 0}
    assert metrics["answered_by_all"]["n_turns"] == 5
    # Judged first turns: evaluation-000 (MiniLM's evidence holds the relevant passage) and -001 (it does not).
    split = metrics["retrieval_vs_reply"]["minilm"]
    assert split["relevant_evidence"]["correct"]["n"] == 1 and split["no_relevant_evidence"]["correct"]["n"] == 1
    assert metrics["retrieval_vs_reply"]["e5_base_v2"]["relevant_evidence"]["correct"]["n"] == 0
    assert metrics["kappa"]["correct"]["cohen_kappa"] == 1.0
    assert "MiniLM" in (run_dir / "rq4_table.md").read_text("utf-8")
    with pytest.raises(FileExistsError):
        EVALUATE.run(run_dir=run_dir, labels_dir=labels_dir, pool_id="pool-1", judging_root=tmp_path / "judging")


def test_run_refuses_non_rq4_stage_and_missing_systems(tmp_path):
    with pytest.raises(ValueError, match="stage 'rq4'"):
        RUN.run(records=records(), tutors=tutors(stage="calibration"), predict=predict, output_dir=tmp_path / "a")
    partial = tutors()
    del partial["biolord_2023"]
    with pytest.raises(ValueError, match="Expected tutors"):
        RUN.run(records=records(), tutors=partial, predict=predict, output_dir=tmp_path / "b")


def test_prediction_file_marks_non_letters_malformed(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"question_id": "v1", "predicted_label": "C"}\n{"question_id": "v2", "predicted_label": "MALFORMED"}\n')
    assert RUN.load_prediction_file(path) == {"v1": "C", "v2": None}


def replies(n):
    return [{"system": "minilm", "query_id": f"q{i}", "turn": 1} for i in range(n)]


def test_assignment_shares_twenty_percent_and_splits_the_rest():
    entries = assign_labellers(replies(20), sheet="blind", rng=Random(1))
    shared = {e["query_id"] for e in entries if e["shared"]}
    assert len(shared) == 4 and len(entries) == 24
    per_labeller = [sum(e["labeller"] == name for e in entries) for name in ("labeller_1", "labeller_2")]
    assert per_labeller == [12, 12] and len({e["item_id"] for e in entries}) == 24
    assert entries == assign_labellers(replies(20), sheet="blind", rng=Random(1))


def test_gradable_replies_skip_fallbacks_and_control_for_grounding():
    turns = {NO_RETRIEVAL: [{"query_id": "q", "turn": 1, "fallback": False}],
             "minilm": [{"query_id": "q", "turn": 1, "fallback": True}, {"query_id": "r", "turn": 1, "fallback": False}]}
    assert len(gradable_replies(turns, "blind")) == 2
    assert gradable_replies(turns, "grounding") == [{"system": "minilm", "query_id": "r", "turn": 1}]


def test_read_scores_rejects_bad_sheets():
    key = [{"item_id": "R-1", "labeller": "labeller_1", "sheet": "blind", "turn": 1, "shared": False}]

    def sheet(**values):
        row = dict.fromkeys(COLUMNS["blind"], "") | {"item_id": "R-1", "labeller": "labeller_1",
                                                    "consistent": "n/a"} | values
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=COLUMNS["blind"])
        writer.writeheader()
        writer.writerow(row)
        return buffer.getvalue()

    assert read_scores(sheet(correct="2", no_unsupported_claims="0"), key, "blind") == {
        "R-1": {"correct": 2, "no_unsupported_claims": 0}}
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        read_scores(sheet(correct="3", no_unsupported_claims="0"), key, "blind")
    with pytest.raises(ValueError, match="second turns only"):
        read_scores(sheet(correct="1", no_unsupported_claims="1", consistent="2"), key, "blind")
    with pytest.raises(ValueError, match="labeller changed"):
        read_scores(sheet(correct="1", no_unsupported_claims="1", labeller="labeller_2"), key, "blind")


def test_kappa_and_mean_of_two_labellers():
    key = []
    scores = {}
    for i, (a, b) in enumerate([(0, 0), (1, 2), (2, 2), (2, 1)]):
        for labeller, value in (("labeller_1", a), ("labeller_2", b)):
            item_id = f"R-{i}-{labeller}"
            key.append({"item_id": item_id, "system": "minilm", "query_id": f"q{i}", "turn": 1,
                        "labeller": labeller, "shared": True, "sheet": "blind"})
            scores[item_id] = {"correct": value}
    result = kappa(key, scores, "correct")
    assert result["shared_replies"] == 4 and result["raw_agreement"] == 0.5
    assert -1 <= result["cohen_kappa"] < result["weighted_kappa"] <= 1
    assert reply_scores(key, scores)[("minilm", "q1", 1)]["correct"] == 1.5


def test_second_turn_fallbacks_are_unjudged():
    turns = {"minilm": [{"query_id": "q", "turn": 2, "fallback": True},
                        {"query_id": "q", "turn": 1, "fallback": True}]}
    assert classify_fallbacks(turns, {"q": {"d"}})["minilm"] == {
        "replies": 2, "fallbacks": 2, "correct": 0, "unnecessary": 1, "unjudged": 1}
