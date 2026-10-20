from clearmed.data.seq2seq import parse_answer_label
from clearmed.evaluation.qa import evaluate_mcq_predictions


def row(index, *, subject="Medicine"):
    return {
        "id": f"id-{index}",
        "question": f"Question {index}",
        "opa": "A answer",
        "opb": "B answer",
        "opc": "C answer",
        "opd": "D answer",
        "cop": index % 4,
        "subject_name": subject,
        "topic_name": "Topic",
    }


def test_question_level_evaluation_and_subject_statistics():
    rows = [row(0, subject="Anatomy"), row(1, subject="Medicine"), row(2, subject="Medicine")]
    labels = iter(["A", "C", "X"])
    metrics = evaluate_mcq_predictions(rows, lambda item: {"predicted_answer": next(labels), "candidate_scores": {}})

    assert metrics["total_evaluated"] == 3
    assert metrics["correct"] == 1
    assert metrics["incorrect"] == 2
    assert metrics["accuracy"] == 1 / 3
    assert metrics["malformed_output_count"] == 1
    assert metrics["confusion_matrix"]["C"]["MALFORMED"] == 1
    assert sum(sum(row.values()) for row in metrics["confusion_matrix"].values()) == metrics["total_evaluated"]
    assert sum(metrics["prediction_distribution"].values()) == metrics["total_evaluated"]
    assert sum(item["total"] for item in metrics["subject_accuracy"]) == metrics["total_evaluated"]
    assert sum(item["correct"] for item in metrics["subject_accuracy"]) == metrics["correct"]
    assert metrics["accuracy_by_gold_label"]["A"] == {"total": 1, "correct": 1, "accuracy": 1.0}
    assert metrics["subject_accuracy"] == [
        {"subject_name": "Anatomy", "total": 1, "correct": 1, "accuracy": 1.0},
        {"subject_name": "Medicine", "total": 2, "correct": 0, "accuracy": 0.0},
    ]
    assert len(metrics["errors"]) == 2


def test_error_limit_and_missing_subject_are_deterministic():
    metrics = evaluate_mcq_predictions([row(0, subject=" "), row(1)], lambda item: {"predicted_answer": "D"}, error_limit=1)

    assert len(metrics["errors"]) == 1
    assert metrics["subject_accuracy"][0]["subject_name"] == "Medicine"
    assert metrics["subject_accuracy"][1]["subject_name"] == "Unknown"


def test_shared_evaluator_scores_standalone_letters_but_keeps_ambiguous_output_malformed():
    rows = [row(0), row(1), row(2)]
    raw_outputs = iter(["A", "B.", "B12."])
    metrics = evaluate_mcq_predictions(
        rows, lambda _: {"predicted_answer": parse_answer_label(next(raw_outputs))}
    )

    assert metrics["correct"] == 2
    assert metrics["malformed_output_count"] == 1
    assert metrics["confusion_matrix"]["C"]["MALFORMED"] == 1
