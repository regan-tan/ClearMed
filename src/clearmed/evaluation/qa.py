"""Reusable, question-level evaluation for four-option MCQ predictors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from clearmed.data.preprocessing import (
    OPTION_FIELDS,
    OPTION_LABELS,
    get_answer_label,
    is_nonblank_text,
    is_valid_qa_example,
)

PredictionFunction = Callable[[Mapping[str, Any]], Mapping[str, Any]]
MALFORMED_PREDICTION = "MALFORMED"


def evaluate_mcq_predictions(
    rows: Iterable[Mapping[str, Any]],
    predict: PredictionFunction,
    *,
    error_limit: int = 50,
) -> dict[str, Any]:
    """Evaluate any MCQ predictor at original-question level.

    Invalid gold rows are skipped.  Predictions outside A--D are counted as
    malformed rather than coerced, which keeps this evaluator comparable with
    future generative systems.
    """
    if error_limit < 0:
        raise ValueError("error_limit must be non-negative.")
    total = correct = malformed = 0
    gold_counts: Counter[str] = Counter()
    gold_correct: Counter[str] = Counter()
    prediction_counts: Counter[str] = Counter()
    subject_total: Counter[str] = Counter()
    subject_correct: Counter[str] = Counter()
    prediction_labels = (*OPTION_LABELS, MALFORMED_PREDICTION)
    confusion: dict[str, Counter[str]] = {label: Counter() for label in OPTION_LABELS}
    errors: list[dict[str, Any]] = []

    for row in rows:
        if not is_valid_qa_example(row):
            continue
        total += 1
        gold = get_answer_label(row)
        gold_counts[gold] += 1
        subject_value = row.get("subject_name")
        subject = subject_value.strip() if is_nonblank_text(subject_value) else "Unknown"
        subject_total[subject] += 1
        result = predict(row)
        prediction = result.get("predicted_answer")
        valid_prediction = prediction in OPTION_LABELS
        if not valid_prediction:
            malformed += 1
            prediction_key = MALFORMED_PREDICTION
        else:
            prediction_key = str(prediction)
        confusion[gold][prediction_key] += 1
        prediction_counts[prediction_key] += 1
        is_correct = valid_prediction and prediction == gold
        if is_correct:
            correct += 1
            gold_correct[gold] += 1
            subject_correct[subject] += 1
        elif len(errors) < error_limit:
            errors.append(
                {
                    "question_id": row.get("id"),
                    "question": row["question"].strip(),
                    "options": {
                        label: row[field].strip()
                        for label, field in zip(OPTION_LABELS, OPTION_FIELDS, strict=True)
                    },
                    "gold_answer": gold,
                    "predicted_answer": prediction if valid_prediction else None,
                    "candidate_scores": result.get("candidate_scores"),
                    "subject_name": row.get("subject_name"),
                    "topic_name": row.get("topic_name"),
                }
            )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    subject_accuracy = [
        {
            "subject_name": subject,
            "total": subject_total[subject],
            "correct": subject_correct[subject],
            "accuracy": rate(subject_correct[subject], subject_total[subject]),
        }
        for subject in sorted(subject_total, key=lambda item: (item.casefold(), item))
    ]
    by_gold_label = {
        label: {
            "total": gold_counts[label],
            "correct": gold_correct[label],
            "accuracy": rate(gold_correct[label], gold_counts[label]),
        }
        for label in OPTION_LABELS
    }
    return {
        "total_evaluated": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": rate(correct, total),
        "malformed_output_count": malformed,
        "malformed_output_rate": rate(malformed, total),
        "accuracy_by_gold_label": by_gold_label,
        "prediction_distribution": dict(sorted(prediction_counts.items())),
        "confusion_matrix": {
            gold: {prediction: confusion[gold][prediction] for prediction in prediction_labels}
            for gold in OPTION_LABELS
        },
        "subject_accuracy": subject_accuracy,
        "errors": errors,
    }
