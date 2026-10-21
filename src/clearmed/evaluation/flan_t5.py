"""FLAN-T5-specific result records and explanation diagnostics.

MCQ scoring remains in :mod:`clearmed.evaluation.qa`; this module only adds
the explanation measurements required for the generative Part I comparison.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from rouge_score import rouge_scorer

from clearmed.data.preprocessing import get_answer_label, is_nonblank_text


def prediction_records(
    rows: Sequence[Mapping[str, Any]], generated: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join source gold fields only into post-inference evaluation records."""
    if len(rows) != len(generated):
        raise ValueError("Rows and generated outputs must have matching lengths.")
    records: list[dict[str, Any]] = []
    for row, output in zip(rows, generated, strict=True):
        reference = row.get("exp").strip() if is_nonblank_text(row.get("exp")) else ""
        records.append({
            **dict(output), "gold_label": get_answer_label(row),
            "predicted_label": output.get("predicted_answer") or "MALFORMED",
            "reference_explanation": reference, "subject_name": row.get("subject_name"),
        })
    return records


def _repetition(record: Mapping[str, Any]) -> tuple[bool, float]:
    tokens = str(record.get("generated_explanation") or "").casefold().split()
    trigrams = list(zip(tokens, tokens[1:], tokens[2:], strict=False))
    if not trigrams:
        return False, 0.0
    repeats = len(trigrams) - len(set(trigrams))
    return repeats > 0, repeats / len(trigrams)


def explanation_metrics(
    records: Sequence[Mapping[str, Any]], *, compute_bertscore: bool = False,
    bertscore_device: str | None = None,
) -> dict[str, Any]:
    """Calculate transparent RQ2 diagnostics for reference-eligible rows."""
    eligible = [record for record in records if record.get("reference_explanation")]
    non_empty = [record for record in eligible if str(record.get("generated_explanation") or "").strip()]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_values = [
        scorer.score(str(record["reference_explanation"]), str(record["generated_explanation"]))["rougeL"].fmeasure
        for record in eligible
    ]
    repeated = [_repetition(record) for record in eligible]
    token_lengths = [int(record.get("generation_token_count", 0)) for record in eligible]
    payload: dict[str, Any] = {
        "explanation_eligible_examples": len(eligible),
        "non_empty_generated_explanations": len(non_empty),
        "non_empty_generated_explanation_rate": len(non_empty) / len(eligible) if eligible else 0.0,
        "explanation_format_compliant_count": sum(bool(record.get("answer_explanation_format_compliant")) for record in eligible),
        "explanation_format_compliant_rate": sum(bool(record.get("answer_explanation_format_compliant")) for record in eligible) / len(eligible) if eligible else 0.0,
        "rouge_l_f1_mean": mean(rouge_values) if rouge_values else None,
        "eos_terminated_count": sum(bool(record.get("generation_eos_observed")) for record in eligible),
        "eos_termination_rate": sum(bool(record.get("generation_eos_observed")) for record in eligible) / len(eligible) if eligible else 0.0,
        "length_limit_truncation_count": sum(bool(record.get("generation_length_limited")) for record in eligible),
        "length_limit_truncation_rate": sum(bool(record.get("generation_length_limited")) for record in eligible) / len(eligible) if eligible else 0.0,
        "mean_generated_token_count": mean(token_lengths) if token_lengths else 0.0,
        "repeated_trigram_count": sum(flag for flag, _ in repeated),
        "repeated_trigram_rate": sum(flag for flag, _ in repeated) / len(eligible) if eligible else 0.0,
        "mean_repeated_trigram_fraction": mean(value for _, value in repeated) if repeated else 0.0,
        "bertscore": {"computed": False, "f1_mean": None},
        "note": "Automatic overlap metrics supplement, not replace, factual-quality and unsupported-claim review.",
    }
    if compute_bertscore and eligible:
        from bert_score import score

        _, _, f1 = score(
            [str(record["generated_explanation"]) for record in eligible],
            [str(record["reference_explanation"]) for record in eligible],
            lang="en", device=bertscore_device, verbose=True,
        )
        payload["bertscore"] = {"computed": True, "f1_mean": float(f1.mean().cpu())}
    return payload


def generation_parse_distribution(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Expose strict parse modes alongside answer accuracy."""
    return dict(sorted(Counter(str(record.get("answer_parse_mode")) for record in records).items()))
