from clearmed.evaluation.flan_t5 import explanation_metrics, prediction_records


def test_explanation_metrics_count_empty_outputs_and_length_limits():
    records = prediction_records(
        [
            {"id": "1", "cop": 0, "exp": "Reference explanation."},
            {"id": "2", "cop": 1, "exp": "Another reference."},
        ],
        [
            {"predicted_answer": "A", "generated_explanation": "Generated explanation.", "answer_parse_mode": "answer_and_explanation", "answer_explanation_format_compliant": True, "generation_token_count": 4, "generation_eos_observed": True, "generation_length_limited": False},
            {"predicted_answer": "B", "generated_explanation": "", "answer_parse_mode": "answer_only", "answer_explanation_format_compliant": False, "generation_token_count": 128, "generation_eos_observed": False, "generation_length_limited": True},
        ],
    )
    metrics = explanation_metrics(records)
    assert metrics["explanation_eligible_examples"] == 2
    assert metrics["non_empty_generated_explanations"] == 1
    assert metrics["eos_terminated_count"] == 1
    assert metrics["length_limit_truncation_count"] == 1
