import pytest

from clearmed.evaluation.retrieval import (
    bootstrap_ci,
    evaluate_rankings,
    paired_test,
    query_metrics,
    relevant_sets,
)


def test_worked_example_by_hand():
    # Relevant passages at ranks 2 and 5.
    metrics = query_metrics(["d1", "d2", "d3", "d4", "d5"], {"d2", "d5"})
    assert metrics == {
        "hit@1": 0.0, "hit@3": 1.0, "hit@5": 1.0,
        "recall@1": 0.0, "recall@3": 0.5, "recall@5": 1.0,
        "mrr@5": 0.5,
    }


def test_relevant_passage_outside_top_five_scores_zero_but_counts_in_recall():
    metrics = query_metrics(["a", "b", "c", "d", "e"], {"a", "z"})
    assert metrics["hit@1"] == 1.0 and metrics["mrr@5"] == 1.0
    assert metrics["recall@5"] == 0.5  # "z" was relevant for another model only
    nothing = query_metrics(["a", "b", "c", "d", "e"], {"z"})
    assert nothing["hit@5"] == 0.0 and nothing["mrr@5"] == 0.0 and nothing["recall@5"] == 0.0


def test_query_without_relevant_passage_is_left_out_and_counted():
    judgments = [
        {"query_id": "q1", "doc_id": "a", "relevant": 1},
        {"query_id": "q1", "doc_id": "b", "relevant": 0},
        {"query_id": "q2", "doc_id": "c", "relevant": 0},
    ]
    sets = relevant_sets(judgments)
    assert sets == {"q1": {"a"}, "q2": set()}
    result = evaluate_rankings({"q1": ["b", "a"], "q2": ["c"]}, sets)
    assert result["n_evaluated"] == 1 and result["excluded"] == ["q2"]
    assert result["mean"]["mrr@5"] == 0.5
    with pytest.raises(ValueError, match="at least one relevant"):
        query_metrics(["a"], set())
    with pytest.raises(ValueError, match="No ranking"):
        evaluate_rankings({"q1": ["a"]}, {"q1": {"a"}, "q3": {"x"}})
    with pytest.raises(ValueError, match="nothing can be scored"):
        evaluate_rankings({"q2": ["c"]}, {"q2": set()})


def test_bootstrap_interval_contains_mean_and_is_deterministic():
    values = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    low, high = bootstrap_ci(values)
    assert low <= 0.7 <= high and 0 <= low < high <= 1
    assert bootstrap_ci(values) == (low, high)
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0)
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_test_detects_clear_difference_and_not_noise():
    better = [1.0] * 20
    worse = [0.0] * 20
    clear = paired_test(better, worse)
    assert clear["mean_difference"] == 1.0 and clear["p_value"] < 0.001
    same = paired_test([1, 0, 1, 0], [1, 0, 1, 0])
    assert same["p_value"] == 1.0 and same["n_differing"] == 0
    mixed = paired_test([1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0])
    assert mixed["mean_difference"] == 0 and mixed["p_value"] > 0.5
    with pytest.raises(ValueError):
        paired_test([1, 0], [1])
