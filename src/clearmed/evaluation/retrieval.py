"""Ranked-retrieval metrics against pooled relevance judgments (RQ3, decision D5).

Judgments are pooled: a passage counts as relevant only if it was judged
relevant for that query; anything unjudged counts as not relevant. A query
whose pool contains no relevant passage cannot separate the models, so it is
left out of every average and reported in ``excluded``.

Metrics per query, for ranked doc IDs ``r`` and relevant set ``R``:

* Hit@k: 1 if any of the top k is relevant, else 0.
* Recall@k: relevant passages in the top k divided by ``|R|``.
* MRR@k: 1 / rank of the first relevant passage within the top k, else 0.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

HIT_KS = (1, 3, 5)
RECALL_KS = (1, 3, 5)
MRR_K = 5
SEED = 42


def query_metrics(ranked: Sequence[str], relevant: set[str]) -> dict[str, float]:
    """Metrics for one query; ``relevant`` must be non-empty."""
    if not relevant:
        raise ValueError("A query needs at least one relevant passage to be scored.")
    flags = [doc_id in relevant for doc_id in ranked]
    metrics: dict[str, float] = {}
    for k in HIT_KS:
        metrics[f"hit@{k}"] = float(any(flags[:k]))
    for k in RECALL_KS:
        metrics[f"recall@{k}"] = sum(flags[:k]) / len(relevant)
    first = next((rank for rank, flag in enumerate(flags[:MRR_K], start=1) if flag), None)
    metrics[f"mrr@{MRR_K}"] = 1.0 / first if first else 0.0
    return metrics


def relevant_sets(judgments: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    """query_id -> relevant doc IDs, for every judged query (possibly empty)."""
    sets: dict[str, set[str]] = {}
    for judgment in judgments:
        relevant = sets.setdefault(str(judgment["query_id"]), set())
        if int(judgment["relevant"]) == 1:
            relevant.add(str(judgment["doc_id"]))
    return sets


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], relevant: Mapping[str, set[str]]
) -> dict[str, Any]:
    """Score one method. Every judged query must have a ranking."""
    missing = sorted(set(relevant) - set(rankings))
    if missing:
        raise ValueError(f"No ranking for {len(missing)} judged queries, e.g. {missing[:3]}.")
    per_query = {}
    excluded = []
    for query_id in sorted(relevant):
        if relevant[query_id]:
            per_query[query_id] = query_metrics(rankings[query_id], relevant[query_id])
        else:
            excluded.append(query_id)
    if not per_query:
        raise ValueError("No query has a relevant passage; nothing can be scored.")
    names = list(next(iter(per_query.values())))
    means = {name: float(np.mean([values[name] for values in per_query.values()])) for name in names}
    return {"n_evaluated": len(per_query), "n_excluded": len(excluded), "excluded": excluded,
            "mean": means, "per_query": per_query}


def bootstrap_ci(values: Sequence[float], *, n_resamples: int = 10_000, seed: int = SEED,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(n_resamples, data.size))].mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def paired_test(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 10_000,
                seed: int = SEED) -> dict[str, float]:
    """Paired comparison of two methods on the same queries.

    Two-sided p-value from a sign-flip permutation test on per-query
    differences (the null: each difference is equally likely to be + or -),
    plus a bootstrap 95% interval for the mean difference ``a - b``.
    """
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0 or diff.size != len(b):
        raise ValueError("Paired samples must be non-empty and the same length.")
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_resamples, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    p_value = (1 + int(np.sum(null >= observed - 1e-12))) / (n_resamples + 1)
    low, high = bootstrap_ci(diff, n_resamples=n_resamples, seed=seed)
    return {"mean_difference": float(diff.mean()), "ci_low": low, "ci_high": high, "p_value": float(p_value),
            "n_queries": int(diff.size), "n_differing": int(np.count_nonzero(diff))}
