"""RQ4 metrics: rubric means, fallback accounting, kappa and retrieval-vs-reply.

Every reply is identified by ``(system, query_id, turn)``. A reply scored by
both labellers gets the mean of their two scores. Means are taken over the
replies a system actually gave; fallbacks are counted separately, as correct
(stage B found no relevant passage for the query) or unnecessary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from typing import Any

from sklearn.metrics import cohen_kappa_score

from clearmed.conversation.pipeline import NO_RETRIEVAL
from clearmed.conversation.rubric import CRITERIA
from clearmed.evaluation.retrieval import bootstrap_ci, paired_test
from clearmed.retrieval.judging import LABELLERS

ReplyKey = tuple[str, str, int]
TABLE_CRITERIA = ("correct", "no_unsupported_claims", "grounded", "consistent")


def reply_scores(
    key: Iterable[Mapping[str, Any]], scores: Mapping[str, Mapping[str, int]]
) -> dict[ReplyKey, dict[str, float]]:
    """Per reply and criterion, the mean over the labellers who scored it.

    A shared item (identical reply from several systems) counts for each of them.
    """
    collected: dict[ReplyKey, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for entry in key:
        for criterion, value in scores.get(entry["item_id"], {}).items():
            for system in entry.get("systems", [entry["system"]]):
                collected[(system, entry["query_id"], int(entry["turn"]))][criterion].append(value)
    return {
        reply: {criterion: sum(values) / len(values) for criterion, values in by_criterion.items()}
        for reply, by_criterion in collected.items()
    }


def kappa(key: Iterable[Mapping[str, Any]], scores: Mapping[str, Mapping[str, int]], criterion: str) -> dict[str, Any]:
    """Cohen's kappa (unweighted and quadratic-weighted) on doubly scored replies."""
    pairs: dict[ReplyKey, dict[str, int]] = defaultdict(dict)
    for entry in key:
        value = scores.get(entry["item_id"], {}).get(criterion)
        if entry["shared"] and value is not None:
            pairs[(entry["system"], entry["query_id"], int(entry["turn"]))][entry["labeller"]] = value
    complete = [pair for pair in pairs.values() if len(pair) == len(LABELLERS)]
    first = [pair[LABELLERS[0]] for pair in complete]
    second = [pair[LABELLERS[1]] for pair in complete]
    result: dict[str, Any] = {
        "shared_replies": len(complete),
        "raw_agreement": sum(a == b for a, b in zip(first, second, strict=True)) / len(complete) if complete else None,
        "cohen_kappa": None,
        "weighted_kappa": None,
    }
    if complete and len(set(first) | set(second)) == 1:
        result.update(cohen_kappa=1.0, weighted_kappa=1.0)  # Every shared reply got the same single score.
    elif complete:
        result.update(
            cohen_kappa=float(cohen_kappa_score(first, second)),
            weighted_kappa=float(cohen_kappa_score(first, second, weights="quadratic")),
        )
    return result


def classify_fallbacks(
    turns_by_system: Mapping[str, Iterable[Mapping[str, Any]]], relevant: Mapping[str, set[str]]
) -> dict[str, dict[str, int]]:
    """Count each system's fallbacks as correct, unnecessary or unjudged.

    Only first turns have stage B judgments; second-turn fallbacks are unjudged.
    """
    counts: dict[str, dict[str, int]] = {}
    for system, turns in turns_by_system.items():
        tally = {"replies": 0, "fallbacks": 0, "correct": 0, "unnecessary": 0, "unjudged": 0}
        for turn in turns:
            tally["replies"] += 1
            if not turn["fallback"]:
                continue
            tally["fallbacks"] += 1
            if int(turn["turn"]) != 1 or turn["query_id"] not in relevant:
                tally["unjudged"] += 1
            elif relevant[turn["query_id"]]:
                tally["unnecessary"] += 1
            else:
                tally["correct"] += 1
        counts[system] = tally
    return counts


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    return {"n": len(values), "mean": sum(values) / len(values), "ci95": list(bootstrap_ci(values))}


def system_table(
    replies: Mapping[ReplyKey, Mapping[str, float]], systems: Sequence[str], *, only: set[tuple[str, int]] | None = None
) -> dict[str, dict[str, Any]]:
    """Mean score with 95% bootstrap CI per system and criterion."""
    table: dict[str, dict[str, Any]] = {}
    for system in systems:
        table[system] = {}
        for criterion in TABLE_CRITERIA:
            values = [
                scores[criterion] for (s, query_id, turn), scores in sorted(replies.items())
                if s == system and criterion in scores and (only is None or (query_id, turn) in only)
            ]
            table[system][criterion] = _summary(values)
    return table


def answered_by_all(turns_by_system: Mapping[str, Iterable[Mapping[str, Any]]]) -> set[tuple[str, int]]:
    """(query_id, turn) pairs that every system answered without a fallback."""
    answered = [
        {(t["query_id"], int(t["turn"])) for t in turns if not t["fallback"]} for turns in turns_by_system.values()
    ]
    return set.intersection(*answered) if answered else set()


def pairwise(
    replies: Mapping[ReplyKey, Mapping[str, float]], systems: Sequence[str], common: set[tuple[str, int]] | None = None
) -> dict[str, dict[str, Any]]:
    """Paired tests between systems, each pair on the turns both of them answered.

    ``common`` restricts every pair to the same turns (e.g. those all systems answered).
    """
    results: dict[str, dict[str, Any]] = {}
    turns = common if common is not None else {(query_id, turn) for _, query_id, turn in replies}
    for a, b in combinations(systems, 2):
        results[f"{a}_vs_{b}"] = {}
        for criterion in ("correct", "no_unsupported_claims", "grounded"):
            items = sorted(
                item for item in turns
                if criterion in replies.get((a, *item), {}) and criterion in replies.get((b, *item), {})
            )
            if not items:
                continue
            results[f"{a}_vs_{b}"][criterion] = paired_test(
                [replies[(a, *item)][criterion] for item in items], [replies[(b, *item)][criterion] for item in items]
            )
    return results


def retrieval_vs_reply(
    turns_by_system: Mapping[str, Iterable[Mapping[str, Any]]],
    replies: Mapping[ReplyKey, Mapping[str, float]],
    relevant: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Reply scores for first turns whose evidence held a judged-relevant passage vs not.

    Unjudged passages count as not relevant, as in RQ3.
    """
    result: dict[str, dict[str, Any]] = {}
    for system, turns in turns_by_system.items():
        if system == NO_RETRIEVAL:
            continue
        groups: dict[str, dict[str, list[float]]] = {"relevant_evidence": defaultdict(list),
                                                     "no_relevant_evidence": defaultdict(list)}
        for turn in turns:
            if turn["fallback"] or int(turn["turn"]) != 1 or turn["query_id"] not in relevant:
                continue
            hit = bool(set(turn["passage_ids"]) & relevant[turn["query_id"]])
            scores = replies.get((system, turn["query_id"], 1), {})
            for criterion in ("correct", "no_unsupported_claims", "grounded"):
                if criterion in scores:
                    groups["relevant_evidence" if hit else "no_relevant_evidence"][criterion].append(scores[criterion])
        result[system] = {
            group: {criterion: _summary(values[criterion]) for criterion in ("correct", "no_unsupported_claims", "grounded")}
            for group, values in groups.items()
        }
    return result


def all_kappas(key: Sequence[Mapping[str, Any]], scores: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    return {
        criterion: kappa([e for e in key if e["sheet"] == sheet], scores, criterion)
        for sheet, criteria in CRITERIA.items() for criterion in criteria
    }


def gate_tradeoff(
    gate_off_turns: Iterable[Mapping[str, Any]],
    replies: Mapping[ReplyKey, Mapping[str, float]],
    *,
    system: str,
    frozen_threshold: float,
    criteria: Sequence[str] = ("correct", "no_unsupported_claims", "grounded"),
) -> dict[str, Any]:
    """Coverage vs reply quality for every possible gate threshold (descriptive, post hoc).

    Uses first turns only: there a gated system that accepts gives exactly its
    gate-off twin's reply, so any threshold can be simulated from the graded
    gate-off replies. Each point accepts the turns whose top-1 score is at least
    the threshold and averages their scores; refused turns are fallbacks.
    """
    turns = [t for t in gate_off_turns if int(t["turn"]) == 1 and t["top1_score"] is not None]
    if not turns:
        raise ValueError(f"No first turns with a top-1 score for {system!r}.")
    scored = sorted(((float(t["top1_score"]), replies.get((system, t["query_id"], 1), {})) for t in turns),
                    key=lambda item: -item[0])

    def point(threshold: float) -> dict[str, Any]:
        accepted = [scores for score, scores in scored if score >= threshold]
        means = {}
        for criterion in criteria:
            values = [scores[criterion] for scores in accepted if criterion in scores]
            means[criterion] = sum(values) / len(values) if values else None
        return {"threshold": threshold, "accepted": len(accepted), "coverage": len(accepted) / len(scored), **means}

    return {
        "system": system,
        "n_turns": len(scored),
        "frozen": point(frozen_threshold),
        "curve": [point(threshold) for threshold in sorted({score for score, _ in scored}, reverse=True)],
    }
