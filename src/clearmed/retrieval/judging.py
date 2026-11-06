"""Blind relevance-judging pools and label import (decision D3).

Stage A pools each model's top-1 passage for the calibration queries (threshold
calibration). Stage B pools the union of every model's top-5 passages for the
evaluation queries (RQ3). Both stages go into one labelling sheet with a
``labeller`` column. Whole queries are split between the two labellers; a fixed
number of on-topic stage B queries go to both so agreement can be measured.

The sheet never shows which model retrieved a passage or at what rank. That
mapping lives only in the pool key, which scripts read when importing labels.
"""

from __future__ import annotations

import csv
import io
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from random import Random
from typing import Any

from sklearn.metrics import cohen_kappa_score

from clearmed.data.preprocessing import OPTION_LABELS, is_nonblank_text

STAGES = ("A", "B")
LABELLERS = ("labeller_1", "labeller_2")
SHEET_COLUMNS = (
    "item_id",
    "labeller",
    "stage",
    "follow_up",
    "question",
    "correct_answer",
    "reference_explanation",
    "passage",
    "relevant",
    "note",
)
LABEL_VALUES = {"0": 0, "1": 1}


def pool_query(rankings: Mapping[str, Sequence[Mapping[str, Any]]], depth: int) -> dict[str, dict[str, int]]:
    """Union of each method's top-``depth`` docs for one query: doc_id -> {method: rank}."""
    pooled: dict[str, dict[str, int]] = {}
    for method_id in sorted(rankings):
        for hit in rankings[method_id][:depth]:
            pooled.setdefault(hit["doc_id"], {})[method_id] = int(hit["rank"])
    return pooled


def assign_queries(
    query_types: Mapping[str, str], *, shared_ids: Sequence[str], rng: Random
) -> dict[str, tuple[str, ...]]:
    """Split whole queries between labellers, balancing each follow-up type;
    shared queries go to both."""
    shared = set(shared_ids)
    solo = [query_id for query_id in query_types if query_id not in shared]
    rng.shuffle(solo)
    solo.sort(key=lambda query_id: query_types[query_id])  # Stable: random order within each type.
    assignment = {query_id: LABELLERS for query_id in shared_ids}
    for position, query_id in enumerate(solo):
        assignment[query_id] = (LABELLERS[position % len(LABELLERS)],)
    return assignment


def _question_cell(record: Mapping[str, Any]) -> str:
    options = record["options"]
    return record["question"] + "\n" + "\n".join(f"{label}. {options[label]}" for label in OPTION_LABELS)


def build_pool(
    *,
    stage: str,
    queries: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    depth: int,
    shared_count: int,
    rng: Random,
) -> list[dict[str, Any]]:
    """Pool key entries for one stage, one per (labeller, query, passage).

    ``rankings`` maps method_id -> query_id -> ranked results. Shared queries
    are drawn from on-topic queries only: off-topic queries have no relevant
    passage, so agreeing on them would inflate kappa.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}.")
    query_ids = [record["query_id"] for record in queries]
    for method_id, by_query in rankings.items():
        missing = set(query_ids) - set(by_query)
        if missing:
            raise ValueError(f"{method_id} has no rankings for {len(missing)} queries, e.g. {sorted(missing)[:3]}.")
    on_topic = [record["query_id"] for record in queries if record["follow_up_type"] != "off_topic"]
    if shared_count > len(on_topic):
        raise ValueError(f"Cannot share {shared_count} queries; only {len(on_topic)} are on-topic.")
    shared_ids = sorted(rng.sample(on_topic, shared_count))
    assignment = assign_queries(
        {record["query_id"]: record["follow_up_type"] for record in queries}, shared_ids=shared_ids, rng=rng
    )

    entries: list[dict[str, Any]] = []
    for record in queries:
        query_id = record["query_id"]
        pooled = pool_query({method: by_query[query_id] for method, by_query in rankings.items()}, depth)
        for labeller in assignment[query_id]:
            doc_ids = sorted(pooled)
            rng.shuffle(doc_ids)
            for doc_id in doc_ids:
                entries.append(
                    {
                        "stage": stage,
                        "labeller": labeller,
                        "query_id": query_id,
                        "question_id": record["question_id"],
                        "follow_up_type": record["follow_up_type"],
                        "doc_id": doc_id,
                        "retrieved_by": dict(sorted(pooled[doc_id].items())),
                        "shared": len(assignment[query_id]) > 1,
                    }
                )
    return entries


def assign_item_ids(entries: list[dict[str, Any]], rng: Random) -> list[dict[str, Any]]:
    """Give every entry an opaque ID and order the sheet: by labeller, then stage,
    then queries in random order with each query's passages kept together."""
    ids = rng.sample(range(100_000, 1_000_000), len(entries))
    by_group: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for entry, number in zip(entries, ids, strict=True):
        entry["item_id"] = f"{entry['stage']}-{number}"
        by_group[(entry["labeller"], entry["stage"])].setdefault(entry["query_id"], []).append(entry)
    ordered: list[dict[str, Any]] = []
    for group in sorted(by_group):
        query_ids = sorted(by_group[group])
        rng.shuffle(query_ids)
        for query_id in query_ids:
            ordered.extend(by_group[group][query_id])
    return ordered


def sheet_rows(
    entries: Iterable[Mapping[str, Any]],
    *,
    queries: Mapping[str, Mapping[str, Any]],
    passages: Mapping[str, Mapping[str, Any]],
    explanations: Mapping[str, str | None],
) -> list[dict[str, str]]:
    """Rows for the labelling sheet. No method names, ranks or scores."""
    rows = []
    for entry in entries:
        record = queries[entry["query_id"]]
        explanation = explanations.get(record["question_id"])
        rows.append(
            {
                "item_id": entry["item_id"],
                "labeller": entry["labeller"],
                "stage": entry["stage"],
                "follow_up": record["follow_up"],
                "question": _question_cell(record),
                "correct_answer": f"{record['gold_label']}. {record['gold_answer_text']}",
                "reference_explanation": explanation if is_nonblank_text(explanation) else "(none in dataset)",
                "passage": passages[entry["doc_id"]]["text"],
                "relevant": "",
                "note": "",
            }
        )
    return rows


def write_sheet(rows: Sequence[Mapping[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def read_labels(sheet_text: str, key: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate a completed sheet against the pool key; return item_id -> 0/1.

    Rejects missing, extra, duplicated or edited item IDs, rows whose labeller
    or stage was changed, and labels that are blank or not 0/1.
    """
    reader = csv.DictReader(io.StringIO(sheet_text.lstrip("﻿")))
    missing_columns = {"item_id", "labeller", "stage", "relevant"} - set(reader.fieldnames or ())
    if missing_columns:
        raise ValueError(f"Sheet is missing columns: {sorted(missing_columns)}.")
    expected = {entry["item_id"]: entry for entry in key}
    labels: dict[str, int] = {}
    seen: set[str] = set()
    problems: list[str] = []
    for line, row in enumerate(reader, start=2):
        item_id = (row["item_id"] or "").strip()
        entry = expected.get(item_id)
        if entry is None:
            problems.append(f"row {line}: unknown item_id {item_id!r}")
            continue
        if item_id in seen:
            problems.append(f"row {line}: duplicate item_id {item_id}")
            continue
        seen.add(item_id)
        if (row["labeller"] or "").strip() != entry["labeller"] or (row["stage"] or "").strip() != entry["stage"]:
            problems.append(f"row {line}: labeller or stage changed for {item_id}")
            continue
        value = (row["relevant"] or "").strip()
        if value not in LABEL_VALUES:
            problems.append(f"row {line}: {item_id} relevant must be 0 or 1, got {value!r}")
            continue
        labels[item_id] = LABEL_VALUES[value]
    missing = [item_id for item_id in expected if item_id not in seen]
    if missing:
        problems.append(f"{len(missing)} item IDs missing from the sheet, e.g. {missing[:3]}")
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise ValueError(f"Sheet rejected ({len(problems)} problems):\n  {shown}{more}")
    return labels


def agreement(key: Sequence[Mapping[str, Any]], labels: Mapping[str, int]) -> dict[str, Any]:
    """Cohen's kappa on passages that both labellers judged for the same query."""
    pairs: dict[tuple[str, str, str], dict[str, int]] = defaultdict(dict)
    for entry in key:
        if entry["shared"]:
            pairs[(entry["stage"], entry["query_id"], entry["doc_id"])][entry["labeller"]] = labels[entry["item_id"]]
    first = [pair[LABELLERS[0]] for pair in pairs.values()]
    second = [pair[LABELLERS[1]] for pair in pairs.values()]
    disagreements = [
        {"stage": stage, "query_id": query_id, "doc_id": doc_id, **pair}
        for (stage, query_id, doc_id), pair in sorted(pairs.items())
        if pair[LABELLERS[0]] != pair[LABELLERS[1]]
    ]
    if not pairs:
        kappa = None
    elif len(set(first) | set(second)) == 1:
        kappa = 1.0  # Both labellers gave every shared item the same single label.
    else:
        kappa = float(cohen_kappa_score(first, second))
    return {
        "shared_items": len(pairs),
        "shared_queries": len({(stage, query_id) for stage, query_id, _ in pairs}),
        "raw_agreement": (len(pairs) - len(disagreements)) / len(pairs) if pairs else None,
        "cohen_kappa": kappa,
        "disagreements": disagreements,
    }


def final_judgments(
    key: Sequence[Mapping[str, Any]],
    labels: Mapping[str, int],
    adjudicated: Mapping[tuple[str, str, str], int] | None = None,
) -> list[dict[str, Any]]:
    """One label per (stage, query, passage). Shared disagreements need an agreed label."""
    adjudicated = adjudicated or {}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for entry in key:
        grouped[(entry["stage"], entry["query_id"], entry["doc_id"])].append(entry)
    judgments = []
    unresolved = []
    for pool_key, entries in sorted(grouped.items()):
        values = {labels[entry["item_id"]] for entry in entries}
        if len(values) == 1:
            relevant = values.pop()
            source = "both" if len(entries) > 1 else entries[0]["labeller"]
        elif pool_key in adjudicated:
            relevant, source = adjudicated[pool_key], "adjudicated"
        else:
            unresolved.append(pool_key)
            continue
        stage, query_id, doc_id = pool_key
        judgments.append(
            {
                "stage": stage,
                "query_id": query_id,
                "question_id": entries[0]["question_id"],
                "doc_id": doc_id,
                "relevant": relevant,
                "source": source,
                "retrieved_by": entries[0]["retrieved_by"],
            }
        )
    if unresolved:
        raise ValueError(f"{len(unresolved)} shared items disagree and have no agreed label, e.g. {unresolved[:2]}.")
    return judgments
