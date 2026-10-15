"""Deterministic, duplicate-aware partitions for MedMCQA training data.

The official MedMCQA splits are deliberately not combined here.  This module
only partitions the official training split and reports, rather than changes,
any question overlap with the held-out official splits.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from hashlib import sha256
from random import Random
from typing import Any

from datasets import Dataset

from clearmed.data.preprocessing import (
    is_valid_generation_example,
    is_valid_qa_example,
    normalize_question_for_comparison,
)

DEFAULT_SEED = 42
DEFAULT_RATIOS: dict[str, float] = {
    "part1_train": 0.75,
    "retrieval_kb": 0.20,
    "internal_dev": 0.05,
}


def validate_partition_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    """Validate ratios and return a plain, insertion-ordered dictionary."""
    if not ratios:
        raise ValueError("At least one partition ratio is required.")
    normalized = dict(ratios)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in normalized.values()
    ):
        raise ValueError("Partition ratios must be positive numbers.")
    total = sum(normalized.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Partition ratios must sum to 1.0; received {total}.")
    return {name: float(value) for name, value in normalized.items()}


def _normalized_question_key(row: Mapping[str, Any], row_index: int) -> str:
    """Return a grouping key, isolating non-text question values safely."""
    question = row.get("question")
    if isinstance(question, str):
        return normalize_question_for_comparison(question)
    return f"__non_text_question_row_{row_index}__"


def duplicate_question_groups(dataset: Dataset) -> dict[str, list[int]]:
    """Return source-row indices grouped by normalized question text.

    Non-string questions are made singleton groups.  This preserves invalid
    source rows without treating unrelated missing values as duplicate text.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        groups[_normalized_question_key(row, index)].append(index)
    return dict(groups)


def partition_training_dataset(
    dataset: Dataset,
    *,
    seed: int = DEFAULT_SEED,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
) -> dict[str, Dataset]:
    """Partition one training dataset while keeping duplicate questions together.

    Question groups are deterministically shuffled using ``seed``.  Each group
    is greedily assigned to the partition that minimizes overall squared
    distance from the requested ratios.  This favours the requested 75/20/5
    balance while never splitting an identical normalized question.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    validated_ratios = validate_partition_ratios(ratios)
    groups = duplicate_question_groups(dataset)
    group_items = sorted(groups.items(), key=lambda item: item[0])
    Random(seed).shuffle(group_items)

    names = tuple(validated_ratios)
    total_rows = len(dataset)
    targets = {name: total_rows * validated_ratios[name] for name in names}
    assigned_indices = {name: [] for name in names}
    assigned_sizes = {name: 0 for name in names}

    for _, indices in group_items:
        group_size = len(indices)

        def score(
            name: str, *, current_group_size: int = group_size
        ) -> tuple[float, int]:
            return (
                sum(
                    (
                        (
                            (assigned_sizes[candidate] + current_group_size)
                            if candidate == name
                            else assigned_sizes[candidate]
                        )
                        - targets[candidate]
                    )
                    ** 2
                    for candidate in names
                ),
                names.index(name),
            )

        destination = min(names, key=score)
        assigned_indices[destination].extend(indices)
        assigned_sizes[destination] += group_size

    partitions = {
        name: dataset.select(sorted(indices)) for name, indices in assigned_indices.items()
    }
    assert_partition_integrity(dataset, partitions)
    return partitions


def _row_id(row: Mapping[str, Any]) -> str:
    if "id" not in row:
        raise ValueError("Dataset rows must contain an 'id' field for split auditing.")
    return str(row["id"])


def _question_index(dataset: Dataset) -> dict[str, set[str]]:
    """Index string questions by their normalized value and row IDs."""
    indexed: dict[str, set[str]] = defaultdict(set)
    for row in dataset:
        question = row.get("question")
        if isinstance(question, str):
            indexed[normalize_question_for_comparison(question)].add(_row_id(row))
    return dict(indexed)


def assert_partition_integrity(
    source: Dataset, partitions: Mapping[str, Dataset]
) -> None:
    """Raise ``ValueError`` when coverage, IDs, or question isolation fail."""
    if not partitions:
        raise ValueError("No partitions were supplied.")

    source_ids = Counter(_row_id(row) for row in source)
    partition_ids: Counter[str] = Counter()
    id_partitions: dict[str, set[str]] = defaultdict(set)
    question_partitions: dict[str, set[str]] = defaultdict(set)

    for name, partition in partitions.items():
        for row in partition:
            row_id = _row_id(row)
            partition_ids[row_id] += 1
            id_partitions[row_id].add(name)
            question = row.get("question")
            if isinstance(question, str):
                question_partitions[normalize_question_for_comparison(question)].add(name)

    if sum(len(partition) for partition in partitions.values()) != len(source):
        raise ValueError("Partition coverage failed: row counts do not match source data.")
    if partition_ids != source_ids:
        raise ValueError("Partition coverage failed: partition IDs do not match source IDs.")
    overlapping_ids = [row_id for row_id, names in id_partitions.items() if len(names) > 1]
    if overlapping_ids:
        raise ValueError("Partition exclusivity failed: a row ID occurs more than once.")
    leaked_questions = [
        question for question, names in question_partitions.items() if len(names) > 1
    ]
    if leaked_questions:
        raise ValueError(
            "Duplicate-question isolation failed: normalized questions cross partitions."
        )


def filter_retrieval_evidence(dataset: Dataset) -> Dataset:
    """Keep only QA-valid rows with a usable expert explanation."""
    return dataset.filter(is_valid_generation_example)


def membership_fingerprint(dataset: Dataset) -> str:
    """Return an order-independent SHA-256 fingerprint of partition contents."""
    serialized_rows = [
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
        for row in dataset
    ]
    payload = "\n".join(sorted(serialized_rows)).encode("utf-8")
    return sha256(payload).hexdigest()


def audit_dataset_overlap(
    partitions: Mapping[str, Dataset], comparison: Dataset
) -> dict[str, dict[str, Any]]:
    """Report row-ID and normalized-question overlaps with another split."""
    comparison_ids = {_row_id(row) for row in comparison}
    comparison_questions = _question_index(comparison)
    report: dict[str, dict[str, Any]] = {}
    for name, partition in partitions.items():
        partition_ids = {_row_id(row) for row in partition}
        partition_questions = _question_index(partition)
        shared_ids = sorted(partition_ids & comparison_ids)
        shared_questions = sorted(set(partition_questions) & set(comparison_questions))
        report[name] = {
            "row_id_overlap_count": len(shared_ids),
            "row_id_examples": shared_ids[:10],
            "normalized_question_overlap_count": len(shared_questions),
            "normalized_question_examples": shared_questions[:10],
        }
    return report


def partition_statistics(dataset: Dataset) -> dict[str, int | float | str]:
    """Calculate validity, duplicate, and fingerprint statistics for a dataset."""
    groups = duplicate_question_groups(dataset)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    qa_valid = sum(is_valid_qa_example(row) for row in dataset)
    explanation_valid = sum(is_valid_generation_example(row) for row in dataset)
    return {
        "rows": len(dataset),
        "qa_valid_rows": qa_valid,
        "explanation_valid_rows": explanation_valid,
        "unique_normalized_questions": len(groups),
        "duplicate_normalized_question_groups": len(duplicate_groups),
        "rows_in_duplicate_question_groups": sum(len(group) for group in duplicate_groups),
        "membership_fingerprint": membership_fingerprint(dataset),
    }


def build_manifest(
    *,
    dataset_name: str,
    source_train: Dataset,
    partitions: Mapping[str, Dataset],
    retrieval_evidence: Dataset,
    seed: int,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    validation: Dataset | None = None,
    test: Dataset | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable reproducibility and leakage-audit manifest."""
    validated_ratios = validate_partition_ratios(ratios)
    assert_partition_integrity(source_train, partitions)
    total = len(source_train)
    partition_stats: dict[str, dict[str, int | float | str]] = {}
    for name, partition in partitions.items():
        stats = partition_statistics(partition)
        stats["percentage"] = (len(partition) / total * 100) if total else 0.0
        stats["configured_ratio"] = validated_ratios[name]
        partition_stats[name] = stats

    internal_overlap = {
        name: {"row_id_overlap_count": 0, "normalized_question_overlap_count": 0}
        for name in partitions
    }
    names = list(partitions)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = partitions[left_name]
            right = partitions[right_name]
            left_ids = {_row_id(row) for row in left}
            right_ids = {_row_id(row) for row in right}
            left_questions = set(_question_index(left))
            right_questions = set(_question_index(right))
            internal_overlap[left_name][right_name] = {
                "row_id_overlap_count": len(left_ids & right_ids),
                "normalized_question_overlap_count": len(left_questions & right_questions),
            }

    cross_overlap = {
        "validation": audit_dataset_overlap(partitions, validation)
        if validation is not None
        else None,
        "test": audit_dataset_overlap(partitions, test) if test is not None else None,
    }
    return {
        "dataset_name": dataset_name,
        "source_metadata": dict(source_metadata or {}),
        "seed": seed,
        "ratios": validated_ratios,
        "source_train_rows": total,
        "source_train_fingerprint": membership_fingerprint(source_train),
        "partitions": partition_stats,
        "retrieval_evidence": {
            "rows": len(retrieval_evidence),
            "membership_fingerprint": membership_fingerprint(retrieval_evidence),
        },
        "source_duplicate_question_statistics": {
            key: value
            for key, value in partition_statistics(source_train).items()
            if key
            in {
                "unique_normalized_questions",
                "duplicate_normalized_question_groups",
                "rows_in_duplicate_question_groups",
            }
        },
        "internal_overlap_audit": internal_overlap,
        "official_split_overlap_audit": cross_overlap,
    }
