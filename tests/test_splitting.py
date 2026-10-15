import pytest
from datasets import Dataset

from clearmed.data.splitting import (
    DEFAULT_RATIOS,
    assert_partition_integrity,
    audit_dataset_overlap,
    filter_retrieval_evidence,
    partition_training_dataset,
)


def make_dataset(size=60):
    rows = []
    for index in range(size):
        question_number = 0 if index in (0, 1) else index
        rows.append(
            {
                "id": f"row-{index}",
                "question": f"Question {question_number}",
                "opa": "A",
                "opb": "B",
                "opc": "C",
                "opd": "D",
                "cop": index % 4,
                "exp": "Explanation" if index % 3 else " ",
                "subject_name": "Medicine",
                "topic_name": "Topic",
            }
        )
    return Dataset.from_list(rows)


def partition_ids(partitions):
    return {name: tuple(dataset["id"]) for name, dataset in partitions.items()}


def test_partition_is_deterministic_and_duplicate_aware():
    dataset = make_dataset()
    first = partition_training_dataset(dataset, seed=42)
    second = partition_training_dataset(dataset, seed=42)

    assert partition_ids(first) == partition_ids(second)
    assert_partition_integrity(dataset, first)
    duplicate_membership = [
        name for name, partition in first.items() if {"row-0", "row-1"} & set(partition["id"])
    ]
    assert duplicate_membership == [
        name
        for name, partition in first.items()
        if {"row-0", "row-1"}.issubset(partition["id"])
    ]


def test_partition_uses_all_rows_and_is_approximately_proportional():
    dataset = make_dataset(100)
    partitions = partition_training_dataset(dataset, seed=7)

    assert sum(len(partition) for partition in partitions.values()) == len(dataset)
    assert len(set().union(*(set(partition["id"]) for partition in partitions.values()))) == len(
        dataset
    )
    for name, ratio in DEFAULT_RATIOS.items():
        assert abs(len(partitions[name]) - len(dataset) * ratio) <= 3


def test_different_seed_remains_valid_and_can_change_membership():
    dataset = make_dataset()
    first = partition_training_dataset(dataset, seed=1)
    second = partition_training_dataset(dataset, seed=2)

    assert_partition_integrity(dataset, first)
    assert_partition_integrity(dataset, second)
    assert partition_ids(first) != partition_ids(second)


def test_retrieval_evidence_requires_qa_content_and_explanation():
    dataset = make_dataset(4)
    rows = dataset.to_list()
    rows[1]["opa"] = ""
    rows[2]["exp"] = None
    rows[3]["exp"] = "Explanation"
    evidence = filter_retrieval_evidence(Dataset.from_list(rows))

    assert evidence["id"] == ["row-3"]


def test_primary_partitions_keep_rows_without_explanations():
    dataset = make_dataset(12)
    partitions = partition_training_dataset(dataset)
    all_rows = [row for partition in partitions.values() for row in partition]

    assert any(row["exp"].strip() == "" for row in all_rows)


def test_empty_dataset_and_invalid_configuration():
    empty = Dataset.from_dict({"id": [], "question": []})
    partitions = partition_training_dataset(empty)
    assert {name: len(partition) for name, partition in partitions.items()} == {
        "part1_train": 0,
        "retrieval_kb": 0,
        "internal_dev": 0,
    }
    with pytest.raises(ValueError, match="sum to 1.0"):
        partition_training_dataset(make_dataset(), ratios={"a": 0.5, "b": 0.4})
    with pytest.raises(TypeError, match="seed"):
        partition_training_dataset(make_dataset(), seed=True)


def test_overlap_audit_reports_official_validation_overlap():
    partitions = partition_training_dataset(make_dataset(12))
    validation = Dataset.from_list(
        [
            {
                "id": "row-0",
                "question": " question 0 ",
                "opa": "A",
                "opb": "B",
                "opc": "C",
                "opd": "D",
                "cop": 0,
                "exp": "Explanation",
            }
        ]
    )
    audit = audit_dataset_overlap(partitions, validation)

    assert sum(item["row_id_overlap_count"] for item in audit.values()) == 1
    assert sum(item["normalized_question_overlap_count"] for item in audit.values()) == 1
