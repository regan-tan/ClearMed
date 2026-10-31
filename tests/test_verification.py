import json

import pytest
from datasets import Dataset

from clearmed.data.splitting import membership_fingerprint
from clearmed.data.verification import (
    EVIDENCE_PARTITION,
    find_duplicates,
    load_manifest,
    manifest_entry,
    verify_against_manifest,
)


def make_rows(questions, *, prefix="row"):
    return [
        {"id": f"{prefix}-{index}", "question": question, "cop": 0}
        for index, question in enumerate(questions)
    ]


def manifest_for(**datasets):
    manifest = {"partitions": {}}
    for name, dataset in datasets.items():
        entry = {"rows": len(dataset), "membership_fingerprint": membership_fingerprint(dataset)}
        if name == EVIDENCE_PARTITION:
            manifest["retrieval_evidence"] = entry
        else:
            manifest["partitions"][name] = entry
    return manifest


def test_verify_accepts_exact_partition_and_reads_evidence_key():
    dataset = Dataset.from_list(make_rows(["Q1", "Q2"]))
    manifest = manifest_for(**{EVIDENCE_PARTITION: dataset, "internal_dev": dataset})

    for name in (EVIDENCE_PARTITION, "internal_dev"):
        result = verify_against_manifest(dataset, name, manifest)
        assert result["status"] == "verified"
        assert result["rows"] == 2


def test_verify_rejects_row_count_and_content_changes():
    dataset = Dataset.from_list(make_rows(["Q1", "Q2"]))
    manifest = manifest_for(internal_dev=dataset)

    with pytest.raises(ValueError, match="row count is 1, expected 2"):
        verify_against_manifest(dataset.select([0]), "internal_dev", manifest)
    altered = Dataset.from_list(make_rows(["Q1", "Q2 edited"]))
    with pytest.raises(ValueError, match="fingerprint differs"):
        verify_against_manifest(altered, "internal_dev", manifest)


def test_manifest_entry_rejects_missing_or_incomplete_entries():
    with pytest.raises(ValueError, match="no entry"):
        manifest_entry({"partitions": {}}, "part1_train")
    with pytest.raises(ValueError, match="no entry"):
        manifest_entry({}, EVIDENCE_PARTITION)
    with pytest.raises(ValueError, match="row count or fingerprint"):
        manifest_entry({"partitions": {"part1_train": {"rows": 3}}}, "part1_train")


def test_load_manifest_reports_missing_and_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="generate_splits.py"):
        load_manifest(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_manifest(broken)
    valid = tmp_path / "manifest.json"
    valid.write_text(json.dumps({"seed": 42}), encoding="utf-8")
    assert load_manifest(valid) == {"seed": 42}


def test_find_duplicates_matches_identical_questions_only():
    dataset = Dataset.from_list(
        make_rows(
            [
                "Which drug should be avoided in a patient with asthma?",
                "  which drug should be avoided in a patient   with ASTHMA?",
                "Which drug is contraindicated in asthmatics?",
                "A patient on timolol develops wheezing. Why?",
            ],
            prefix="kb",
        )
    )
    reference = Dataset.from_list(
        make_rows(["Which drug should be avoided in a patient with asthma?"], prefix="val")
    )

    duplicates = find_duplicates(dataset, reference)

    assert [item["id"] for item in duplicates] == ["kb-0", "kb-1"]
    assert all(item["matched_by"] == ["normalized_question"] for item in duplicates)


def test_find_duplicates_matches_shared_row_ids_and_handles_non_text_questions():
    dataset = Dataset.from_list(
        [
            {"id": "shared", "question": "Different wording"},
            {"id": "other", "question": None},
        ]
    )
    reference = Dataset.from_list([{"id": "shared", "question": "Original wording"}])

    duplicates = find_duplicates(dataset, reference)

    assert duplicates == [
        {"index": 0, "id": "shared", "question": "Different wording", "matched_by": ["row_id"]}
    ]
