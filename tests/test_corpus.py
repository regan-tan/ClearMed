import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset

from clearmed.data.splitting import membership_fingerprint
from clearmed.retrieval.corpus import build_corpus, build_passage

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_retrieval_corpus.py"
SCRIPT_SPEC = spec_from_file_location("build_retrieval_corpus_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def mcq(row_id, question, *, cop=2, exp="Propranolol blocks beta-2 receptors."):
    return {
        "id": row_id,
        "question": question,
        "opa": "Amlodipine",
        "opb": "Losartan",
        "opc": "Propranolol",
        "opd": "Hydrochlorothiazide",
        "cop": cop,
        "exp": exp,
        "subject_name": "Pharmacology",
        "topic_name": "Autonomic drugs",
    }


def evidence_rows():
    return [
        mcq("kb-0", "Which drug should be avoided in a patient with asthma?"),
        mcq("kb-1", "A patient on timolol develops wheezing. Why?", cop=0),
        mcq("kb-2", "Which drug is contraindicated in asthmatics?"),
    ]


def evidence_manifest(evidence):
    return {
        "retrieval_evidence": {
            "rows": len(evidence),
            "membership_fingerprint": membership_fingerprint(evidence),
        }
    }


def test_build_passage_keeps_only_question_answer_text_and_explanation():
    passage = build_passage(mcq("kb-0", "  Which drug should be avoided in asthma? "))

    assert passage == {
        "doc_id": "kb-0",
        "text": (
            "Question: Which drug should be avoided in asthma?\n"
            "Answer: Propranolol\n"
            "Explanation: Propranolol blocks beta-2 receptors."
        ),
        "question": "Which drug should be avoided in asthma?",
        "answer": "Propranolol",
        "explanation": "Propranolol blocks beta-2 receptors.",
        "subject": "Pharmacology",
        "topic": "Autonomic drugs",
    }
    for distractor in ("Amlodipine", "Losartan", "Hydrochlorothiazide"):
        assert distractor not in passage["text"]


def test_build_passage_rejects_rows_without_explanation_and_blanks_missing_metadata():
    with pytest.raises(ValueError, match="explanation"):
        build_passage(mcq("kb-0", "Question?", exp=" "))
    row = mcq("kb-1", "Question?")
    row["subject_name"] = " "
    del row["topic_name"]
    passage = build_passage(row)
    assert passage["subject"] is None and passage["topic"] is None


def test_build_corpus_removes_validation_duplicates_and_reports_them():
    evidence = Dataset.from_list(evidence_rows())
    validation = Dataset.from_list(
        [mcq("val-0", "which drug should be AVOIDED in a patient with asthma?")]
    )

    passages, report = build_corpus(
        evidence, manifest=evidence_manifest(evidence), evaluation=validation
    )

    assert [passage["doc_id"] for passage in passages] == ["kb-1", "kb-2"]
    assert report["rows_in"] == 3
    assert report["rows_removed"] == 1
    assert report["passages_out"] == 2
    assert report["removed_examples"] == [
        {
            "id": "kb-0",
            "question": "Which drug should be avoided in a patient with asthma?",
            "matched_by": ["normalized_question"],
        }
    ]
    assert report["manifest_verification"]["status"] == "verified"


def test_build_corpus_with_no_overlap_keeps_every_passage_in_order():
    evidence = Dataset.from_list(evidence_rows())
    validation = Dataset.from_list([mcq("val-0", "An unrelated cardiology question?")])

    passages, report = build_corpus(
        evidence, manifest=evidence_manifest(evidence), evaluation=validation
    )

    assert [passage["doc_id"] for passage in passages] == ["kb-0", "kb-1", "kb-2"]
    assert report["rows_removed"] == 0
    assert report["removed_examples"] == []


def test_build_corpus_refuses_evidence_that_differs_from_manifest():
    evidence = Dataset.from_list(evidence_rows())
    manifest = evidence_manifest(evidence.select([0, 1]))

    with pytest.raises(ValueError, match="does not match the manifest"):
        build_corpus(evidence, manifest=manifest, evaluation=Dataset.from_list([mcq("v", "Q?")]))


def write_splits(splits_dir, *, evidence, internal_dev):
    evidence.save_to_disk(str(splits_dir / "retrieval_kb_evidence"))
    internal_dev.save_to_disk(str(splits_dir / "internal_dev"))
    manifest = {
        "seed": 42,
        "partitions": {
            "internal_dev": {
                "rows": len(internal_dev),
                "membership_fingerprint": membership_fingerprint(internal_dev),
            }
        },
        **evidence_manifest(evidence),
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_script_writes_passages_and_config_and_never_overwrites(tmp_path):
    splits_dir = tmp_path / "splits"
    evidence = Dataset.from_list(evidence_rows())
    write_splits(
        splits_dir,
        evidence=evidence,
        internal_dev=Dataset.from_list([mcq("dev-0", "A dermatology question?")]),
    )
    validation = Dataset.from_list(
        [mcq("val-0", "Which drug should be avoided in a patient with asthma?")]
    )
    output_dir = tmp_path / "run"

    config = SCRIPT_MODULE.run(
        splits_dir=splits_dir, output_dir=output_dir, evaluation=validation, run_id="test-run"
    )

    lines = (output_dir / "passages.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["doc_id"] for line in lines] == ["kb-1", "kb-2"]
    saved = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert saved == config
    assert saved["rows_removed"] == 1
    assert saved["internal_dev_verification"]["status"] == "verified"
    assert saved["evaluation_split_rows"] == 1
    with pytest.raises(FileExistsError, match="not empty"):
        SCRIPT_MODULE.run(
            splits_dir=splits_dir, output_dir=output_dir, evaluation=validation, run_id="test-run"
        )


def test_script_refuses_knowledge_base_that_overlaps_internal_dev(tmp_path):
    splits_dir = tmp_path / "splits"
    evidence = Dataset.from_list(evidence_rows())
    write_splits(
        splits_dir,
        evidence=evidence,
        internal_dev=Dataset.from_list([mcq("dev-0", "A patient on timolol develops wheezing. Why?")]),
    )

    with pytest.raises(ValueError, match="leakage detected"):
        SCRIPT_MODULE.run(
            splits_dir=splits_dir,
            output_dir=tmp_path / "run",
            evaluation=Dataset.from_list([mcq("val-0", "Unrelated?")]),
            run_id="test-run",
        )
    assert not (tmp_path / "run").exists()
