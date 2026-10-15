import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "generate_splits.py"
SCRIPT_SPEC = spec_from_file_location("generate_splits_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)
generate_splits = SCRIPT_MODULE.generate_splits


def make_rows(prefix, count, *, shared_question=None):
    return [
        {
            "id": f"{prefix}-{index}",
            "question": shared_question if index == 0 and shared_question else f"{prefix} q {index}",
            "opa": "A",
            "opb": "B",
            "opc": "C",
            "opd": "D",
            "cop": index % 4,
            "exp": "Explanation" if index % 2 else "",
            "subject_name": "Medicine",
            "topic_name": "Topic",
        }
        for index in range(count)
    ]


def test_generate_splits_persists_artifacts_and_manifest(tmp_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(make_rows("train", 20)),
            "validation": Dataset.from_list(make_rows("validation", 2, shared_question="train q 1")),
            "test": Dataset.from_list(make_rows("test", 2)),
        }
    )

    manifest = generate_splits(dataset, tmp_path / "splits", seed=42)

    manifest_path = tmp_path / "splits" / "manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert len(load_from_disk(str(tmp_path / "splits" / "part1_train"))) > 0
    assert set(load_from_disk(str(tmp_path / "splits" / "retrieval_kb_evidence"))["exp"]) == {
        "Explanation"
    }
    validation_audit = manifest["official_split_overlap_audit"]["validation"]
    assert sum(item["normalized_question_overlap_count"] for item in validation_audit.values()) == 1


def test_generate_splits_refuses_to_overwrite_existing_output(tmp_path):
    output_dir = tmp_path / "splits"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(make_rows("train", 5)),
            "validation": Dataset.from_list(make_rows("validation", 1)),
            "test": Dataset.from_list(make_rows("test", 1)),
        }
    )

    with pytest.raises(FileExistsError, match="not empty"):
        generate_splits(dataset, output_dir)


def test_cli_main_uses_mocked_dataset(tmp_path, monkeypatch):
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(make_rows("train", 8)),
            "validation": Dataset.from_list(make_rows("validation", 1)),
            "test": Dataset.from_list(make_rows("test", 1)),
        }
    )
    output_dir = tmp_path / "cli-splits"
    monkeypatch.setattr(SCRIPT_MODULE, "load_medmcqa", lambda: dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_splits.py", "--seed", "9", "--output-dir", str(output_dir)],
    )

    SCRIPT_MODULE.main()

    assert (output_dir / "manifest.json").exists()
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["seed"] == 9
