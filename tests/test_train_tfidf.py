import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset

from clearmed.baselines.tfidf import TfidfMCQBaseline
from clearmed.data.splitting import membership_fingerprint

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "train_tfidf.py"
SCRIPT_SPEC = spec_from_file_location("train_tfidf_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def rows(count, *, start=0):
    return [
        {
            "id": f"row-{index}",
            "question": f"What treats diagnosis {index}?",
            "opa": f"treatment {index} alpha",
            "opb": f"treatment {index} beta",
            "opc": f"treatment {index} gamma",
            "opd": f"treatment {index} delta",
            "cop": index % 4,
            "exp": "",
            "subject_name": "Medicine" if index % 2 else "Surgery",
            "topic_name": "Topic",
        }
        for index in range(start, start + count)
    ]


def test_cli_runs_from_saved_synthetic_partitions(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    dev_dir = tmp_path / "dev"
    Dataset.from_list(rows(12)).save_to_disk(str(train_dir))
    Dataset.from_list(rows(6, start=100)).save_to_disk(str(dev_dir))
    model_dir = tmp_path / "model"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tfidf.py", "--train-dir", str(train_dir), "--dev-dir", str(dev_dir),
            "--model-dir", str(model_dir), "--output-dir", str(output_dir),
            "--max-train-questions", "8", "--max-dev-questions", "4", "--min-df", "1",
        ],
    )

    SCRIPT_MODULE.main()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["training_question_count"] == 8
    assert metrics["selected_development_questions"] == 4
    assert metrics["saved_model_reload_verified"]
    assert metrics["manifest_verification"]["status"] == "inapplicable"
    assert metrics["manifest_verification"]["reason"] == "custom_dataset_paths"
    assert "inapplicable" in metrics["manifest_verification"]["message"]
    assert (output_dir / "subject_accuracy.csv").is_file()
    assert TfidfMCQBaseline.load(model_dir).predict_mcq(rows(1)[0])["predicted_answer"] in "ABCD"


def test_cli_missing_partition_has_actionable_error(tmp_path):
    args = SCRIPT_MODULE.build_parser().parse_args(
        ["--train-dir", str(tmp_path / "missing-train"), "--dev-dir", str(tmp_path / "missing-dev")]
    )
    with pytest.raises(FileNotFoundError, match="generate_splits"):
        SCRIPT_MODULE.run_experiment(args)


def test_cli_rejects_overlapping_training_and_development_data(tmp_path):
    train_dir = tmp_path / "train"
    dev_dir = tmp_path / "dev"
    Dataset.from_list(rows(8)).save_to_disk(str(train_dir))
    overlapping = rows(4, start=100)
    overlapping[0]["id"] = "row-0"
    overlapping[1]["question"] = rows(1)[0]["question"].upper()
    Dataset.from_list(overlapping).save_to_disk(str(dev_dir))
    args = SCRIPT_MODULE.build_parser().parse_args(
        ["--train-dir", str(train_dir), "--dev-dir", str(dev_dir), "--model-dir", str(tmp_path / "model"), "--output-dir", str(tmp_path / "output")]
    )

    with pytest.raises(ValueError, match="Dataset leakage detected"):
        SCRIPT_MODULE.run_experiment(args)
    assert not (tmp_path / "model").exists()
    assert not (tmp_path / "output").exists()


def _write_canonical_partitions(tmp_path, monkeypatch, *, invalid_manifest=False):
    monkeypatch.chdir(tmp_path)
    splits_dir = tmp_path / "data" / "splits"
    train = Dataset.from_list(rows(4))
    dev = Dataset.from_list(rows(4, start=100))
    train.save_to_disk(str(splits_dir / "part1_train"))
    dev.save_to_disk(str(splits_dir / "internal_dev"))
    manifest_path = splits_dir / "manifest.json"
    manifest_path.write_text(
        "not JSON"
        if invalid_manifest
        else json.dumps(
            {
                "partitions": {
                    "part1_train": {
                        "rows": len(train),
                        "membership_fingerprint": membership_fingerprint(train),
                    },
                    "internal_dev": {
                        "rows": len(dev),
                        "membership_fingerprint": membership_fingerprint(dev),
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    return train, dev, splits_dir, manifest_path


def test_canonical_manifest_override_is_rejected(tmp_path, monkeypatch):
    train, dev, splits_dir, _ = _write_canonical_partitions(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="supplied --manifest"):
        SCRIPT_MODULE._verify_manifest(
            train=train,
            development=dev,
            train_dir=splits_dir / "part1_train",
            dev_dir=splits_dir / "internal_dev",
            manifest_path=tmp_path / "other-manifest.json",
        )


def test_missing_canonical_manifest_is_rejected(tmp_path, monkeypatch):
    train, dev, splits_dir, manifest_path = _write_canonical_partitions(tmp_path, monkeypatch)
    manifest_path.unlink()

    with pytest.raises(FileNotFoundError, match="manifest verification"):
        SCRIPT_MODULE._verify_manifest(
            train=train,
            development=dev,
            train_dir=splits_dir / "part1_train",
            dev_dir=splits_dir / "internal_dev",
            manifest_path=manifest_path,
        )


def test_invalid_canonical_manifest_is_rejected(tmp_path, monkeypatch):
    train, dev, splits_dir, manifest_path = _write_canonical_partitions(
        tmp_path, monkeypatch, invalid_manifest=True
    )

    with pytest.raises(ValueError, match="not valid JSON"):
        SCRIPT_MODULE._verify_manifest(
            train=train,
            development=dev,
            train_dir=splits_dir / "part1_train",
            dev_dir=splits_dir / "internal_dev",
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("train_limit", "dev_limit"),
    [("1", "1"), ("4", "4"), ("99", "99")],
    ids=["small-subset", "no-op-limit", "oversized-limit"],
)
def test_canonical_manifest_verification_precedes_all_limits(
    tmp_path, monkeypatch, train_limit, dev_limit
):
    _, _, _, manifest_path = _write_canonical_partitions(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"]["internal_dev"]["membership_fingerprint"] = "mismatched"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    args = SCRIPT_MODULE.build_parser().parse_args(
        [
            "--max-train-questions",
            train_limit,
            "--max-dev-questions",
            dev_limit,
            "--model-dir",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    with pytest.raises(ValueError, match="manifest verification failed"):
        SCRIPT_MODULE.run_experiment(args)
    assert not (tmp_path / "model").exists()
    assert not (tmp_path / "output").exists()
