import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset

from clearmed.data.splitting import membership_fingerprint
from clearmed.training.seq2seq import load_checkpoint


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = spec_from_file_location(f"{name}_for_test", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAIN = load_script("train_seq2seq")
EVALUATE = load_script("evaluate_seq2seq")
INSPECT = load_script("inspect_seq2seq_data")

DRUGS = ["aspirin", "insulin", "salbutamol", "warfarin", "metformin", "heparin"]


def rows(count, *, start=0):
    out = []
    for index in range(start, start + count):
        options = [DRUGS[(index + k) % len(DRUGS)] for k in range(4)]
        cop = index % 4
        out.append(
            {
                "id": f"row-{index}",
                "question": f"Case {index}: which drug is {options[cop]}?",
                "opa": options[0], "opb": options[1], "opc": options[2], "opd": options[3],
                "cop": cop,
                "exp": f"{options[cop]} is correct." if index % 2 else "",
                "subject_name": "Pharmacology",
                "topic_name": "Topic",
            }
        )
    return out


def save_partitions(tmp_path, train_rows, dev_rows):
    train_dir, dev_dir = tmp_path / "train", tmp_path / "dev"
    Dataset.from_list(train_rows).save_to_disk(str(train_dir))
    Dataset.from_list(dev_rows).save_to_disk(str(dev_dir))
    return train_dir, dev_dir


def train_args(tmp_path, train_dir, dev_dir, *extra):
    return TRAIN.build_parser().parse_args(
        [
            "--train-dir", str(train_dir), "--dev-dir", str(dev_dir),
            "--checkpoint-root", str(tmp_path / "ckpt"), "--output-root", str(tmp_path / "out"),
            "--run-id", "smoke", "--device", "cpu", "--min-freq", "1",
            "--embedding-dim", "16", "--hidden-dim", "16", "--attention-dim", "16",
            "--batch-size", "4", "--max-explanation-tokens", "8",
            *extra,
        ]
    )


def test_train_then_evaluate_end_to_end(tmp_path):
    train_dir, dev_dir = save_partitions(tmp_path, rows(16), rows(6, start=100))
    result = TRAIN.run_training(train_args(tmp_path, train_dir, dev_dir, "--max-epochs", "2", "--max-train-questions", "12"))

    out = tmp_path / "out" / "smoke"
    config = json.loads((out / "config.json").read_text())
    assert config["pretrained_weights"] is None
    assert config["manifest_verification"]["status"] == "inapplicable"
    assert config["train_data_statistics"]["examples"] == 12
    history = json.loads((out / "training_history.json").read_text())
    assert [h["epoch"] for h in history] == [1, 2]
    assert result["training_summary"]["best"]["epoch"] in (1, 2)

    best_dir = tmp_path / "ckpt" / "smoke" / "best"
    _, vocab, _, metadata = load_checkpoint(best_dir)
    assert metadata["extra"]["selected_epoch"] == result["training_summary"]["best"]["epoch"]
    assert "row-100" not in vocab.stoi  # sanity: ids are never tokens

    eval_args = EVALUATE.build_parser().parse_args(
        [
            "--checkpoint-dir", str(best_dir), "--output-dir", str(tmp_path / "eval"),
            "--train-dir", str(train_dir), "--dev-dir", str(dev_dir), "--device", "cpu",
        ]
    )
    metrics = EVALUATE.run_evaluation(eval_args)
    mcq = metrics["mcq_metrics"]
    assert mcq["total_evaluated"] == 6
    assert mcq["correct"] + mcq["incorrect"] == 6
    lines = (tmp_path / "eval" / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == 6
    record = json.loads(lines[0])
    assert set(record) >= {"example_id", "gold_label", "predicted_label", "raw_output", "generated_explanation"}
    assert record["predicted_label"] in {"A", "B", "C", "D", "MALFORMED"}
    assert metrics["explanation_overlap"]["rows_with_reference"] == 3


def test_resume_continues_and_matches_uninterrupted_run(tmp_path):
    train_dir, dev_dir = save_partitions(tmp_path, rows(12), rows(4, start=100))
    full = TRAIN.run_training(train_args(tmp_path / "a", train_dir, dev_dir, "--max-epochs", "2", "--patience", "5"))

    TRAIN.run_training(train_args(tmp_path / "b", train_dir, dev_dir, "--max-epochs", "1", "--patience", "5"))
    resumed = TRAIN.run_training(
        train_args(tmp_path / "b", train_dir, dev_dir, "--max-epochs", "2", "--patience", "5", "--resume")
    )
    full_history = json.loads((tmp_path / "a" / "out" / "smoke" / "training_history.json").read_text())
    resumed_history = json.loads((tmp_path / "b" / "out" / "smoke" / "training_history.json").read_text())
    assert [h["epoch"] for h in resumed_history] == [1, 2]
    assert resumed_history[-1]["train_loss"] == pytest.approx(full_history[-1]["train_loss"], rel=1e-5)
    assert resumed["training_summary"]["best"]["epoch"] == full["training_summary"]["best"]["epoch"]


def test_early_stopping_halts_training(tmp_path):
    train_dir, dev_dir = save_partitions(tmp_path, rows(8), rows(4, start=100))
    # lr=0 => nothing improves after epoch 1, so patience 1 stops after epoch 2.
    result = TRAIN.run_training(
        train_args(tmp_path, train_dir, dev_dir, "--max-epochs", "10", "--patience", "1", "--learning-rate", "0")
    )
    assert result["training_summary"]["stopped_early"]
    assert result["training_summary"]["epochs_run"] == 2


def test_train_rejects_leaky_partitions_before_writing(tmp_path):
    dev = rows(4, start=100)
    dev[0]["question"] = rows(1)[0]["question"].upper()
    train_dir, dev_dir = save_partitions(tmp_path, rows(8), dev)
    with pytest.raises(ValueError, match="Dataset leakage detected"):
        TRAIN.run_training(train_args(tmp_path, train_dir, dev_dir))
    assert not (tmp_path / "ckpt").exists() and not (tmp_path / "out").exists()


def test_train_refuses_to_overwrite_a_run(tmp_path):
    train_dir, dev_dir = save_partitions(tmp_path, rows(8), rows(4, start=100))
    TRAIN.run_training(train_args(tmp_path, train_dir, dev_dir, "--max-epochs", "1"))
    with pytest.raises(FileExistsError):
        TRAIN.run_training(train_args(tmp_path, train_dir, dev_dir, "--max-epochs", "1"))


def test_canonical_manifest_mismatch_blocks_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    splits = tmp_path / "data" / "splits"
    train, dev = Dataset.from_list(rows(8)), Dataset.from_list(rows(4, start=100))
    train.save_to_disk(str(splits / "part1_train"))
    dev.save_to_disk(str(splits / "internal_dev"))
    (splits / "manifest.json").write_text(
        json.dumps(
            {
                "partitions": {
                    "part1_train": {"rows": len(train), "membership_fingerprint": membership_fingerprint(train)},
                    "internal_dev": {"rows": len(dev), "membership_fingerprint": "tampered"},
                }
            }
        )
    )
    args = TRAIN.build_parser().parse_args(["--max-train-questions", "2", "--device", "cpu", "--run-id", "x"])
    with pytest.raises(ValueError, match="manifest verification failed"):
        TRAIN.run_training(args)
    assert not (tmp_path / "checkpoints").exists()


def test_inspection_reports_coverage_and_lengths():
    report = INSPECT.inspect_rows(rows(10) + [{"id": "bad", "question": ""}], INSPECT.Seq2SeqDataConfig())
    assert report["rows"] == 11 and report["qa_valid_rows"] == 10
    assert report["rows_with_explanation"] == 5
    assert report["explanation_coverage"] == 0.5
    assert report["explanation_tokens"]["count"] == 5
