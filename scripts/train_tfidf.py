"""Train and evaluate the reproducible TF-IDF medical MCQ baseline.

The runner reads only saved Step 1 partitions; it never downloads MedMCQA or
constructs new train/development splits.  Its primary metric is one predicted
answer per original MCQ, not binary candidate-pair accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import scipy
import sklearn
from datasets import Dataset, load_from_disk

from clearmed.baselines.tfidf import TfidfMCQBaseline, TfidfMCQConfig
from clearmed.data.preprocessing import OPTION_FIELDS
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.qa import evaluate_mcq_predictions


def _canonical_partition_paths() -> tuple[Path, Path, Path]:
    """Return canonical split artifacts relative to the project working directory."""
    splits_dir = Path("data/splits").resolve()
    return (
        splits_dir / "part1_train",
        splits_dir / "internal_dev",
        splits_dir / "manifest.json",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _load_partition(path: Path, name: str) -> Dataset:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing {name} partition at {path}. Generate local saved partitions first with "
            "`uv run python scripts/generate_splits.py --seed 42`."
        )
    try:
        dataset = load_from_disk(str(path))
    except Exception as error:  # datasets gives several storage-specific errors
        raise RuntimeError(f"Could not load saved {name} partition at {path}: {error}") from error
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Saved {name} partition must be a Hugging Face Dataset, not {type(dataset)}.")
    required = {"id", "question", *OPTION_FIELDS, "cop"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"{name} partition is missing required QA columns: {', '.join(missing)}")
    return dataset


def _subset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    """Deterministically choose MCQs before four-way candidate expansion."""
    if limit is None or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def _prepare_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} is not empty. Choose a new location or pass --overwrite."
        )
    path.mkdir(parents=True, exist_ok=True)


def _verify_manifest(
    *,
    train: Dataset,
    development: Dataset,
    train_dir: Path,
    dev_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Verify complete canonical partitions or report custom-path inapplicability."""
    canonical_train, canonical_dev, canonical_manifest = _canonical_partition_paths()
    uses_canonical_train = train_dir.resolve() == canonical_train
    uses_canonical_dev = dev_dir.resolve() == canonical_dev
    if uses_canonical_train != uses_canonical_dev:
        raise ValueError(
            "Canonical partition paths must be used together. Use both data/splits/part1_train "
            "and data/splits/internal_dev with data/splits/manifest.json, or provide two "
            "genuinely custom partition directories."
        )
    if not uses_canonical_train:
        return {
            "status": "inapplicable",
            "reason": "custom_dataset_paths",
            "message": "Canonical manifest verification is inapplicable for custom dataset paths.",
            "manifest_path": str(manifest_path),
        }
    if manifest_path.resolve() != canonical_manifest:
        raise ValueError(
            "Canonical partitions require data/splits/manifest.json. The supplied --manifest "
            f"path ({manifest_path}) conflicts with the canonical manifest ({canonical_manifest}). "
            "Remove --manifest or use two genuinely custom partition directories."
        )
    if not canonical_manifest.is_file():
        raise FileNotFoundError(
            f"Canonical partitions require manifest verification, but {canonical_manifest} is missing. "
            "Restore it or use explicit custom dataset paths."
        )
    try:
        manifest = json.loads(canonical_manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest {canonical_manifest} is not valid JSON: {error}") from error

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise TypeError(
            f"Manifest {canonical_manifest} has no partitions object for canonical verification."
        )
    actual_partitions = {"part1_train": train, "internal_dev": development}
    verification: dict[str, Any] = {}
    mismatches: list[str] = []
    for name, dataset in actual_partitions.items():
        expected = partitions.get(name)
        if not isinstance(expected, dict):
            raise TypeError(
                f"Manifest {canonical_manifest} is missing metadata for canonical partition {name!r}."
            )
        expected_rows = expected.get("rows")
        expected_fingerprint = expected.get("membership_fingerprint")
        if not isinstance(expected_rows, int) or not isinstance(expected_fingerprint, str):
            raise TypeError(
                f"Manifest {canonical_manifest} has incomplete row-count or fingerprint metadata for {name!r}."
            )
        actual_rows = len(dataset)
        actual_fingerprint = membership_fingerprint(dataset)
        verification[name] = {
            "expected_rows": expected_rows,
            "actual_rows": actual_rows,
            "expected_membership_fingerprint": expected_fingerprint,
            "actual_membership_fingerprint": actual_fingerprint,
        }
        if actual_rows != expected_rows:
            mismatches.append(f"{name} row count is {actual_rows}, expected {expected_rows}")
        if actual_fingerprint != expected_fingerprint:
            mismatches.append(f"{name} membership fingerprint differs from manifest")
    if mismatches:
        raise ValueError(
            "Canonical partition manifest verification failed: "
            + "; ".join(mismatches)
            + ". Do not train from altered data/splits artifacts; restore them or pass explicit custom paths."
        )
    return {
        "status": "verified",
        "manifest_path": str(canonical_manifest),
        "dataset_name": manifest.get("dataset_name"),
        "seed": manifest.get("seed"),
        "partitions": verification,
    }


def _write_outputs(output_dir: Path, metrics: dict[str, Any], evaluation: dict[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "subject_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_name", "total", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(evaluation["subject_accuracy"])
    with (output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in evaluation["errors"]:
            handle.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Run a local experiment and return its serializable metrics payload."""
    complete_train = _load_partition(args.train_dir, "part1_train")
    complete_dev = _load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(
        complete_train,
        complete_dev,
        train_name="part1_train",
        development_name="internal_dev",
    )
    manifest_verification = _verify_manifest(
        train=complete_train,
        development=complete_dev,
        train_dir=args.train_dir,
        dev_dir=args.dev_dir,
        manifest_path=args.manifest,
    )
    train = _subset(complete_train, args.max_train_questions, args.seed)
    dev = _subset(complete_dev, args.max_dev_questions, args.seed)
    _prepare_directory(args.model_dir, overwrite=args.overwrite)
    _prepare_directory(args.output_dir, overwrite=args.overwrite)

    config = TfidfMCQConfig(
        min_df=args.min_df,
        max_features=args.max_features,
        random_state=args.seed,
        batch_size=args.batch_size,
    )
    model = TfidfMCQBaseline(config)
    print(f"Training on {len(train):,} selected MCQs (four candidate pairs each)...")
    training_start = perf_counter()
    model.fit(train)
    training_seconds = perf_counter() - training_start
    model.save(args.model_dir, overwrite=args.overwrite)

    print(f"Evaluating {len(dev):,} selected internal-development MCQs...")
    evaluation_start = perf_counter()
    evaluation = evaluate_mcq_predictions(dev, model.predict_mcq, error_limit=args.error_limit)
    evaluation_seconds = perf_counter() - evaluation_start

    reload_verified = True
    if len(dev):
        reloaded = TfidfMCQBaseline.load(args.model_dir)
        original = model.predict_mcq(dev[0])
        restored = reloaded.predict_mcq(dev[0])
        reload_verified = original == restored
        if not reload_verified:
            raise RuntimeError("Saved and reloaded TF-IDF model produced different predictions.")

    evaluation_metrics = {key: value for key, value in evaluation.items() if key not in {"errors", "subject_accuracy"}}
    metrics: dict[str, Any] = {
        "model_name": "TF-IDF binary candidate correctness baseline",
        "model_type": "tfidf_binary_logistic_regression",
        "dataset_partitions": {"train": "part1_train", "development": "internal_dev"},
        "selected_train_questions": len(train),
        "selected_development_questions": len(dev),
        "training_question_count": model.training_question_count,
        "training_candidate_pair_count": model.training_candidate_pair_count,
        "tfidf_vocabulary_size": model.vocabulary_size,
        "effective_min_df": model.effective_min_df,
        "seed": args.seed,
        "model_config": asdict(config),
        "limits": {
            "max_train_questions": args.max_train_questions,
            "max_dev_questions": args.max_dev_questions,
        },
        "durations_seconds": {"training": training_seconds, "evaluation": evaluation_seconds},
        "saved_model_reload_verified": reload_verified,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "manifest_verification": manifest_verification,
        "development_metrics": evaluation_metrics,
        "notes": {
            "primary_metric": "Question-level MCQ answer accuracy on internal_dev.",
            "candidate_scores": "Independent binary correctness scores; they are not normalized four-class probabilities.",
            "malformed_outputs": "Expected to be zero for this constrained classifier; retained for comparability with generative models.",
        },
    }
    _write_outputs(args.output_dir, metrics, evaluation)
    (args.output_dir / "experiment_config.json").write_text(
        json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/tfidf"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tfidf"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-questions", type=_positive_int)
    parser.add_argument("--max-dev-questions", type=_positive_int)
    parser.add_argument("--min-df", type=_positive_int, default=2)
    parser.add_argument("--max-features", type=_positive_int, default=50_000)
    parser.add_argument("--batch-size", type=_positive_int, default=2_000)
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing files in non-empty artifact directories.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = run_experiment(args)
    development = metrics["development_metrics"]
    print(
        "Complete: "
        f"accuracy={development['accuracy']:.4f} "
        f"({development['correct']}/{development['total_evaluated']}), "
        f"vocabulary={metrics['tfidf_vocabulary_size']:,}"
    )
    print(f"Model artifacts: {args.model_dir}")
    print(f"Evaluation artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
