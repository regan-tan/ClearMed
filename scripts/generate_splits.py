"""Generate reproducible, duplicate-aware MedMCQA training partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

from clearmed.data.loader import DATASET_NAME, load_medmcqa
from clearmed.data.splitting import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    build_manifest,
    filter_retrieval_evidence,
    partition_training_dataset,
)


def _dataset_metadata(dataset: Dataset) -> dict[str, Any]:
    """Extract stable, available source metadata without assuming a cache layout."""
    info = dataset.info
    return {
        "builder_name": info.builder_name,
        "config_name": info.config_name,
        "version": str(info.version) if info.version else None,
        "dataset_size": info.dataset_size,
        "features": list(dataset.features),
    }


def generate_splits(
    dataset: DatasetDict,
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create all split artifacts and return the saved manifest data.

    The output directory must be empty or absent so an existing experiment is
    never overwritten silently.
    """
    required_splits = {"train", "validation", "test"}
    missing = required_splits - set(dataset)
    if missing:
        raise ValueError(f"MedMCQA dataset is missing official splits: {sorted(missing)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory {output_dir} is not empty; choose a new directory."
        )

    train = dataset["train"]
    partitions = partition_training_dataset(train, seed=seed, ratios=DEFAULT_RATIOS)
    retrieval_evidence = filter_retrieval_evidence(partitions["retrieval_kb"])
    manifest = build_manifest(
        dataset_name=DATASET_NAME,
        source_train=train,
        partitions=partitions,
        retrieval_evidence=retrieval_evidence,
        seed=seed,
        ratios=DEFAULT_RATIOS,
        validation=dataset["validation"],
        test=dataset["test"],
        source_metadata=_dataset_metadata(train),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, partition in partitions.items():
        partition.save_to_disk(str(output_dir / name))
    retrieval_evidence.save_to_disk(str(output_dir / "retrieval_kb_evidence"))
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _print_summary(manifest: dict[str, Any]) -> None:
    print(f"Dataset: {manifest['dataset_name']}")
    print(f"Seed: {manifest['seed']}")
    print(f"Official training rows: {manifest['source_train_rows']}")
    for name, stats in manifest["partitions"].items():
        print(
            f"{name}: {stats['rows']} rows ({stats['percentage']:.2f}%), "
            f"QA-valid={stats['qa_valid_rows']}, explanation-valid={stats['explanation_valid_rows']}"
        )
    print(f"retrieval_kb_evidence: {manifest['retrieval_evidence']['rows']} rows")
    for split_name, audit in manifest["official_split_overlap_audit"].items():
        question_overlap = sum(
            item["normalized_question_overlap_count"] for item in audit.values()
        )
        id_overlap = sum(item["row_id_overlap_count"] for item in audit.values())
        print(f"{split_name} overlap: row IDs={id_overlap}, normalized questions={question_overlap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()

    manifest = generate_splits(load_medmcqa(), args.output_dir, seed=args.seed)
    _print_summary(manifest)
    print(f"Saved partitions and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
