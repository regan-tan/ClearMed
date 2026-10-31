"""Build the Part II retrieval corpus from the canonical saved partitions.

Reads only ``data/splits`` (no path overrides), verifies the evidence subset
against ``manifest.json``, removes passages that duplicate an official
validation question, and writes ``passages.jsonl`` plus ``config.json`` to a
new run directory. Existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import datasets
from datasets import Dataset, load_from_disk

from clearmed.data.loader import DATASET_NAME, load_medmcqa_split
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.data.verification import (
    EVIDENCE_PARTITION,
    load_manifest,
    verify_against_manifest,
)
from clearmed.retrieval.corpus import build_corpus

CANONICAL_SPLITS_DIR = Path("data/splits")
DEFAULT_OUTPUT_ROOT = Path("outputs/retrieval/corpus")


def _load_partition(path: Path, name: str) -> Dataset:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing {name} partition at {path}. Generate the splits first with "
            "`uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits`."
        )
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Saved {name} partition must be a Hugging Face Dataset, not {type(dataset)}.")
    return dataset


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory {path} is not empty; choose a new --run-id.")
    path.mkdir(parents=True, exist_ok=True)


def run(
    *, splits_dir: Path, output_dir: Path, evaluation: Dataset, run_id: str
) -> dict[str, Any]:
    """Build the corpus and write its artifacts; return the saved config."""
    manifest = load_manifest(splits_dir / "manifest.json")
    evidence = _load_partition(splits_dir / EVIDENCE_PARTITION, EVIDENCE_PARTITION)
    internal_dev = _load_partition(splits_dir / "internal_dev", "internal_dev")
    internal_dev_verification = verify_against_manifest(internal_dev, "internal_dev", manifest)
    # Calibration queries come from internal_dev, so its questions must not be indexed either.
    assert_train_dev_disjoint(
        evidence, internal_dev, train_name=EVIDENCE_PARTITION, development_name="internal_dev"
    )
    _prepare_output_dir(output_dir)

    passages, report = build_corpus(evidence, manifest=manifest, evaluation=evaluation)
    passages_path = output_dir / "passages.jsonl"
    with passages_path.open("w", encoding="utf-8", newline="\n") as handle:
        for passage in passages:
            handle.write(json.dumps(passage, ensure_ascii=False, sort_keys=True) + "\n")

    config: dict[str, Any] = {
        "run_id": run_id,
        "dataset_name": DATASET_NAME,
        "splits_dir": splits_dir.as_posix(),
        "manifest_seed": manifest.get("seed"),
        "internal_dev_verification": internal_dev_verification,
        "internal_dev_overlap": {"row_id": 0, "normalized_question": 0},
        "evaluation_split_rows": len(evaluation),
        "evaluation_split_fingerprint": membership_fingerprint(evaluation),
        **report,
        "passages_file": passages_path.name,
        "passages_sha256": sha256(passages_path.read_bytes()).hexdigest(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "datasets": datasets.__version__,
        },
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        help="Name of the new run directory (default: current UTC time).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_root / args.run_id
    config = run(
        splits_dir=CANONICAL_SPLITS_DIR,
        output_dir=output_dir,
        evaluation=load_medmcqa_split("validation"),
        run_id=args.run_id,
    )
    print(
        f"Corpus complete: {config['passages_out']:,} passages from {config['rows_in']:,} rows; "
        f"{config['rows_removed']} removed as validation duplicates."
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
