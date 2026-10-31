"""Embed the Plan 1 retrieval corpus with one model and save a searchable index.

Run once per method (minilm, e5_base_v2, biolord_2023) against the same corpus
run, so all three indexes cover identical passages. ``--limit`` builds a small
smoke index from the first N passages; it is recorded in the run config.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from clearmed.retrieval.corpus import load_corpus
from clearmed.retrieval.embedders import (
    CONFIG_DIR,
    METHOD_IDS,
    Embedder,
    SentenceTransformerEncoder,
    TextEncoder,
    load_embedding_config,
)
from clearmed.retrieval.index import EMBEDDINGS_FILE, DenseIndex

DEFAULT_OUTPUT_ROOT = Path("outputs/retrieval")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def run(
    *,
    method_id: str,
    corpus_run: Path,
    output_dir: Path,
    run_id: str,
    limit: int | None = None,
    device: str = "cpu",
    config_dir: Path | None = None,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    """Build and save one index; return the saved run config."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory {output_dir} is not empty; choose a new --run-id.")
    config = load_embedding_config(method_id, config_dir or CONFIG_DIR)
    passages, corpus_config = load_corpus(corpus_run)
    if limit is not None:
        passages = passages[:limit]
    texts = [passage["text"] for passage in passages]
    doc_ids = [passage["doc_id"] for passage in passages]

    embedder = Embedder(config, encoder or SentenceTransformerEncoder(config, device=device))
    truncation_rate = embedder.truncation_rate(texts)
    started = perf_counter()
    vectors = embedder.encode_documents(texts)
    build_seconds = perf_counter() - started
    description = embedder.describe(dimension=int(vectors.shape[1]))

    index = DenseIndex(
        vectors,
        doc_ids,
        {
            **{key: description[key] for key in ("method_id", "model_name", "revision", "dimension")},
            "corpus_run_id": corpus_config["run_id"],
            "passages_sha256": corpus_config["passages_sha256"],
            "passage_count": len(doc_ids),
            "limit": limit,
        },
    )
    index.save(output_dir)

    run_config: dict[str, Any] = {
        "run_id": run_id,
        "embedding": description,
        "corpus": {
            "run_id": corpus_config["run_id"],
            "passages_sha256": corpus_config["passages_sha256"],
            "passages_in_corpus": corpus_config["passages_out"],
            "passages_indexed": len(doc_ids),
            "limit": limit,
        },
        "truncation_rate": truncation_rate,
        "build_seconds": build_seconds,
        "index_size_bytes": (output_dir / EMBEDDINGS_FILE).stat().st_size,
        "indexing": {"type": "exact", "library": "numpy", "tie_break": "corpus order"},
        "device": device,
        "environment": _environment(),
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_config


def _environment() -> dict[str, str]:
    environment = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__}
    for package in ("torch", "transformers", "sentence_transformers"):
        try:
            environment[package] = __import__(package).__version__
        except ImportError:
            environment[package] = "not installed"
    return environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHOD_IDS)
    parser.add_argument(
        "--corpus-run", required=True, type=Path, help="A Plan 1 run, e.g. outputs/retrieval/corpus/<run_id>."
    )
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=_positive_int, help="Index only the first N passages (smoke test).")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_root / args.method / args.run_id
    config = run(
        method_id=args.method,
        corpus_run=args.corpus_run,
        output_dir=output_dir,
        run_id=args.run_id,
        limit=args.limit,
        device=args.device,
    )
    print(
        f"Index complete: {config['corpus']['passages_indexed']:,} passages, "
        f"dimension {config['embedding']['dimension']}, "
        f"{config['truncation_rate']:.1%} truncated, {config['build_seconds']:.0f}s."
    )
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
