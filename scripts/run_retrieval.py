"""Run a frozen query set through one or more saved indexes.

Each method gets ``outputs/retrieval/<method_id>/<run_id>/rankings.jsonl`` with
the top-k passages and cosine scores per query, plus ``config.json``. The
answer line follows the stage rules in ``clearmed.retrieval.queries``: gold is
allowed only for calibration and RQ3; predicted answers need ``--predictions``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

from clearmed.retrieval.embedders import (
    CONFIG_DIR,
    METHOD_IDS,
    Embedder,
    SentenceTransformerEncoder,
    TextEncoder,
    load_embedding_config,
)
from clearmed.retrieval.index import DenseIndex
from clearmed.retrieval.queries import STAGE_ANSWER_SOURCES, query_text_for

QUERY_FILES = {
    "calibration": Path("data/retrieval_eval/queries_calibration.jsonl"),
    "evaluation": Path("data/retrieval_eval/queries_eval.jsonl"),
}
DEFAULT_STAGE = {"calibration": "calibration", "evaluation": "rq3"}
OUTPUT_ROOT = Path("outputs/retrieval")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    body = path.read_bytes()
    records = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"{path} contains no queries.")
    return records, sha256(body).hexdigest()


def load_predictions(path: Path) -> dict[str, str]:
    """Read ``{"question_id", "predicted_label"}`` lines into a lookup."""
    predictions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            predictions[str(record["question_id"])] = record["predicted_label"]
    return predictions


def build_query_texts(
    records: list[dict[str, Any]], *, stage: str, answer_source: str, predictions: dict[str, str] | None
) -> list[str]:
    texts = []
    for record in records:
        predicted = None
        if answer_source == "predicted":
            if predictions is None or record["question_id"] not in predictions:
                raise ValueError(f"No predicted answer for question {record['question_id']}.")
            predicted = predictions[record["question_id"]]
        texts.append(
            query_text_for(record, stage=stage, answer_source=answer_source, predicted_label=predicted)
        )
    return texts


def run(
    *,
    method_id: str,
    queries_path: Path,
    index_dir: Path,
    output_dir: Path,
    run_id: str,
    stage: str,
    answer_source: str,
    k: int = 5,
    predictions: dict[str, str] | None = None,
    config_dir: Path | None = None,
    encoder: TextEncoder | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Run directory {output_dir} is not empty; choose a new --run-id.")
    records, queries_sha256 = load_records(queries_path)
    texts = build_query_texts(records, stage=stage, answer_source=answer_source, predictions=predictions)

    config = load_embedding_config(method_id, config_dir or CONFIG_DIR)
    index = DenseIndex.load(index_dir)
    embedder = Embedder(config, encoder or SentenceTransformerEncoder(config))
    started = perf_counter()
    vectors = embedder.encode_queries(texts)
    description = embedder.describe(dimension=int(vectors.shape[1]))
    index.ensure_compatible(description)
    hits = index.search(vectors, k=k)
    seconds = perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "rankings.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record, text, results in zip(records, texts, hits, strict=True):
            row = {
                "query_id": record["query_id"],
                "question_id": record["question_id"],
                "query_set": record["query_set"],
                "follow_up_type": record["follow_up_type"],
                "method_id": method_id,
                "query_text": text,
                "results": results,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    run_config: dict[str, Any] = {
        "run_id": run_id,
        "method_id": method_id,
        "stage": stage,
        "answer_source": answer_source,
        "k": k,
        "queries_file": queries_path.as_posix(),
        "queries_sha256": queries_sha256,
        "query_count": len(records),
        "index_dir": index_dir.as_posix(),
        "index": index.metadata,
        "embedding": description,
        "query_seconds": seconds,
        "seconds_per_query": seconds / len(records),
        "environment": {"python": sys.version, "platform": platform.platform()},
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, choices=sorted(QUERY_FILES))
    parser.add_argument("--index-run", required=True, help="Run ID of the indexes, e.g. full-20261016.")
    parser.add_argument("--methods", nargs="+", choices=METHOD_IDS, default=list(METHOD_IDS))
    parser.add_argument("--stage", choices=sorted(STAGE_ANSWER_SOURCES))
    parser.add_argument("--answer-source", choices=("gold", "predicted"), default="gold")
    parser.add_argument("--predictions", type=Path, help="JSONL of question_id and predicted_label.")
    parser.add_argument("--k", type=_positive_int, default=5)
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stage = args.stage or DEFAULT_STAGE[args.queries]
    run_id = args.run_id or (
        f"{args.queries}-{args.answer_source}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    )
    predictions = load_predictions(args.predictions) if args.predictions else None
    for method_id in args.methods:
        output_dir = OUTPUT_ROOT / method_id / run_id
        config = run(
            method_id=method_id,
            queries_path=QUERY_FILES[args.queries],
            index_dir=OUTPUT_ROOT / method_id / args.index_run,
            output_dir=output_dir,
            run_id=run_id,
            stage=stage,
            answer_source=args.answer_source,
            k=args.k,
            predictions=predictions,
        )
        print(f"{method_id}: {config['query_count']} queries, top-{config['k']} -> {output_dir}")


if __name__ == "__main__":
    main()
