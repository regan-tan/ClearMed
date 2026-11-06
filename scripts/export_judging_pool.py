"""Export the blind labelling sheet for retrieval relevance judgments (Plan 4).

Stage A: each model's top-1 passage for the 80 calibration queries.
Stage B: the union of every model's top-5 passages for the 60 evaluation queries.

Writes one sheet with both stages and both labellers to
``outputs/retrieval/judging/<pool_id>/labelling_sheet.csv`` (passage text, git
ignored) and the pool key to ``data/retrieval_eval/judging/<pool_id>/`` (tracked;
read only by scripts). Rankings must come from the current frozen query files
and from indexes built on the given corpus run. Nothing is overwritten.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from typing import Any

from datasets import Dataset, load_from_disk

from clearmed.data.loader import load_medmcqa_split
from clearmed.data.verification import load_manifest, verify_against_manifest
from clearmed.retrieval.corpus import load_corpus
from clearmed.retrieval.embedders import METHOD_IDS
from clearmed.retrieval.judging import (
    assign_item_ids,
    build_pool,
    sheet_rows,
    write_sheet,
)

SEED = 42
QUERY_FILES = {
    "A": Path("data/retrieval_eval/queries_calibration.jsonl"),
    "B": Path("data/retrieval_eval/queries_eval.jsonl"),
}
DEPTHS = {"A": 1, "B": 5}
SHARED_QUERIES = {"A": 0, "B": 12}
RANKINGS_ROOT = Path("outputs/retrieval")
SHEET_ROOT = Path("outputs/retrieval/judging")
KEY_ROOT = Path("data/retrieval_eval/judging")
CORPUS_ROOT = Path("outputs/retrieval/corpus")
SPLITS_DIR = Path("data/splits")


def load_queries(path: Path) -> tuple[list[dict[str, Any]], str]:
    body = path.read_bytes()
    return [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()], sha256(body).hexdigest()


def load_rankings(
    rankings_root: Path, run_id: str, *, queries_sha256: str, passages_sha256: str, depth: int
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """method_id -> query_id -> results, after checking each run matches the inputs."""
    rankings: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method_id in METHOD_IDS:
        run_dir = rankings_root / method_id / run_id
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        if config["queries_sha256"] != queries_sha256:
            raise ValueError(f"{run_dir} was run on a different query file; rerun retrieval.")
        if config["index"]["passages_sha256"] != passages_sha256:
            raise ValueError(f"{run_dir} used an index built on a different corpus.")
        if config["answer_source"] != "gold" or config["k"] < depth:
            raise ValueError(f"{run_dir} must be a gold-answer run with k >= {depth}.")
        with (run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        rankings[method_id] = {row["query_id"]: row["results"] for row in rows}
    return rankings


def run(
    *,
    pool_id: str,
    run_ids: dict[str, str],
    corpus_run: Path,
    explanations: dict[str, str | None],
    query_files: dict[str, Path] = QUERY_FILES,
    rankings_root: Path = RANKINGS_ROOT,
    sheet_root: Path = SHEET_ROOT,
    key_root: Path = KEY_ROOT,
    shared: dict[str, int] = SHARED_QUERIES,
) -> dict[str, Any]:
    sheet_dir, key_dir = sheet_root / pool_id, key_root / pool_id
    for directory in (sheet_dir, key_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} is not empty; choose a new --pool-id.")

    passages, corpus_config = load_corpus(corpus_run)
    by_doc = {passage["doc_id"]: passage for passage in passages}
    rng = Random(f"{SEED}-judging")
    entries: list[dict[str, Any]] = []
    queries: dict[str, dict[str, Any]] = {}
    inputs: dict[str, Any] = {}
    for stage in ("A", "B"):
        records, queries_sha256 = load_queries(query_files[stage])
        rankings = load_rankings(
            rankings_root,
            run_ids[stage],
            queries_sha256=queries_sha256,
            passages_sha256=corpus_config["passages_sha256"],
            depth=DEPTHS[stage],
        )
        entries += build_pool(
            stage=stage,
            queries=records,
            rankings=rankings,
            depth=DEPTHS[stage],
            shared_count=shared[stage],
            rng=rng,
        )
        queries.update({record["query_id"]: record for record in records})
        inputs[stage] = {
            "queries_file": query_files[stage].as_posix(),
            "queries_sha256": queries_sha256,
            "rankings_run_id": run_ids[stage],
            "depth": DEPTHS[stage],
            "shared_queries": shared[stage],
        }
    missing = {entry["question_id"] for entry in entries} - set(explanations)
    if missing:
        raise ValueError(f"No source row for {len(missing)} questions, e.g. {sorted(missing)[:3]}.")
    entries = assign_item_ids(entries, rng)
    sheet = write_sheet(sheet_rows(entries, queries=queries, passages=by_doc, explanations=explanations))

    sheet_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sheet_dir / "labelling_sheet.csv"
    sheet_path.write_text(sheet, encoding="utf-8-sig", newline="")
    key_body = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries)
    (key_dir / "pool_key.jsonl").write_text(key_body, encoding="utf-8", newline="\n")

    counts = Counter((entry["stage"], entry["labeller"]) for entry in entries)
    summary = {
        "pool_id": pool_id,
        "seed": SEED,
        "corpus_run_id": corpus_config["run_id"],
        "passages_sha256": corpus_config["passages_sha256"],
        "methods": list(METHOD_IDS),
        "stages": inputs,
        "items": len(entries),
        "items_per_stage_labeller": {f"{stage}/{labeller}": n for (stage, labeller), n in sorted(counts.items())},
        "unique_pairs_per_stage": {
            stage: len({(e["query_id"], e["doc_id"]) for e in entries if e["stage"] == stage}) for stage in ("A", "B")
        },
        "shared_items": sum(entry["shared"] for entry in entries),
        "sheet_file": sheet_path.as_posix(),
        "sheet_sha256": sha256(sheet_path.read_bytes()).hexdigest(),
        "pool_key_sha256": sha256(key_body.encode("utf-8")).hexdigest(),
    }
    (key_dir / "pool.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_explanations(splits_dir: Path, validation: Dataset) -> dict[str, str | None]:
    """question_id -> reference explanation, for the labellers' reference column."""
    manifest = load_manifest(splits_dir / "manifest.json")
    internal_dev = load_from_disk(str(splits_dir / "internal_dev"))
    verify_against_manifest(internal_dev, "internal_dev", manifest)
    return {str(row["id"]): row.get("exp") for dataset in (internal_dev, validation) for row in dataset}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-id", required=True, help="e.g. pool-20261020")
    parser.add_argument("--calibration-run", required=True, help="Stage A rankings run ID.")
    parser.add_argument("--evaluation-run", required=True, help="Stage B rankings run ID.")
    parser.add_argument("--corpus-run", required=True, help="Corpus run ID, e.g. 20261016-142815.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(
        pool_id=args.pool_id,
        run_ids={"A": args.calibration_run, "B": args.evaluation_run},
        corpus_run=CORPUS_ROOT / args.corpus_run,
        explanations=load_explanations(SPLITS_DIR, load_medmcqa_split("validation")),
    )
    print(json.dumps(summary["items_per_stage_labeller"], indent=2))
    print(f"Sheet: {summary['sheet_file']} ({summary['items']} rows, {summary['shared_items']} shared)")


if __name__ == "__main__":
    main()
