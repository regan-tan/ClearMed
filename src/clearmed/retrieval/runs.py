"""Load saved retrieval runs and the frozen judgments that score them."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

RANKINGS_ROOT = Path("outputs/retrieval")
JUDGING_ROOT = Path("data/retrieval_eval/judging")


def load_ranking_run(
    method_id: str, run_id: str, *, expected_queries_sha256: str, root: Path = RANKINGS_ROOT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rankings rows and config for one method, checked against the judged query file."""
    run_dir = root / method_id / run_id
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["queries_sha256"] != expected_queries_sha256:
        raise ValueError(f"{run_dir} was run on a different query file than the judged pool.")
    if config["answer_source"] != "gold":
        raise ValueError(f"{run_dir} must be a gold-answer run (calibration and RQ3 use gold context).")
    with (run_dir / "rankings.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows, config


def load_judged_pool(pool_id: str, root: Path = JUDGING_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """pool.json, the imported judgments and their checksum."""
    pool_dir = root / pool_id
    pool = json.loads((pool_dir / "pool.json").read_text(encoding="utf-8"))
    path = pool_dir / "judgments.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No judgments at {path}; run scripts/import_labels.py first.")
    body = path.read_bytes()
    judgments = [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]
    return pool, judgments, sha256(body).hexdigest()


def top1_items(
    rows: list[dict[str, Any]], labels: dict[tuple[str, str], int]
) -> list[dict[str, Any]]:
    """Each query's top-1 passage, score and judged label (query_id, doc_id) -> 0/1."""
    items = []
    for row in rows:
        top = row["results"][0]
        key = (row["query_id"], top["doc_id"])
        if key not in labels:
            raise ValueError(f"Top-1 passage {top['doc_id']} for {row['query_id']} was never judged.")
        items.append({"query_id": row["query_id"], "follow_up_type": row["follow_up_type"],
                      "doc_id": top["doc_id"], "score": float(top["score"]), "relevant": labels[key]})
    return items
