"""Calibrate and freeze each model's confidence-gate threshold (Plan 5).

Uses only stage A: each model's top-1 passage for the internal_dev calibration
queries, with its judged label. Writes the per-model precision/recall/F1 curve
to ``outputs/retrieval/calibration/<run_id>/`` and freezes the chosen
thresholds in ``configs/retrieval/gate_thresholds.json``. That file is never
overwritten: thresholds must not be retuned, least of all on validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from clearmed.retrieval.embedders import METHOD_IDS
from clearmed.retrieval.gate import THRESHOLDS_PATH, calibrate_threshold
from clearmed.retrieval.runs import (
    JUDGING_ROOT,
    RANKINGS_ROOT,
    load_judged_pool,
    load_ranking_run,
    top1_items,
)

OUTPUT_ROOT = Path("outputs/retrieval/calibration")


def run(
    *,
    pool_id: str,
    run_id: str,
    judging_root: Path = JUDGING_ROOT,
    rankings_root: Path = RANKINGS_ROOT,
    output_root: Path = OUTPUT_ROOT,
    thresholds_path: Path = THRESHOLDS_PATH,
) -> dict[str, Any]:
    output_dir = output_root / run_id
    if thresholds_path.exists():
        raise FileExistsError(f"Thresholds are frozen in {thresholds_path}; they must not be recalibrated.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new --run-id.")

    pool, judgments, judgments_sha = load_judged_pool(pool_id, judging_root)
    stage = pool["stages"]["A"]
    labels = {(j["query_id"], j["doc_id"]): int(j["relevant"]) for j in judgments if j["stage"] == "A"}
    results: dict[str, Any] = {}
    items_by_method = {}
    for method_id in METHOD_IDS:
        rows, _ = load_ranking_run(
            method_id, stage["rankings_run_id"], expected_queries_sha256=stage["queries_sha256"], root=rankings_root
        )
        items = top1_items(rows, labels)
        calibration = calibrate_threshold([i["score"] for i in items], [i["relevant"] for i in items])
        threshold = calibration["chosen"]["threshold"]
        off_topic = [i for i in items if i["follow_up_type"] == "off_topic"]
        results[method_id] = {
            **calibration,
            "off_topic_rejected": sum(i["score"] < threshold for i in off_topic),
            "off_topic_total": len(off_topic),
        }
        items_by_method[method_id] = items

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "items.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for method_id, items in items_by_method.items():
            for item in items:
                handle.write(json.dumps({"method_id": method_id, **item}, sort_keys=True) + "\n")
    summary = {
        "run_id": run_id,
        "pool_id": pool_id,
        "stage": "A",
        "rankings_run_id": stage["rankings_run_id"],
        "queries_sha256": stage["queries_sha256"],
        "judgments_sha256": judgments_sha,
        "rule": "accept if top-1 score >= threshold; threshold = observed score with best F1, ties -> higher "
        "precision, then higher threshold",
        "methods": results,
    }
    (output_dir / "calibration.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")

    frozen = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "calibration_run": (output_dir / "calibration.json").as_posix(),
        "pool_id": pool_id,
        "judgments_sha256": judgments_sha,
        "note": "Calibrated on stage A (internal_dev) only. Never retune on validation.",
        "thresholds": {
            method_id: {key: result["chosen"][key] for key in ("threshold", "precision", "recall", "f1")}
            | {"n": result["n"], "n_positive": result["n_positive"]}
            for method_id, result in results.items()
        },
    }
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-id", required=True)
    parser.add_argument("--run-id", default=f"calibration-{datetime.now(UTC):%Y%m%d}")
    args = parser.parse_args()
    summary = run(pool_id=args.pool_id, run_id=args.run_id)
    for method_id, result in summary["methods"].items():
        chosen = result["chosen"]
        print(
            f"{method_id}: threshold {chosen['threshold']:.4f}  P {chosen['precision']:.2f}  R {chosen['recall']:.2f}  "
            f"F1 {chosen['f1']:.2f}  ({result['n_positive']}/{result['n']} relevant); off-topic rejected "
            f"{result['off_topic_rejected']}/{result['off_topic_total']}"
        )


if __name__ == "__main__":
    main()
