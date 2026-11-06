"""Score the three embedding models on the RQ3 benchmark (Plan 5).

Joins each model's evaluation rankings with the stage B pooled judgments and
writes, to ``outputs/retrieval/eval/<run_id>/``:

* ``metrics.json``: Hit@1/3/5, Recall@1/3/5 and MRR@5 per model with 95%
  bootstrap intervals, paired tests for every model pair, a breakdown by
  follow-up type, efficiency figures and, if thresholds are frozen, how the
  gate performs on these held-out queries;
* ``per_query.jsonl``: every query's scores per model;
* ``rq3_table.md``: the report table.

Queries whose pool holds no relevant passage are left out and counted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from clearmed.evaluation.retrieval import (
    bootstrap_ci,
    evaluate_rankings,
    paired_test,
    relevant_sets,
)
from clearmed.retrieval.embedders import METHOD_IDS
from clearmed.retrieval.gate import THRESHOLDS_PATH, confusion_at
from clearmed.retrieval.runs import (
    JUDGING_ROOT,
    RANKINGS_ROOT,
    load_judged_pool,
    load_ranking_run,
    top1_items,
)

OUTPUT_ROOT = Path("outputs/retrieval/eval")
TABLE_METRICS = ("hit@1", "hit@3", "hit@5", "recall@3", "recall@5", "mrr@5")
NAMES = {"minilm": "MiniLM", "e5_base_v2": "E5-base-v2", "biolord_2023": "BioLORD-2023"}


def _efficiency(config: dict[str, Any]) -> dict[str, Any]:
    index_config_path = Path(config["index_dir"]) / "config.json"
    index_config = json.loads(index_config_path.read_text("utf-8")) if index_config_path.is_file() else {}
    return {
        "seconds_per_query": config.get("seconds_per_query"),
        "dimension": config["embedding"].get("dimension"),
        "effective_max_seq_length": config["embedding"].get("effective_max_seq_length"),
        "index_build_seconds": index_config.get("build_seconds"),
        "index_size_bytes": index_config.get("index_size_bytes"),
        "passage_truncation_rate": index_config.get("truncation_rate"),
    }


def _gate_check(rows: list[dict[str, Any]], labels: dict, threshold: float) -> dict[str, Any]:
    items = top1_items(rows, labels)
    on_topic = [i for i in items if i["follow_up_type"] != "off_topic"]
    off_topic = [i for i in items if i["follow_up_type"] == "off_topic"]
    return {
        "threshold": threshold,
        "on_topic": confusion_at([i["score"] for i in on_topic], [i["relevant"] for i in on_topic], threshold),
        "all_queries": confusion_at([i["score"] for i in items], [i["relevant"] for i in items], threshold),
        "off_topic_rejected": sum(i["score"] < threshold for i in off_topic),
        "off_topic_total": len(off_topic),
    }


def _table(summary: dict[str, Any]) -> str:
    header = "| Model | " + " | ".join(TABLE_METRICS) + " |"
    lines = [header, "|" + "---|" * (len(TABLE_METRICS) + 1)]
    for method_id, result in summary["methods"].items():
        cells = []
        for metric in TABLE_METRICS:
            mean = result["mean"][metric]
            low, high = result["ci95"][metric]
            cells.append(f"{mean:.3f} [{low:.2f}, {high:.2f}]")
        lines.append(f"| {NAMES.get(method_id, method_id)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Mean over {summary['n_evaluated']} evaluation queries with at least one relevant pooled passage "
        f"({summary['n_excluded']} of {summary['n_judged_queries']} left out). Brackets: 95% bootstrap CI."
    )
    return "\n".join(lines) + "\n"


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
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new --run-id.")
    pool, judgments, judgments_sha = load_judged_pool(pool_id, judging_root)
    stage = pool["stages"]["B"]
    stage_b = [j for j in judgments if j["stage"] == "B"]
    relevant = relevant_sets(stage_b)
    labels = {(j["query_id"], j["doc_id"]): int(j["relevant"]) for j in stage_b}
    thresholds = (
        json.loads(thresholds_path.read_text("utf-8"))["thresholds"] if thresholds_path.is_file() else None
    )

    methods: dict[str, Any] = {}
    follow_up_type: dict[str, str] = {}
    for method_id in METHOD_IDS:
        rows, config = load_ranking_run(
            method_id, stage["rankings_run_id"], expected_queries_sha256=stage["queries_sha256"], root=rankings_root
        )
        follow_up_type.update({row["query_id"]: row["follow_up_type"] for row in rows})
        rankings = {row["query_id"]: [hit["doc_id"] for hit in row["results"]] for row in rows}
        result = evaluate_rankings(rankings, relevant)
        result["ci95"] = {
            name: bootstrap_ci([q[name] for q in result["per_query"].values()]) for name in result["mean"]
        }
        result["efficiency"] = _efficiency(config)
        if thresholds and method_id in thresholds:
            result["gate_heldout"] = _gate_check(rows, labels, thresholds[method_id]["threshold"])
        methods[method_id] = result

    evaluated = sorted(next(iter(methods.values()))["per_query"])
    metric_names = list(next(iter(methods.values()))["mean"])
    pairwise = {
        f"{a}_vs_{b}": {
            name: paired_test(
                [methods[a]["per_query"][q][name] for q in evaluated],
                [methods[b]["per_query"][q][name] for q in evaluated],
            )
            for name in metric_names
        }
        for a, b in combinations(METHOD_IDS, 2)
    }
    by_type: dict[str, Any] = defaultdict(dict)
    for method_id, result in methods.items():
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for query_id, values in result["per_query"].items():
            groups[follow_up_type[query_id]].append(values)
        for type_name, values in sorted(groups.items()):
            by_type[type_name][method_id] = {
                "n": len(values), **{m: sum(v[m] for v in values) / len(values) for m in TABLE_METRICS}
            }
    excluded = methods[METHOD_IDS[0]]["excluded"]
    summary = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pool_id": pool_id,
        "rankings_run_id": stage["rankings_run_id"],
        "queries_sha256": stage["queries_sha256"],
        "judgments_sha256": judgments_sha,
        "relevance": "pooled stage B judgments; unjudged passages count as not relevant",
        "n_judged_queries": len(relevant),
        "n_evaluated": len(evaluated),
        "n_excluded": len(excluded),
        "excluded_by_type": dict(sorted(
            {t: sum(follow_up_type[q] == t for q in excluded) for t in set(follow_up_type.values())}.items()
        )),
        "methods": {
            method_id: {key: value for key, value in result.items() if key not in ("per_query", "excluded")}
            for method_id, result in methods.items()
        },
        "pairwise": pairwise,
        "by_follow_up_type": dict(by_type),
        "thresholds_file": thresholds_path.as_posix() if thresholds else None,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    with (output_dir / "per_query.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in evaluated:
            record = {"query_id": query_id, "follow_up_type": follow_up_type[query_id],
                      **{m: methods[m]["per_query"][query_id] for m in METHOD_IDS}}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output_dir / "rq3_table.md").write_text(_table(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-id", required=True)
    parser.add_argument("--run-id", default=f"rq3-{datetime.now(UTC):%Y%m%d}")
    args = parser.parse_args()
    run(pool_id=args.pool_id, run_id=args.run_id)
    print((OUTPUT_ROOT / args.run_id / "rq3_table.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
