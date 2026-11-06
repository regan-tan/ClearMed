"""Score RQ4 from the completed rubric sheets (Plan 7).

Reads a conversation-eval run, its rubric key, the completed sheets in
``data/retrieval_eval/rq4/<run_id>/`` and the stage B retrieval judgments,
and writes ``metrics.json`` and ``rq4_table.md`` to the run directory:

* mean rubric scores per system with 95% bootstrap CIs, over all replies and
  over the replies every system gave;
* fallbacks counted as correct (no relevant passage in stage B) or unnecessary;
* Cohen's kappa per criterion on the doubly scored replies;
* paired tests between systems;
* reply scores when the evidence held a judged-relevant passage vs not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clearmed.conversation.pipeline import GATE_OFF_SUFFIX, NO_RETRIEVAL
from clearmed.conversation.rubric import SHEETS, read_scores
from clearmed.evaluation.conversation import (
    TABLE_CRITERIA,
    all_kappas,
    answered_by_all,
    classify_fallbacks,
    pairwise,
    reply_scores,
    retrieval_vs_reply,
    system_table,
)
from clearmed.evaluation.retrieval import relevant_sets
from clearmed.retrieval.runs import JUDGING_ROOT, load_judged_pool

EVAL_ROOT = Path("outputs/conversation/eval")
LABELS_ROOT = Path("data/retrieval_eval/rq4")
NAMES = {"no_retrieval": "No retrieval", "minilm": "MiniLM", "e5_base_v2": "E5-base-v2", "biolord_2023": "BioLORD-2023"}
NAMES |= {f"{m}{GATE_OFF_SUFFIX}": f"{name} (gate off)" for m, name in list(NAMES.items())[1:]}
HEADERS = {"correct": "Correct", "no_unsupported_claims": "No unsupported claims", "grounded": "Grounded",
           "consistent": "Consistent"}


def _cell(summary: dict[str, Any]) -> str:
    if not summary["n"]:
        return "N/A"
    low, high = summary["ci95"]
    return f"{summary['mean']:.2f} [{low:.2f}, {high:.2f}] (n={summary['n']})"


def _table(title: str, table: dict[str, Any], fallbacks: dict[str, Any]) -> str:
    lines = [f"**{title}**", "", "| System | " + " | ".join(HEADERS.values()) + " | Fallbacks (correct / unnecessary) |",
             "|" + "---|" * (len(HEADERS) + 2)]
    for system, row in table.items():
        f = fallbacks[system]
        fallback = "N/A" if system == NO_RETRIEVAL or system.endswith(GATE_OFF_SUFFIX) else f"{f['fallbacks']} ({f['correct']} / {f['unnecessary']})"
        lines.append(f"| {NAMES.get(system, system)} | " + " | ".join(_cell(row[c]) for c in TABLE_CRITERIA)
                     + f" | {fallback} |")
    return "\n".join(lines) + "\n"


def run(
    *, run_dir: Path, labels_dir: Path, pool_id: str, judging_root: Path = JUDGING_ROOT
) -> dict[str, Any]:
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"{run_dir / 'metrics.json'} exists; RQ4 metrics are written once per run.")
    config = json.loads((run_dir / "config.json").read_text("utf-8"))
    systems = list(config.get("system_order", config["systems"]))
    turns_by_system = {
        system: [json.loads(line) for line in (run_dir / system / "predictions.jsonl").read_text("utf-8").splitlines()
                 if line.strip()]
        for system in systems
    }
    key = [json.loads(line) for line in (run_dir / "rubric" / "key.jsonl").read_text("utf-8").splitlines()
           if line.strip()]
    scores: dict[str, dict[str, int]] = {}
    for sheet in SHEETS:
        scores |= read_scores((labels_dir / f"{sheet}_sheet.csv").read_text("utf-8"), key, sheet)

    _, judgments, judgments_sha = load_judged_pool(pool_id, judging_root)
    relevant = relevant_sets(j for j in judgments if j["stage"] == "B")
    replies = reply_scores(key, scores)
    common = answered_by_all(turns_by_system)
    # The systems that always answer: no retrieval and every gate-off system. Comparing them on the same
    # turns isolates what the evidence itself adds, without the gate deciding which turns count.
    ungated = [s for s in systems if s == NO_RETRIEVAL or s.endswith(GATE_OFF_SUFFIX)]
    ungated_common = answered_by_all({s: turns_by_system[s] for s in ungated}) if len(ungated) > 1 else set()
    fallbacks = classify_fallbacks(turns_by_system, relevant)
    summary = {
        "run_id": run_dir.name,
        "dry_run": config.get("dry_run", False),
        "pool_id": pool_id,
        "judgments_sha256": judgments_sha,
        "scoring": "0-2 per criterion; doubly scored replies use the mean of both labellers; means exclude fallbacks",
        "all_replies": system_table(replies, systems),
        "answered_by_all": {"n_turns": len(common), "table": system_table(replies, systems, only=common)},
        "fallbacks": fallbacks,
        "kappa": all_kappas(key, scores),
        "always_answering": {"systems": ungated, "n_turns": len(ungated_common),
                             "table": system_table(replies, ungated, only=ungated_common),
                             "pairwise": pairwise(replies, ungated, ungated_common)} if len(ungated) > 1 else None,
        "pairwise_answered_by_both": pairwise(replies, systems),
        "retrieval_vs_reply": retrieval_vs_reply(turns_by_system, replies, relevant),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = "\n".join([
        _table("All replies", summary["all_replies"], fallbacks),
        _table(f"Turns every system answered ({len(common)})", summary["answered_by_all"]["table"], fallbacks),
        *([_table(f"Always-answering systems on the same turns ({len(ungated_common)})",
                  summary["always_answering"]["table"], fallbacks)] if summary["always_answering"] else []),
        "Means are 0-2 with 95% bootstrap CIs; fallbacks are excluded from the means and counted separately.",
    ])
    (run_dir / "rq4_table.md").write_text(table + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pool-id", default="pool-20261020b")
    args = parser.parse_args()
    run(run_dir=EVAL_ROOT / args.run_id, labels_dir=LABELS_ROOT / args.run_id, pool_id=args.pool_id)
    print((EVAL_ROOT / args.run_id / "rq4_table.md").read_text("utf-8"))


if __name__ == "__main__":
    main()
