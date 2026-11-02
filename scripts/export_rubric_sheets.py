"""Export the RQ4 rubric sheets for one conversation-eval run (Plan 7).

Writes ``outputs/conversation/eval/<run_id>/rubric/``:

* ``blind_sheet.csv``: every non-fallback reply from all four systems,
  shuffled, system hidden; score ``correct``, ``no_unsupported_claims`` and
  (second turns only) ``consistent`` as 0, 1 or 2;
* ``grounding_sheet.csv``: retrieval replies with the evidence they got;
  score ``grounded`` as 0, 1 or 2;
* ``key.jsonl``: item ID -> system, query, turn, labeller (do not share);
* ``summary.json``: rows per labeller.

Completed sheets go to ``data/retrieval_eval/rq4/<run_id>/`` under the same
file names, for ``evaluate_conversation.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from random import Random
from typing import Any

from clearmed.conversation.rubric import (
    SHEETS,
    assign_labellers,
    gradable_replies,
    sheet_rows,
    write_sheet,
)

EVAL_ROOT = Path("outputs/conversation/eval")
SEED = 42


def read_turns(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    config = json.loads((run_dir / "config.json").read_text("utf-8"))
    return {
        system: [json.loads(line) for line in (run_dir / system / "predictions.jsonl").read_text("utf-8").splitlines()
                 if line.strip()]
        for system in config["systems"]
    }


def run(*, run_dir: Path, records: Mapping[str, Mapping[str, Any]], explanations: Mapping[str, str | None]) -> dict:
    output_dir = run_dir / "rubric"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; sheets for this run were already exported.")
    turns_by_system = read_turns(run_dir)
    turns = {(system, t["query_id"], int(t["turn"])): t for system, ts in turns_by_system.items() for t in ts}
    rng = Random(f"{SEED}-rq4-rubric")
    key: list[dict[str, Any]] = []
    sheets: dict[str, str] = {}
    for sheet in SHEETS:
        entries = assign_labellers(gradable_replies(turns_by_system, sheet), sheet=sheet, rng=rng)
        rows = sheet_rows(entries, sheet=sheet, records=records, turns=turns, explanations=explanations)
        sheets[sheet] = write_sheet(rows, sheet)
        key += entries

    output_dir.mkdir(parents=True)
    for sheet, text in sheets.items():
        (output_dir / f"{sheet}_sheet.csv").write_text(text, encoding="utf-8", newline="")
    with (output_dir / "key.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for entry in key:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    summary = {
        sheet: {
            "rows_per_labeller": dict(sorted(Counter(e["labeller"] for e in key if e["sheet"] == sheet).items())),
            "replies": len({(e["system"], e["query_id"], e["turn"]) for e in key if e["sheet"] == sheet}),
            "shared_replies": len({(e["system"], e["query_id"], e["turn"]) for e in key
                                   if e["sheet"] == sheet and e["shared"]}),
        }
        for sheet in SHEETS
    }
    summary["fallbacks_not_exported"] = {
        system: sum(t["fallback"] for t in ts) for system, ts in turns_by_system.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    return summary


def _validation_explanations(question_ids: set[str]) -> dict[str, str | None]:
    from clearmed.data.loader import load_medmcqa_split

    return {str(row["id"]): row.get("exp") for row in load_medmcqa_split("validation")
            if str(row["id"]) in question_ids}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir = EVAL_ROOT / args.run_id
    config = json.loads((run_dir / "config.json").read_text("utf-8"))
    records = {
        r["query_id"]: r for r in map(json.loads, Path(config["queries"]).read_text("utf-8").splitlines()) if r.get("rq4")
    }
    explanations = _validation_explanations({r["question_id"] for r in records.values()})
    print(json.dumps(run(run_dir=run_dir, records=records, explanations=explanations), indent=2))


if __name__ == "__main__":
    main()
