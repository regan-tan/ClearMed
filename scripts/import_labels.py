"""Import the completed labelling sheet and produce final relevance judgments.

1. Download the finished Google Sheet as CSV to
   ``data/retrieval_eval/labels/<pool_id>/labelling_sheet.csv``.
2. Run this script. It rejects blank or non-0/1 labels and missing, extra or
   edited item IDs, then reports Cohen's kappa on the shared queries.
3. If the labellers disagree on any shared passage, it writes
   ``disagreements.csv`` next to the sheet and stops. Discuss each one, fill
   ``agreed_label``, save it as ``adjudication.csv`` and run again.
4. With every disagreement resolved, it writes ``judgments.jsonl`` (one label
   per query and passage) and ``agreement.json`` into the pool's key folder.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from clearmed.retrieval.judging import (
    LABEL_VALUES,
    LABELLERS,
    agreement,
    final_judgments,
    read_labels,
)

KEY_ROOT = Path("data/retrieval_eval/judging")
LABELS_ROOT = Path("data/retrieval_eval/labels")
ADJUDICATION_COLUMNS = ("stage", "query_id", "doc_id", *LABELLERS, "follow_up", "question", "passage", "agreed_label")


def load_key(key_dir: Path) -> list[dict[str, Any]]:
    pool = json.loads((key_dir / "pool.json").read_text(encoding="utf-8"))
    body = (key_dir / "pool_key.jsonl").read_bytes()
    if sha256(body).hexdigest() != pool["pool_key_sha256"]:
        raise ValueError(f"{key_dir / 'pool_key.jsonl'} does not match its recorded checksum.")
    return [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]


def write_disagreements(path: Path, disagreements: list[dict[str, Any]], sheet_text: str, key: list[dict]) -> None:
    """Template for the team to fill ``agreed_label``, with the text needed to discuss each item."""
    rows_by_item = {row["item_id"]: row for row in csv.DictReader(io.StringIO(sheet_text.lstrip("﻿")))}
    item_for = {(entry["stage"], entry["query_id"], entry["doc_id"]): entry["item_id"] for entry in key}
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for item in disagreements:
            sheet_row = rows_by_item[item_for[(item["stage"], item["query_id"], item["doc_id"])]]
            writer.writerow(
                {
                    **{column: item[column] for column in ("stage", "query_id", "doc_id", *LABELLERS)},
                    "follow_up": sheet_row["follow_up"],
                    "question": sheet_row["question"],
                    "passage": sheet_row["passage"],
                    "agreed_label": "",
                }
            )


def read_adjudication(path: Path) -> dict[tuple[str, str, str], int]:
    adjudicated = {}
    for line, row in enumerate(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))), start=2):
        value = (row.get("agreed_label") or "").strip()
        if value not in LABEL_VALUES:
            raise ValueError(f"{path} row {line}: agreed_label must be 0 or 1, got {value!r}.")
        adjudicated[(row["stage"].strip(), row["query_id"].strip(), row["doc_id"].strip())] = LABEL_VALUES[value]
    return adjudicated


def run(*, pool_id: str, key_root: Path = KEY_ROOT, labels_root: Path = LABELS_ROOT) -> dict[str, Any]:
    key_dir, labels_dir = key_root / pool_id, labels_root / pool_id
    outputs = [key_dir / "judgments.jsonl", key_dir / "agreement.json"]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Judgments already imported; refusing to overwrite {existing}.")

    key = load_key(key_dir)
    sheet_path = labels_dir / "labelling_sheet.csv"
    sheet_text = sheet_path.read_text(encoding="utf-8-sig")
    labels = read_labels(sheet_text, key)
    stats = agreement(key, labels)
    disagreements = stats.pop("disagreements")
    report = {"pool_id": pool_id, **stats, "disagreement_count": len(disagreements), "status": "complete"}

    adjudication_path = labels_dir / "adjudication.csv"
    adjudicated = read_adjudication(adjudication_path) if adjudication_path.exists() else {}
    unresolved = [item for item in disagreements if (item["stage"], item["query_id"], item["doc_id"]) not in adjudicated]
    if unresolved:
        template = labels_dir / "disagreements.csv"
        if not template.exists():
            write_disagreements(template, unresolved, sheet_text, key)
        return {**report, "status": "needs_adjudication", "unresolved": len(unresolved), "template": template.as_posix()}

    judgments = final_judgments(key, labels, adjudicated)
    report["sheet_sha256"] = sha256(sheet_path.read_bytes()).hexdigest()
    report["adjudicated"] = sum(judgment["source"] == "adjudicated" for judgment in judgments)
    report["judgments"] = len(judgments)
    report["relevant_per_stage"] = {
        stage: sum(j["relevant"] for j in judgments if j["stage"] == stage) for stage in ("A", "B")
    }
    outputs[0].write_text(
        "".join(json.dumps(judgment, sort_keys=True) + "\n" for judgment in judgments), encoding="utf-8", newline="\n"
    )
    outputs[1].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-id", required=True)
    report = run(pool_id=parser.parse_args().pool_id)
    print(json.dumps(report, indent=2))
    if report["status"] == "needs_adjudication":
        print(f"\n{report['unresolved']} shared passages disagree. Fill agreed_label in {report['template']}, "
              "save it as adjudication.csv in the same folder, then run this again.")


if __name__ == "__main__":
    main()
