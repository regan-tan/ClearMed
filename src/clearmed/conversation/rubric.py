"""RQ4 rubric sheets: blind reply scoring and grounding scoring (Plan 7).

Two sheets, both shuffled and with every system name hidden behind opaque
item IDs; the key that maps IDs back to systems is written separately.

* Blind sheet: every non-fallback reply from all four systems, scored 0-2 on
  ``correct`` and ``no_unsupported_claims``, plus ``consistent`` for second
  turns only.
* Grounding sheet: the retrieval systems' non-fallback replies next to the
  evidence they received, scored 0-2 on ``grounded``.

Fallback replies are never exported; they are counted automatically. A fixed
share of items (default 20%) goes to both labellers for Cohen's kappa; the
rest is split evenly between them.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Mapping, Sequence
from random import Random
from typing import Any

from clearmed.conversation.pipeline import NO_RETRIEVAL, evidence_block
from clearmed.data.preprocessing import is_nonblank_text
from clearmed.retrieval.judging import LABELLERS

SHEETS = ("blind", "grounding")
CRITERIA = {"blind": ("correct", "no_unsupported_claims", "consistent"), "grounding": ("grounded",)}
SCORE_VALUES = {"0": 0, "1": 1, "2": 2}
NOT_APPLICABLE = "n/a"
SHARED_FRACTION = 0.2
COMMON_COLUMNS = (
    "item_id", "labeller", "question", "correct_answer", "reference_explanation", "previous_turn", "follow_up", "reply",
)
COLUMNS = {
    "blind": (*COMMON_COLUMNS, *CRITERIA["blind"], "note"),
    "grounding": (*COMMON_COLUMNS[:6], "follow_up", "evidence", "reply", *CRITERIA["grounding"], "note"),
}


def gradable_replies(turns_by_system: Mapping[str, Iterable[Mapping[str, Any]]], sheet: str) -> list[dict[str, Any]]:
    """Non-fallback replies for one sheet; grounding leaves out the no-retrieval control.

    Systems that produced the very same prompt and reply for a turn (a gated
    system that accepted its evidence and its gate-off twin) share one item:
    ``system`` names the first of them and ``systems`` lists all, so the reply
    is scored once and the score counts for each.
    """
    if sheet not in SHEETS:
        raise ValueError(f"Unknown sheet {sheet!r}; expected one of {SHEETS}.")
    replies: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for system, turns in sorted(turns_by_system.items()):
        if sheet == "grounding" and system == NO_RETRIEVAL:
            continue
        for t in turns:
            if t["fallback"]:
                continue
            identity = (t["query_id"], int(t["turn"]), t["prompt"], t["reply"])
            if identity in replies:
                replies[identity]["systems"].append(system)
            else:
                replies[identity] = {"system": system, "systems": [system], "query_id": t["query_id"],
                                     "turn": int(t["turn"])}
    return list(replies.values())


def assign_labellers(
    replies: Sequence[Mapping[str, Any]], *, sheet: str, rng: Random, shared_fraction: float = SHARED_FRACTION
) -> list[dict[str, Any]]:
    """Key entries: each reply once per labeller who scores it, with an opaque ID.

    The returned order is the sheet order: by labeller, then shuffled.
    """
    order = list(range(len(replies)))
    rng.shuffle(order)
    shared_count = round(len(replies) * shared_fraction)
    entries: list[dict[str, Any]] = []
    for position, index in enumerate(order):
        shared = position < shared_count
        labellers = LABELLERS if shared else (LABELLERS[(position - shared_count) % len(LABELLERS)],)
        entries += [{**replies[index], "labeller": labeller, "shared": shared} for labeller in labellers]
    prefix = "R" if sheet == "blind" else "G"
    for entry, number in zip(entries, rng.sample(range(100_000, 1_000_000), len(entries)), strict=True):
        entry["item_id"] = f"{prefix}-{number}"
        entry["sheet"] = sheet
    by_labeller = {labeller: [e for e in entries if e["labeller"] == labeller] for labeller in LABELLERS}
    ordered: list[dict[str, Any]] = []
    for labeller in LABELLERS:
        rng.shuffle(by_labeller[labeller])
        ordered += by_labeller[labeller]
    return ordered


def sheet_rows(
    entries: Iterable[Mapping[str, Any]],
    *,
    sheet: str,
    records: Mapping[str, Mapping[str, Any]],
    turns: Mapping[tuple[str, str, int], Mapping[str, Any]],
    explanations: Mapping[str, str | None],
) -> list[dict[str, str]]:
    """Sheet rows. No system names, passage IDs, scores or gate decisions."""
    rows = []
    for entry in entries:
        record = records[entry["query_id"]]
        turn = turns[(entry["system"], entry["query_id"], entry["turn"])]
        explanation = explanations.get(record["question_id"])
        options = "\n".join(f"{label}. {text}" for label, text in record["options"].items())
        previous = turn.get("previous_turn")
        row = {
            "item_id": entry["item_id"],
            "labeller": entry["labeller"],
            "question": f"{record['question']}\n{options}",
            "correct_answer": f"{record['gold_label']}. {record['gold_answer_text']}",
            "reference_explanation": explanation if is_nonblank_text(explanation) else "(none in dataset)",
            "previous_turn": f"Student: {previous[0]}\nTutor: {previous[1]}" if previous else "",
            "follow_up": turn["follow_up"],
            "reply": turn["reply"],
            "note": "",
        }
        if sheet == "blind":
            row.update(correct="", no_unsupported_claims="",
                       consistent="" if entry["turn"] == 2 else NOT_APPLICABLE)
        else:
            row.update(evidence=evidence_block(turn["prompt"]), grounded="")
        rows.append(row)
    return rows


def write_sheet(rows: Sequence[Mapping[str, str]], sheet: str) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS[sheet], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def read_scores(sheet_text: str, key: Sequence[Mapping[str, Any]], sheet: str) -> dict[str, dict[str, int]]:
    """Validate a completed sheet against its key; return item_id -> criterion -> 0/1/2."""
    expected = {entry["item_id"]: entry for entry in key if entry["sheet"] == sheet}
    rows = list(csv.DictReader(io.StringIO(sheet_text)))
    missing_columns = set(COLUMNS[sheet]) - set(rows[0] if rows else {})
    if missing_columns:
        raise ValueError(f"{sheet} sheet is missing columns {sorted(missing_columns)}.")
    scores: dict[str, dict[str, int]] = {}
    problems = []
    for row in rows:
        item_id = row["item_id"].strip()
        entry = expected.get(item_id)
        if entry is None:
            problems.append(f"unknown item {item_id!r}")
            continue
        if item_id in scores:
            problems.append(f"{item_id} appears twice")
            continue
        if row["labeller"].strip() != entry["labeller"]:
            problems.append(f"{item_id}: labeller changed to {row['labeller']!r}")
        values: dict[str, int] = {}
        for criterion in CRITERIA[sheet]:
            cell = row[criterion].strip()
            if criterion == "consistent" and entry["turn"] != 2:
                if cell not in ("", NOT_APPLICABLE):
                    problems.append(f"{item_id}: consistent applies to second turns only")
                continue
            if cell not in SCORE_VALUES:
                problems.append(f"{item_id}: {criterion} must be 0, 1 or 2, got {cell!r}")
                continue
            values[criterion] = SCORE_VALUES[cell]
        scores[item_id] = values
    absent = sorted(set(expected) - set(scores))
    if absent:
        problems.append(f"{len(absent)} items missing from the sheet, e.g. {absent[:3]}")
    if problems:
        raise ValueError(f"{sheet} sheet has {len(problems)} problem(s): " + "; ".join(problems[:10]))
    return scores
