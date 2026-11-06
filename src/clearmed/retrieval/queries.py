"""Follow-up query sets and the three-part retrieval query.

A retrieval query joins the student's follow-up, an answer line and the MCQ.
The follow-up comes first so that a model with a short input limit (BioLORD
reads 128 tokens) never cuts it off. The answer line is the gold answer only
for threshold calibration and RQ3; RQ4 and the live tutor must use the model's
predicted answer. Reference explanations are never part of a query.

Follow-up wording never depends on the gold answer: "Why not option X?" picks X
uniformly from all four options, so frozen queries stay valid when RQ4 swaps
in predicted answers.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from random import Random
from typing import Any

from datasets import Dataset

from clearmed.data.preprocessing import (
    OPTION_LABELS,
    get_answer_label,
    get_answer_text,
    get_question_and_options,
    is_nonblank_text,
    is_valid_qa_example,
)

STAGE_ANSWER_SOURCES: dict[str, frozenset[str]] = {
    "calibration": frozenset({"gold", "predicted"}),
    "rq3": frozenset({"gold", "predicted"}),
    "rq4": frozenset({"predicted"}),
    "demo": frozenset({"predicted"}),
}
ON_TOPIC_FOLLOW_UPS: dict[str, str] = {
    "why_correct": "Why is that the correct answer?",
    "why_not_option": "Why not option {option}?",
    "key_clue": "What is the key clue in this question?",
    "mechanism": "Can you explain the mechanism behind the answer?",
}
OFF_TOPIC_FOLLOW_UPS = (
    "What is the capital of France?",
    "Can you recommend a good movie for tonight?",
    "How do I fix a flat bicycle tyre?",
    "Who won the last football World Cup?",
    "How long should I boil pasta?",
    "Which programming language should I learn first?",
    "Can you write a short poem about the sea?",
    "How many planets are in the solar system?",
    "What time zone is Singapore in?",
    "How do I change my phone's wallpaper?",
)
SECOND_TURNS = (
    "How might this be tested differently in another exam question?",
    "What is a common mistake students make with this topic?",
    "Can you summarise the key point in one sentence?",
)
# Questions that point at an image, figure or plate we do not have.
VISUAL_REFERENCE = re.compile(
    r"\b(image|figure|fig\.|picture|photograph|photo|colou?r plate|diagram|shown (in|below|above|here))\b",
    re.IGNORECASE,
)


def _normalize_option(text: str) -> str:
    # Keep inner symbols: "Gram +ve" and "Gram -ve", or "p<0.05" and "p>0.05", are different answers.
    return " ".join(text.casefold().split()).strip(" .,;")


def is_usable_query_mcq(row: Mapping[str, Any]) -> bool:
    """A valid MCQ a student could answer from the text alone.

    Rejects rows whose four options are not distinct (e.g. arrows lost from
    "↑ TIBC") and questions that refer to an image we cannot show.
    """
    if not is_valid_qa_example(row):
        return False
    question, options = get_question_and_options(row)
    if len({_normalize_option(option) for option in options}) < len(OPTION_LABELS):
        return False
    return VISUAL_REFERENCE.search(question) is None


def build_query_text(record: Mapping[str, Any], answer_label: str) -> str:
    """Join follow-up, answer line and MCQ. Contains no explanation text."""
    if answer_label not in OPTION_LABELS:
        raise ValueError(f"Answer label must be one of {OPTION_LABELS}, got {answer_label!r}.")
    options = record["options"]
    option_lines = "\n".join(f"{label}. {options[label]}" for label in OPTION_LABELS)
    return (
        f"Follow-up: {record['follow_up']}\n"
        f"Answer: {answer_label}. {options[answer_label]}\n"
        f"Question: {record['question']}\n{option_lines}"
    )


def query_text_for(
    record: Mapping[str, Any],
    *,
    stage: str,
    answer_source: str,
    predicted_label: str | None = None,
) -> str:
    """Build a query, enforcing which answer source each stage may use."""
    allowed = STAGE_ANSWER_SOURCES.get(stage)
    if allowed is None:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {sorted(STAGE_ANSWER_SOURCES)}.")
    if answer_source not in allowed:
        raise ValueError(
            f"Stage {stage!r} may not use the {answer_source!r} answer; allowed: {sorted(allowed)}."
        )
    if answer_source == "gold":
        return build_query_text(record, record["gold_label"])
    if predicted_label is None:
        raise ValueError("A predicted answer label is required for predicted-answer queries.")
    return build_query_text(record, predicted_label)


def _query_record(row: Mapping[str, Any], *, query_id: str, query_set: str, source_split: str) -> dict[str, Any]:
    question, options = get_question_and_options(row)
    subject = row.get("subject_name")
    return {
        "query_id": query_id,
        "query_set": query_set,
        "source_split": source_split,
        "question_id": str(row["id"]),
        "question": question,
        "options": dict(zip(OPTION_LABELS, options, strict=True)),
        "subject": subject.strip() if is_nonblank_text(subject) else None,
        # Gold fields are for evaluation exports only; queries read them solely in gold mode.
        "gold_label": get_answer_label(row),
        "gold_answer_text": get_answer_text(row),
    }


def build_query_set(
    rows: Dataset,
    *,
    query_set: str,
    source_split: str,
    size: int,
    off_topic_share: float = 0.15,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Sample MCQs and attach follow-ups; identical inputs give identical output."""
    rng = Random(f"{seed}-{query_set}")
    candidates = [index for index, row in enumerate(rows) if is_usable_query_mcq(row)]
    if size > len(candidates):
        raise ValueError(f"Requested {size} queries but only {len(candidates)} usable MCQs exist.")
    chosen = rng.sample(candidates, size)
    off_topic_count = round(size * off_topic_share)
    on_topic_types = list(ON_TOPIC_FOLLOW_UPS)
    types = ["off_topic"] * off_topic_count + [
        on_topic_types[position % len(on_topic_types)] for position in range(size - off_topic_count)
    ]
    rng.shuffle(types)
    off_topic_texts = list(OFF_TOPIC_FOLLOW_UPS)
    rng.shuffle(off_topic_texts)

    records: list[dict[str, Any]] = []
    off_topic_seen = 0
    for position, (index, follow_up_type) in enumerate(zip(chosen, types, strict=True), start=1):
        record = _query_record(
            rows[index], query_id=f"{query_set}-{position:03d}", query_set=query_set, source_split=source_split
        )
        option_asked = None
        if follow_up_type == "off_topic":
            follow_up = off_topic_texts[off_topic_seen % len(off_topic_texts)]
            off_topic_seen += 1
        elif follow_up_type == "why_not_option":
            option_asked = rng.choice(OPTION_LABELS)
            follow_up = ON_TOPIC_FOLLOW_UPS[follow_up_type].format(option=option_asked)
        else:
            follow_up = ON_TOPIC_FOLLOW_UPS[follow_up_type]
        record.update(
            {
                "follow_up_type": follow_up_type,
                "follow_up": follow_up,
                "option_asked": option_asked,
                "rq4": False,
                "second_turn": None,
            }
        )
        records.append(record)
    return records


def mark_rq4_subset(
    records: Sequence[Mapping[str, Any]], *, size: int = 30, two_turn: int = 10, seed: int = 42
) -> list[dict[str, Any]]:
    """Flag the RQ4 subset and give some on-topic members a second turn."""
    if size > len(records):
        raise ValueError(f"RQ4 subset of {size} exceeds the {len(records)} evaluation queries.")
    rng = Random(f"{seed}-rq4")
    selected = set(rng.sample(range(len(records)), size))
    on_topic = sorted(index for index in selected if records[index]["follow_up_type"] != "off_topic")
    if two_turn > len(on_topic):
        raise ValueError(f"Only {len(on_topic)} on-topic RQ4 queries for {two_turn} second turns.")
    second_turn_indices = sorted(rng.sample(on_topic, two_turn))
    marked = [dict(record) for record in records]
    for index in selected:
        marked[index]["rq4"] = True
    for turn, index in enumerate(second_turn_indices):
        marked[index]["second_turn"] = SECOND_TURNS[turn % len(SECOND_TURNS)]
    return marked


def select_demo_questions(
    rows: Dataset, *, exclude_ids: set[str], size: int = 30, seed: int = 42
) -> list[str]:
    """Pick question IDs round-robin across subjects so the demo is varied."""
    rng = Random(f"{seed}-demo")
    by_subject: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        row_id = str(row["id"])
        if row_id in exclude_ids or not is_usable_query_mcq(row):
            continue
        subject = row.get("subject_name")
        by_subject[subject.strip() if is_nonblank_text(subject) else "Unknown"].append(row_id)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    for subject in subjects:
        rng.shuffle(by_subject[subject])
    chosen: list[str] = []
    while len(chosen) < size and any(by_subject[subject] for subject in subjects):
        for subject in subjects:
            if by_subject[subject] and len(chosen) < size:
                chosen.append(by_subject[subject].pop())
    if len(chosen) < size:
        raise ValueError(f"Only {len(chosen)} eligible demo questions for a set of {size}.")
    return chosen
