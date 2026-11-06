"""Build the Part II evidence corpus from the retrieval knowledge base.

Each passage is one MedMCQA training question with its correct answer text and
expert explanation. Distractor options and the bare ``cop`` label are dropped.
Passages that duplicate an evaluation question are removed before any
embedding, so retrieval can never return an evaluation item's own answer key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from datasets import Dataset

from clearmed.data.preprocessing import (
    get_answer_text,
    get_question_and_options,
    is_nonblank_text,
    is_valid_generation_example,
)
from clearmed.data.splitting import assert_train_dev_disjoint
from clearmed.data.verification import (
    EVIDENCE_PARTITION,
    find_duplicates,
    verify_against_manifest,
)

PASSAGE_TEMPLATE = "Question: {question}\nAnswer: {answer}\nExplanation: {explanation}"
REMOVED_EXAMPLE_LIMIT = 10


def _optional_text(value: Any) -> str | None:
    return value.strip() if is_nonblank_text(value) else None


def build_passage(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one explanation-bearing MedMCQA row into a retrieval passage."""
    if not is_valid_generation_example(row):
        raise ValueError(
            f"Row {row.get('id')!r} needs a question, four options, a valid answer and an explanation."
        )
    question, _ = get_question_and_options(row)
    answer = get_answer_text(row)
    explanation = row["exp"].strip()
    return {
        "doc_id": str(row["id"]),
        "text": PASSAGE_TEMPLATE.format(question=question, answer=answer, explanation=explanation),
        "question": question,
        "answer": answer,
        "explanation": explanation,
        "subject": _optional_text(row.get("subject_name")),
        "topic": _optional_text(row.get("topic_name")),
    }


def build_corpus(
    evidence: Dataset,
    *,
    manifest: Mapping[str, Any],
    evaluation: Dataset,
    evaluation_name: str = "validation",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify, de-duplicate and format the evidence partition.

    Returns the passages and a JSON-serialisable report. Raises ``ValueError``
    when the evidence differs from the manifest or when any overlap with
    *evaluation* survives filtering.
    """
    verification = verify_against_manifest(evidence, EVIDENCE_PARTITION, manifest)
    duplicates = find_duplicates(evidence, evaluation)
    removed_indices = {item["index"] for item in duplicates}
    kept = evidence.select(
        [index for index in range(len(evidence)) if index not in removed_indices]
    )
    assert_train_dev_disjoint(
        kept,
        evaluation,
        train_name="filtered retrieval corpus",
        development_name=evaluation_name,
    )
    passages = [build_passage(row) for row in kept]
    report = {
        "manifest_verification": verification,
        "evaluation_split": evaluation_name,
        "rows_in": len(evidence),
        "rows_removed": len(duplicates),
        "removed_examples": [
            {key: item[key] for key in ("id", "question", "matched_by")}
            for item in duplicates[:REMOVED_EXAMPLE_LIMIT]
        ],
        "passages_out": len(passages),
        "overlap_after_filtering": {"row_id": 0, "normalized_question": 0},
        "passage_template": PASSAGE_TEMPLATE,
    }
    return passages, report


def load_corpus(corpus_run: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read passages from a corpus run after checking they are unaltered."""
    config_path = corpus_run / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No corpus run at {corpus_run}. Build one with `uv run python scripts/build_retrieval_corpus.py`."
        )
    corpus_config = json.loads(config_path.read_text(encoding="utf-8"))
    passages_path = corpus_run / corpus_config["passages_file"]
    digest = sha256(passages_path.read_bytes()).hexdigest()
    if digest != corpus_config["passages_sha256"]:
        raise ValueError(f"{passages_path} does not match its recorded checksum; rebuild the corpus.")
    with passages_path.open(encoding="utf-8") as handle:
        passages = [json.loads(line) for line in handle]
    return passages, corpus_config
