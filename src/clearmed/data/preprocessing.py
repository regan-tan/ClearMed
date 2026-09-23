"""Canonical MedMCQA validation and text formatting helpers.

The numeric answer mapping is verified from MedMCQA's ``cop`` ClassLabel:
``['a', 'b', 'c', 'd']``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

OPTION_FIELDS = ("opa", "opb", "opc", "opd")
OPTION_LABELS = ("A", "B", "C", "D")
ANSWER_INDEX_TO_LABEL = dict(enumerate(OPTION_LABELS))


def is_nonblank_text(value: Any) -> bool:
    """Return whether *value* is a string with non-whitespace content."""
    return isinstance(value, str) and bool(value.strip())


def get_answer_index(row: Mapping[str, Any]) -> int:
    """Return the verified MedMCQA answer index or raise ValueError."""
    answer_index = row.get("cop")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ValueError("cop must be an integer MedMCQA answer index.")  # noqa: TRY004
    if answer_index not in ANSWER_INDEX_TO_LABEL:
        raise ValueError("cop must be a verified answer index from 0 through 3.")
    return answer_index


def get_answer_label(row: Mapping[str, Any]) -> str:
    """Return the canonical A/B/C/D label for a row's gold answer."""
    return ANSWER_INDEX_TO_LABEL[get_answer_index(row)]


def get_answer_text(row: Mapping[str, Any]) -> str:
    """Return the nonblank option text selected by the gold answer."""
    option_field = OPTION_FIELDS[get_answer_index(row)]
    option_text = row.get(option_field)
    if not is_nonblank_text(option_text):
        raise ValueError(f"Correct answer option {option_field!r} must be nonblank text.")
    return option_text.strip()


def _get_question_and_options(row: Mapping[str, Any]) -> tuple[str, tuple[str, str, str, str]]:
    question = row.get("question")
    if not is_nonblank_text(question):
        raise ValueError("question must be nonblank text.")
    options = tuple(row.get(field) for field in OPTION_FIELDS)
    if not all(is_nonblank_text(option) for option in options):
        raise ValueError("All four answer options must be nonblank text.")
    return question.strip(), tuple(option.strip() for option in options)  # type: ignore[return-value]


def format_question(row: Mapping[str, Any]) -> str:
    """Format a question and its four options without exposing gold information."""
    question, options = _get_question_and_options(row)
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(OPTION_LABELS, options, strict=True)
    )
    return f"Question:\n{question}\n\nOptions:\n{option_lines}"


def format_generation_input(row: Mapping[str, Any]) -> str:
    """Format the canonical future text-to-text input, without gold information."""
    return "Answer the following medical exam question and explain your answer.\n\n" + format_question(row)


def format_generation_target(row: Mapping[str, Any]) -> str:
    """Format a gold answer and explanation for a generation-valid row."""
    if not is_valid_generation_example(row):
        raise ValueError("A generation target requires a generation-valid row.")
    return f"Answer: {get_answer_label(row)}\nExplanation: {row['exp'].strip()}"


def is_valid_qa_example(row: Mapping[str, Any]) -> bool:
    """Return whether a row has all content required for QA evaluation."""
    try:
        _get_question_and_options(row)
        get_answer_index(row)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def is_valid_generation_example(row: Mapping[str, Any]) -> bool:
    """Return whether a QA-valid row also has a usable expert explanation."""
    return is_valid_qa_example(row) and is_nonblank_text(row.get("exp"))


def normalize_question_for_comparison(text: str) -> str:
    """Conservatively normalize text for exact question-duplicate comparisons."""
    if not isinstance(text, str):
        raise TypeError("Question text must be a string.")
    return " ".join(text.strip().split()).casefold()
