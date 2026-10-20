"""FLAN-T5 input/target preparation for ClearMed Part I.

Sources are derived solely from question and option fields.  Gold answers and
reference explanations are only used while forming supervised targets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from clearmed.data.preprocessing import (
    OPTION_LABELS,
    format_generation_input,
    get_answer_label,
    get_question_and_options,
    is_nonblank_text,
    is_valid_qa_example,
)


@dataclass(frozen=True)
class FlanT5DataConfig:
    """Recorded token limits for the pretrained text-to-text model."""

    max_source_length: int = 512
    max_target_length: int = 192

    def __post_init__(self) -> None:
        for name in ("max_source_length", "max_target_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")


def format_flan_t5_target(row: Mapping[str, Any]) -> str:
    """Build a supervised answer target, without inventing an explanation."""
    if not is_valid_qa_example(row):
        raise ValueError("A FLAN-T5 target requires a QA-valid MCQ row.")
    target = f"Answer: {get_answer_label(row)}"
    if is_nonblank_text(row.get("exp")):
        target += f"\nExplanation: {row['exp'].strip()}"
    return target


def _render_source(question: str, options: Sequence[str]) -> str:
    option_lines = "\n".join(
        f"{label}. {option}" for label, option in zip(OPTION_LABELS, options, strict=True)
    )
    return (
        "Answer the following medical exam question and explain your answer.\n\n"
        f"Question:\n{question}\n\nOptions:\n{option_lines}"
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"]
    return list(ids)


def _within_limit(tokenizer: Any, text: str, maximum: int) -> bool:
    """Count source tokens exactly as the model-facing tokenizer call does."""
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"]) <= maximum


def format_flan_t5_input(
    row: Mapping[str, Any], tokenizer: Any, config: FlanT5DataConfig
) -> tuple[str, bool]:
    """Format a source, truncating individual content segments when needed.

    The normal path is exactly :func:`format_generation_input`.  A long source
    is shortened token-by-token across the question and all four options so an
    option can never silently disappear.  A limit too small even for one token
    per segment is rejected with an actionable error.
    """
    question, options = get_question_and_options(row)
    canonical = format_generation_input(row)
    if _within_limit(tokenizer, canonical, config.max_source_length):
        return canonical, False

    segment_ids = [_token_ids(tokenizer, question), *(_token_ids(tokenizer, value) for value in options)]
    if any(not values for values in segment_ids):
        raise ValueError("Each question and option must contain at least one tokenizer token.")
    selected = [1] * len(segment_ids)

    def render() -> str:
        pieces = [tokenizer.decode(ids[:size], skip_special_tokens=True).strip() for ids, size in zip(segment_ids, selected, strict=True)]
        return _render_source(pieces[0], pieces[1:])

    minimum = render()
    if not _within_limit(tokenizer, minimum, config.max_source_length):
        raise ValueError(
            "max_source_length is too small to preserve the question and all four options; "
            "increase it rather than dropping options."
        )

    # Deterministic round-robin allocation preserves every segment before
    # giving additional room to any one of them.
    while True:
        grew = False
        for index, values in enumerate(segment_ids):
            if selected[index] >= len(values):
                continue
            selected[index] += 1
            if _within_limit(tokenizer, render(), config.max_source_length):
                grew = True
            else:
                selected[index] -= 1
        if not grew:
            break
    return render(), True


@dataclass(frozen=True)
class FlanT5Example:
    example_id: str
    source: str
    target: str | None
    source_truncated: bool
    has_explanation: bool


class FlanT5MCQDataset(torch.utils.data.Dataset[FlanT5Example]):
    """QA-valid FLAN-T5 examples, retaining invalid-row accounting."""

    def __init__(
        self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, config: FlanT5DataConfig, *, with_targets: bool = True
    ) -> None:
        self.examples: list[FlanT5Example] = []
        self.skipped_invalid = 0
        for row in rows:
            if not is_valid_qa_example(row):
                self.skipped_invalid += 1
                continue
            source, truncated = format_flan_t5_input(row, tokenizer, config)
            has_explanation = is_nonblank_text(row.get("exp"))
            self.examples.append(
                FlanT5Example(
                    example_id=str(row.get("id")), source=source,
                    target=format_flan_t5_target(row) if with_targets else None,
                    source_truncated=truncated, has_explanation=has_explanation,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> FlanT5Example:
        return self.examples[index]

    def statistics(self) -> dict[str, int | float]:
        total = len(self.examples)
        with_explanation = sum(example.has_explanation for example in self.examples)
        return {
            "examples": total,
            "skipped_invalid_rows": self.skipped_invalid,
            "with_explanation": with_explanation,
            "answer_only_targets": total - with_explanation,
            "explanation_coverage": with_explanation / total if total else 0.0,
            "source_truncated_count": sum(example.source_truncated for example in self.examples),
            "source_truncated_rate": sum(example.source_truncated for example in self.examples) / total if total else 0.0,
        }


def collate_flan_t5_examples(
    examples: Sequence[FlanT5Example], tokenizer: Any, config: FlanT5DataConfig
) -> dict[str, Any]:
    """Pad sources and masked labels in the format expected by T5."""
    if not examples:
        raise ValueError("Cannot collate an empty FLAN-T5 batch.")
    encoded = tokenizer(
        [example.source for example in examples], padding=True, truncation=False, return_tensors="pt",
    )
    if encoded["input_ids"].shape[1] > config.max_source_length:
        raise RuntimeError("FLAN-T5 source formatting exceeded its configured maximum length.")
    batch: dict[str, Any] = {"example_ids": [example.example_id for example in examples], **dict(encoded)}
    if all(example.target is not None for example in examples):
        target = tokenizer(
            [example.target for example in examples], max_length=config.max_target_length,
            padding=True, truncation=True, return_tensors="pt",
        )["input_ids"]
        batch["labels"] = target.masked_fill(target.eq(tokenizer.pad_token_id), -100)
    return batch
