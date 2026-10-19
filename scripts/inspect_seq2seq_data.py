"""Inspect explanation availability and token lengths in part1_train.

Use this to choose Seq2Seq truncation limits from data rather than guesses:

    uv run python scripts/inspect_seq2seq_data.py
    uv run python scripts/inspect_seq2seq_data.py --output outputs/seq2seq/data_inspection.json

Only the training partition is read (internal_dev must not drive preprocessing).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk

from clearmed.data.preprocessing import (
    get_question_and_options,
    is_nonblank_text,
    is_valid_qa_example,
)
from clearmed.data.seq2seq import Seq2SeqDataConfig, tokenize

PERCENTILES = (50, 75, 90, 95, 99)


def length_summary(lengths: Sequence[int], limits: Sequence[int]) -> dict[str, Any]:
    if not lengths:
        return {"count": 0}
    array = np.asarray(lengths)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "max": int(array.max()),
        **{f"p{p}": float(np.percentile(array, p)) for p in PERCENTILES},
        "share_over_limit": {str(limit): float((array > limit).mean()) for limit in limits},
    }


def inspect_rows(rows, limits: Seq2SeqDataConfig) -> dict[str, Any]:
    total = valid = with_exp = 0
    question_lengths: list[int] = []
    option_lengths: list[int] = []
    explanation_lengths: list[int] = []
    short_explanations = 0
    for row in rows:
        total += 1
        if not is_valid_qa_example(row):
            continue
        valid += 1
        question, options = get_question_and_options(row)
        question_lengths.append(len(tokenize(question)))
        option_lengths.extend(len(tokenize(option)) for option in options)
        if is_nonblank_text(row.get("exp")):
            with_exp += 1
            length = len(tokenize(row["exp"]))
            explanation_lengths.append(length)
            short_explanations += length < 5
    return {
        "rows": total,
        "qa_valid_rows": valid,
        "rows_with_explanation": with_exp,
        "explanation_coverage": with_exp / valid if valid else 0.0,
        "very_short_explanations_lt5_tokens": short_explanations,
        "question_tokens": length_summary(question_lengths, (limits.max_question_tokens, 64, 128)),
        "option_tokens": length_summary(option_lengths, (limits.max_option_tokens, 16, 32)),
        "explanation_tokens": length_summary(explanation_lengths, (limits.max_explanation_tokens, 64, 256)),
        "current_default_limits": {
            "max_question_tokens": limits.max_question_tokens,
            "max_option_tokens": limits.max_option_tokens,
            "max_explanation_tokens": limits.max_explanation_tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--output", type=Path, help="Optional JSON path (refuses to overwrite).")
    args = parser.parse_args()
    report = inspect_rows(load_from_disk(str(args.train_dir)), Seq2SeqDataConfig())
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"{args.output} exists; choose another path.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
