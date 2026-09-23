"""Read-only, reproducible audit of the MedMCQA dataset."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from clearmed.data.loader import DATASET_NAME, load_medmcqa
from clearmed.data.preprocessing import (
    ANSWER_INDEX_TO_LABEL,
    OPTION_FIELDS,
    is_nonblank_text,
    is_valid_generation_example,
    is_valid_qa_example,
    normalize_question_for_comparison,
)


def _blank_counts(rows: Any) -> Counter[str]:
    fields = ("question", *OPTION_FIELDS, "exp", "subject_name", "topic_name")
    return Counter({field: sum(not is_nonblank_text(value) for value in rows[field]) for field in fields})


def _question_index(rows: Any) -> dict[str, list[str]]:
    indexed: dict[str, list[str]] = defaultdict(list)
    for row_id, question in zip(rows["id"], rows["question"], strict=True):
        if isinstance(question, str):
            indexed[normalize_question_for_comparison(question)].append(row_id)
    return indexed


def main() -> None:
    print(f"Loading {DATASET_NAME} once...\n")
    dataset = load_medmcqa()
    split_indexes: dict[str, dict[str, list[str]]] = {}

    print("=== A. DATASET SPLITS ===")
    for name, split in dataset.items():
        print(f"{name}: {len(split)} rows")

    print("\n=== B. DATASET SCHEMA ===")
    for name, split in dataset.items():
        print(f"{name} fields: {list(split.features)}")
        print(f"{name} features: {split.features}")
    cop = dataset["train"].features["cop"]
    choice_type = dataset["train"].features["choice_type"]
    print(f"cop metadata: {cop}")
    print(f"choice_type metadata: {choice_type}")
    print("Verified answer-index mapping:")
    for index, label in ANSWER_INDEX_TO_LABEL.items():
        print(f"  {index} -> {label}")

    print("\n=== C-G. PER-SPLIT ANALYSIS ===")
    for name, split in dataset.items():
        choice_counts = Counter(split["choice_type"])
        answer_counts: Counter[str] = Counter()
        invalid_answers = 0
        columns = {field: split[field] for field in ("question", *OPTION_FIELDS, "cop", "exp")}
        for answer in columns["cop"]:
            if isinstance(answer, bool) or not isinstance(answer, int) or answer not in ANSWER_INDEX_TO_LABEL:
                invalid_answers += 1
            else:
                answer_counts[ANSWER_INDEX_TO_LABEL[answer]] += 1
        blanks = _blank_counts(split)
        qa_valid = 0
        generation_valid = 0
        for values in zip(*(columns[field] for field in columns), strict=True):
            row = dict(zip(columns, values, strict=True))
            qa_valid += is_valid_qa_example(row)
            generation_valid += is_valid_generation_example(row)
        print(f"\n--- {name} ---")
        print(f"choice_type values/counts: {dict(sorted(choice_counts.items()))}")
        for choice_value in sorted(choice_counts):
            sample_index = split["choice_type"].index(choice_value)
            print(f"sample {choice_value!r}: id={split['id'][sample_index]}, question={split['question'][sample_index]!r}")
        print("answer distribution: " + ", ".join(f"{label}={answer_counts[label]}" for label in "ABCD"))
        print(f"invalid/missing cop: {invalid_answers}")
        print("blank content: " + ", ".join(f"{field}={blanks[field]}" for field in blanks))
        print(f"Valid QA examples: {qa_valid} / {len(split)}")
        print(f"Valid generation examples: {generation_valid} / {len(split)}")
        split_indexes[name] = _question_index(split)

    print("\n=== H. TRAIN SUBJECT ANALYSIS ===")
    subjects = Counter(subject for subject in dataset["train"]["subject_name"] if is_nonblank_text(subject))
    print(f"unique nonblank subjects: {len(subjects)}")
    print(f"top 10 subjects: {subjects.most_common(10)}")

    print("\n=== I. DUPLICATE QUESTION ANALYSIS ===")
    for name, index in split_indexes.items():
        duplicate_keys = {question: ids for question, ids in index.items() if len(ids) > 1}
        duplicate_rows = sum(len(ids) - 1 for ids in duplicate_keys.values())
        print(f"{name}: {len(duplicate_keys)} duplicate normalized questions; {duplicate_rows} rows beyond first")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(split_indexes[left]) & set(split_indexes[right])
        print(f"{left} intersection {right}: {len(overlap)} overlapping unique normalized questions")
        for question in sorted(overlap)[:3]:
            print(f"  sample: {question!r}; {left} IDs={split_indexes[left][question][:3]}; {right} IDs={split_indexes[right][question][:3]}")

    print("\n=== J. SUMMARY ===")
    print(f"verified cop mapping: {ANSWER_INDEX_TO_LABEL}")
    print("choice_type values: " + repr(sorted({value for split in dataset.values() for value in split["choice_type"]})))
    print("QA validity and generation validity counts are listed above for each official split.")
    print("No splits were merged; future retrieval must use training evidence only.")


if __name__ == "__main__":
    main()
