"""Generate the frozen follow-up query sets and the demo question set.

Writes 80 calibration queries (internal_dev), 60 evaluation queries (official
validation, 30 flagged for RQ4 with 10 second turns) and 30 demo question IDs.
Evaluation and demo questions are drawn only from validation questions with no
duplicate in part1_train or retrieval_kb. Existing files are never replaced;
delete them deliberately to regenerate.
"""

from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk

from clearmed.data.loader import load_medmcqa_split
from clearmed.data.splitting import membership_fingerprint
from clearmed.data.verification import (
    find_duplicates,
    load_manifest,
    verify_against_manifest,
)
from clearmed.retrieval.queries import (
    STAGE_ANSWER_SOURCES,
    build_query_set,
    mark_rq4_subset,
    select_demo_questions,
)

CANONICAL_SPLITS_DIR = Path("data/splits")
QUERY_DIR = Path("data/retrieval_eval")
DEMO_PATH = Path("configs/demo/demo_questions.json")
SEED = 42
SIZES = {"calibration": 80, "evaluation": 60, "rq4": 30, "two_turn": 10, "demo": 30}


def _load_verified(splits_dir: Path, name: str, manifest: dict[str, Any]) -> Dataset:
    path = splits_dir / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {name} at {path}; generate the splits first.")
    dataset = load_from_disk(str(path))
    verify_against_manifest(dataset, name, manifest)
    return dataset


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    body = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    path.write_text(body, encoding="utf-8", newline="\n")
    return sha256(body.encode("utf-8")).hexdigest()


def run(
    *, splits_dir: Path, validation: Dataset, query_dir: Path, demo_path: Path, sizes: dict[str, int] = SIZES
) -> dict[str, Any]:
    outputs = [
        query_dir / "queries_calibration.jsonl",
        query_dir / "queries_eval.jsonl",
        query_dir / "query_sets.json",
        demo_path,
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Query sets are frozen; refusing to overwrite {existing}.")

    manifest = load_manifest(splits_dir / "manifest.json")
    internal_dev = _load_verified(splits_dir, "internal_dev", manifest)
    part1_train = _load_verified(splits_dir, "part1_train", manifest)
    retrieval_kb = _load_verified(splits_dir, "retrieval_kb", manifest)

    # D4 layer 2: evaluation and demo questions never duplicate training or KB questions.
    duplicate_indices = {
        item["index"] for reference in (part1_train, retrieval_kb) for item in find_duplicates(validation, reference)
    }
    clean_validation = validation.select(
        [index for index in range(len(validation)) if index not in duplicate_indices]
    )

    calibration = build_query_set(
        internal_dev, query_set="calibration", source_split="internal_dev", size=sizes["calibration"], seed=SEED
    )
    evaluation = mark_rq4_subset(
        build_query_set(
            clean_validation, query_set="evaluation", source_split="validation", size=sizes["evaluation"], seed=SEED
        ),
        size=sizes["rq4"],
        two_turn=sizes["two_turn"],
        seed=SEED,
    )
    evaluation_ids = {record["question_id"] for record in evaluation}
    selected = Dataset.from_list(
        [{"id": record["question_id"], "question": record["question"]} for record in evaluation]
    )
    if find_duplicates(selected, part1_train) or find_duplicates(selected, retrieval_kb):
        raise ValueError("An evaluation question duplicates part1_train or retrieval_kb.")
    demo_ids = select_demo_questions(
        clean_validation, exclude_ids=evaluation_ids, size=sizes["demo"], seed=SEED
    )

    query_dir.mkdir(parents=True, exist_ok=True)
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    checksums = {
        "queries_calibration.jsonl": _write_jsonl(outputs[0], calibration),
        "queries_eval.jsonl": _write_jsonl(outputs[1], evaluation),
    }
    demo = {
        "seed": SEED,
        "source_split": "validation",
        "rule": "Random, balanced across subjects; excludes evaluation questions and part1_train/retrieval_kb duplicates.",
        "question_ids": demo_ids,
    }
    demo_path.write_text(json.dumps(demo, indent=2) + "\n", encoding="utf-8")

    summary = {
        "seed": SEED,
        "sizes": sizes,
        "query_format": "Follow-up, then 'Answer: <label>. <option text>', then the question and options A-D.",
        "mcq_filter": "Valid MCQ with four distinct options and no reference to an image, figure or plate.",
        "stage_answer_sources": {stage: sorted(sources) for stage, sources in STAGE_ANSWER_SOURCES.items()},
        "follow_up_types": {
            "calibration": dict(sorted(Counter(record["follow_up_type"] for record in calibration).items())),
            "evaluation": dict(sorted(Counter(record["follow_up_type"] for record in evaluation).items())),
        },
        "rq4_queries": sum(record["rq4"] for record in evaluation),
        "rq4_second_turns": sum(record["second_turn"] is not None for record in evaluation),
        "validation_rows": len(validation),
        "validation_fingerprint": membership_fingerprint(validation),
        "validation_duplicates_excluded": len(duplicate_indices),
        "internal_dev_fingerprint": manifest["partitions"]["internal_dev"]["membership_fingerprint"],
        "file_sha256": checksums,
        "demo_questions": len(demo_ids),
    }
    outputs[2].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    summary = run(
        splits_dir=CANONICAL_SPLITS_DIR,
        validation=load_medmcqa_split("validation"),
        query_dir=QUERY_DIR,
        demo_path=DEMO_PATH,
    )
    print(json.dumps(summary["follow_up_types"], indent=2))
    print(
        f"RQ4 queries: {summary['rq4_queries']} ({summary['rq4_second_turns']} with a second turn); "
        f"demo questions: {summary['demo_questions']}; "
        f"validation duplicates excluded: {summary['validation_duplicates_excluded']}."
    )


if __name__ == "__main__":
    main()
