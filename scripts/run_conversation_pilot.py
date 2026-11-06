"""Generator pilot for the tutor pipeline (Plan 6).

Runs on-topic follow-ups from the calibration queries (internal_dev, never
validation) through the full pipeline with a FLAN-T5 + LoRA checkpoint, using
the checkpoint's own predicted answer. Candidates are taken in a seeded,
type-balanced order until ``--n`` replies pass the gate; fallbacks and
malformed predictions are counted, not graded.

Writes ``outputs/conversation/pilot/<run_id>/``:

* ``turns.jsonl``: every attempted follow-up with the full pipeline log;
* ``pilot_sheet.csv``: the graded replies, with blank ``addresses_follow_up``
  and ``uses_new_evidence_fact`` columns (0/1) to fill in;
* ``config.json``.

After grading, ``--score`` reads ``pilot_sheet.csv`` and writes
``pilot_result.json``. A reply passes if it addresses the follow-up and uses
at least one fact from the evidence that is not in the MCQ; the pilot passes
when at least 60% of replies pass (6 of 10).

``--gate-off`` answers every follow-up with its retrieved evidence, so a pilot
of the reply model is not limited to the follow-ups the gate accepts. Two
gate-off pilots then see the same follow-ups in the same order, and
``--compare RUN_A RUN_B`` gives a paired comparison of their graded sheets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Any

from clearmed.conversation.pipeline import (
    GATE_OFF_SUFFIX,
    TutorPipeline,
    evidence_block,
    mcq_row,
)
from clearmed.evaluation.retrieval import bootstrap_ci, paired_test
from clearmed.retrieval.queries import ON_TOPIC_FOLLOW_UPS

QUERIES_PATH = Path("data/retrieval_eval/queries_calibration.jsonl")
OUTPUT_ROOT = Path("outputs/conversation/pilot")
SEED = 42
PASS_RATE = 0.6  # 6 of 10, fixed before the first pilot.
SHEET_COLUMNS = (
    "query_id", "follow_up_type", "question", "options", "predicted_label", "gold_label",
    "follow_up", "evidence", "reply", "stray_answer_letter",
    "addresses_follow_up", "uses_new_evidence_fact", "notes",
)
# Answer-letter prefixes copied from MedMCQA explanations, e.g. "Ans. (c)" or "Answer is B".
# A bare lowercase letter counts only before punctuation or "i.e.", so "the answer is a hormone" does not match.
STRAY_ANSWER_LETTER = re.compile(
    r"\b(?i:ans(?:wer)?)\b[\s.:\-]*(?:(?i:is)\b[\s.:\-]*)?"
    r"(?:\(\s*['\"]?[a-dA-D]['\"]?\s*\)|['\"‘’][a-dA-D]['\"‘’]|[A-D]\b|[a-d](?=[.,:;)]|\s*$|\s+i\.?e))"
)


def candidate_order(records: Sequence[Mapping[str, Any]], seed: int = SEED) -> list[Mapping[str, Any]]:
    """On-topic records, shuffled per follow-up type and interleaved round-robin."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record["follow_up_type"] in ON_TOPIC_FOLLOW_UPS:
            groups[record["follow_up_type"]].append(record)
    rng = Random(seed)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    ordered: list[Mapping[str, Any]] = []
    for position in range(max((len(g) for g in groups.values()), default=0)):
        ordered += [groups[name][position] for name in sorted(groups) if position < len(groups[name])]
    return ordered


def explanation_of(reply: str) -> str:
    return reply.split("Explanation:", 1)[1].strip() if "Explanation:" in reply else reply


def run(
    *,
    records: Sequence[Mapping[str, Any]],
    tutor: TutorPipeline,
    predict: Callable[[list[dict[str, Any]]], list[str | None]],
    output_dir: Path,
    n: int = 10,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new --run-id.")
    candidates = candidate_order(records)
    predicted = predict([{"id": str(r["question_id"]), **mcq_row(r)} for r in candidates])
    turns: list[dict[str, Any]] = []
    graded: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    counts = {"attempted": 0, "malformed_prediction": 0, "fallback": 0}
    for record, label in zip(candidates, predicted, strict=True):
        if len(graded) >= n:
            break
        counts["attempted"] += 1
        if label is None:
            counts["malformed_prediction"] += 1
            turns.append({"query_id": record["query_id"], "predicted_label": None, "skipped": "malformed_prediction"})
            continue
        result = tutor.respond(record, predicted_label=label)
        turns.append(result)
        if result["fallback"]:
            counts["fallback"] += 1
        else:
            graded.append((record, result))
    if len(graded) < n:
        raise RuntimeError(f"Only {len(graded)} of {n} follow-ups passed the gate; nothing written.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "turns.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for turn in turns:
            handle.write(json.dumps(turn, sort_keys=True) + "\n")
    with (output_dir / "pilot_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for record, result in graded:
            writer.writerow({
                "query_id": record["query_id"],
                "follow_up_type": record["follow_up_type"],
                "question": record["question"],
                "options": " | ".join(f"{k}. {v}" for k, v in record["options"].items()),
                "predicted_label": result["predicted_label"],
                "gold_label": record["gold_label"],
                "follow_up": result["follow_up"],
                "evidence": evidence_block(result["prompt"]),
                "reply": result["reply"],
                "stray_answer_letter": int(bool(STRAY_ANSWER_LETTER.search(explanation_of(result["reply"])))),
                "addresses_follow_up": "", "uses_new_evidence_fact": "", "notes": "",
            })
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "method_id": tutor.method_id,
        "gate_off": tutor.gate_off,
        "threshold": tutor.gate.threshold if tutor.gate else None,
        "stage": tutor.stage,
        "seed": SEED,
        "n_graded": len(graded),
        **counts,
        "replies_with_stray_answer_letter": sum(
            bool(STRAY_ANSWER_LETTER.search(explanation_of(r["reply"]))) for _, r in graded
        ),
        "predicted_correct": sum(r["predicted_label"] == rec["gold_label"] for rec, r in graded),
        **dict(metadata or {}),
    }
    (output_dir / "config.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    return summary


def _graded_rows(output_dir: Path) -> list[dict[str, str]]:
    with (output_dir / "pilot_sheet.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for column in ("addresses_follow_up", "uses_new_evidence_fact"):
            if row[column].strip() not in ("0", "1"):
                raise ValueError(f"{row['query_id']}: {column} must be 0 or 1, got {row[column]!r}.")
    return rows


def _passes(row: Mapping[str, str]) -> bool:
    return row["addresses_follow_up"].strip() == row["uses_new_evidence_fact"].strip() == "1"


def score(output_dir: Path) -> dict[str, Any]:
    """Tally a graded ``pilot_sheet.csv`` into ``pilot_result.json``."""
    rows = _graded_rows(output_dir)
    passed = [r["query_id"] for r in rows if _passes(r)]
    pass_mark = math.ceil(PASS_RATE * len(rows))
    result = {
        "n": len(rows),
        "passed": len(passed),
        "pass_rate": len(passed) / len(rows) if rows else None,
        "pass_rate_ci95": list(bootstrap_ci([float(_passes(r)) for r in rows])) if rows else None,
        "passed_ids": passed,
        "addresses_follow_up": sum(r["addresses_follow_up"].strip() == "1" for r in rows),
        "uses_new_evidence_fact": sum(r["uses_new_evidence_fact"].strip() == "1" for r in rows),
        "stray_answer_letter": sum(r["stray_answer_letter"] == "1" for r in rows),
        "pass_mark": pass_mark,
        "pilot_passed": len(passed) >= pass_mark,
    }
    (output_dir / "pilot_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def compare(dir_a: Path, dir_b: Path) -> dict[str, Any]:
    """Paired pass/fail comparison of two graded pilots on the same follow-ups (prints only)."""
    rows_a = {r["query_id"]: _passes(r) for r in _graded_rows(dir_a)}
    rows_b = {r["query_id"]: _passes(r) for r in _graded_rows(dir_b)}
    if set(rows_a) != set(rows_b):
        raise ValueError("The two pilots graded different follow-ups; compare gate-off pilots of the same size.")
    ids = sorted(rows_a)
    return {
        "a": dir_a.name,
        "b": dir_b.name,
        "n": len(ids),
        "passed_a": sum(rows_a.values()),
        "passed_b": sum(rows_b.values()),
        "only_a_passed": sum(rows_a[i] and not rows_b[i] for i in ids),
        "only_b_passed": sum(rows_b[i] and not rows_a[i] for i in ids),
        "a_minus_b": paired_test([float(rows_a[i]) for i in ids], [float(rows_b[i]) for i in ids]),
    }


def _real_components(args: argparse.Namespace) -> tuple[TutorPipeline, Callable, dict[str, Any]]:
    import torch

    from clearmed.conversation.pipeline import DenseRetriever, FlanT5Generator
    from clearmed.retrieval.corpus import load_corpus
    from clearmed.retrieval.embedders import (
        Embedder,
        SentenceTransformerEncoder,
        load_embedding_config,
    )
    from clearmed.retrieval.gate import Gate
    from clearmed.retrieval.index import DenseIndex
    from clearmed.training.flan_t5_lora import FlanT5Predictor, load_adapter_checkpoint

    device = torch.device(args.device)
    model, tokenizer, data_config, adapter_metadata = load_adapter_checkpoint(args.adapter_dir, device)
    passages, _ = load_corpus(args.corpus_run)
    config = load_embedding_config(args.method_id)
    retriever = DenseRetriever(
        Embedder(config, SentenceTransformerEncoder(config)),
        DenseIndex.load(Path("outputs/retrieval") / args.method_id / args.index_run),
        passages,
    )
    generator = FlanT5Generator(model, tokenizer, max_input_tokens=data_config.max_source_length, device=device,
                                no_repeat_ngram_size=args.no_repeat_ngram_size, banned_phrases=args.ban_phrase,
                                use_adapter=not args.reply_without_adapter)
    tutor = TutorPipeline(generator, stage="calibration", retriever=retriever,
                          gate=None if args.gate_off else Gate.load(args.method_id),
                          method_id=f"{args.method_id}{GATE_OFF_SUFFIX}" if args.gate_off else args.method_id,
                          gate_off=args.gate_off,
                          clean_evidence=args.clean_evidence, state_answer=args.state_answer)
    predictor = FlanT5Predictor(model, tokenizer, data_config, device)

    def predict(rows: list[dict[str, Any]]) -> list[str | None]:
        return [p["predicted_answer"] for p in predictor.generate_batch(rows)]

    metadata = {
        "adapter_dir": Path(args.adapter_dir).as_posix(),
        "model_name": adapter_metadata["model_name"],
        "max_source_length": data_config.max_source_length,
        "index_run": args.index_run,
        "corpus_run": Path(args.corpus_run).as_posix(),
        "queries": QUERIES_PATH.as_posix(),
        "clean_evidence": args.clean_evidence,
        "state_answer": args.state_answer,
        "generation": generator.settings,
    }
    return tutor, predict, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--score", action="store_true", help="Tally a graded pilot_sheet.csv.")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--method-id", default="minilm")
    parser.add_argument("--index-run", default="full-20261016")
    parser.add_argument("--corpus-run", type=Path, default=Path("outputs/retrieval/corpus/20261016-142815"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-evidence", action="store_true",
                        help="Strip answer-letter prefixes (e.g. \"Ans. is 'a' i.e.\") from passages in the prompt.")
    parser.add_argument("--state-answer", action="store_true",
                        help="Add a \"Tutor's answer: X. <option>\" line (the predicted answer) to the prompt.")
    parser.add_argument("--no-repeat-ngram-size", type=int, help="Forbid repeating any n-gram of this size.")
    parser.add_argument("--ban-phrase", action="append", default=[], help="Text the reply may not contain.")
    parser.add_argument("--reply-without-adapter", action="store_true",
                        help="Write replies with plain FLAN-T5 (LoRA off); the MCQ answer still uses the adapter.")
    parser.add_argument("--gate-off", action="store_true",
                        help="Answer every follow-up with its retrieved evidence (no confidence gate).")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                        help="Paired comparison of two graded pilots; prints only.")
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(OUTPUT_ROOT / args.compare[0], OUTPUT_ROOT / args.compare[1]), indent=2))
        return
    if args.run_id is None:
        parser.error("--run-id is required unless --compare is given.")
    output_dir = OUTPUT_ROOT / args.run_id
    if args.score:
        print(json.dumps(score(output_dir), indent=2))
        return
    if args.adapter_dir is None:
        parser.error("--adapter-dir is required to run the pilot.")
    records = [json.loads(line) for line in QUERIES_PATH.read_text("utf-8").splitlines() if line.strip()]
    tutor, predict, metadata = _real_components(args)
    print(json.dumps(run(records=records, tutor=tutor, predict=predict, output_dir=output_dir, n=args.n,
                         metadata=metadata), indent=2))


if __name__ == "__main__":
    main()
