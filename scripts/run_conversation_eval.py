"""Run the RQ4 queries through the four tutor systems (Plan 7).

The 30 RQ4 evaluation queries (and their 10 second turns) go through no
retrieval, MiniLM, E5 and BioLORD on identical inputs: the same frozen
FLAN-T5 + LoRA checkpoint, prompt template and ``Answer:`` pre-fill, and that
checkpoint's own predicted answer. Each retrieval system uses its frozen gate.
A question whose predicted answer is malformed is skipped by every system and
counted.

Writes ``outputs/conversation/eval/<run_id>/``: ``mcq_predictions.jsonl``,
``<system>/predictions.jsonl`` (passage IDs, scores, gate decision, prompt
and reply per turn) and ``config.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from clearmed.conversation.pipeline import NO_RETRIEVAL, TutorPipeline, mcq_row
from clearmed.retrieval.embedders import METHOD_IDS

QUERIES_PATH = Path("data/retrieval_eval/queries_eval.jsonl")
OUTPUT_ROOT = Path("outputs/conversation/eval")
SYSTEMS = (NO_RETRIEVAL, *METHOD_IDS)


def run(
    *,
    records: Sequence[Mapping[str, Any]],
    tutors: Mapping[str, TutorPipeline],
    predict: Callable[[list[dict[str, Any]]], list[str | None]],
    output_dir: Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty; choose a new --run-id.")
    if set(tutors) != set(SYSTEMS):
        raise ValueError(f"Expected tutors for {SYSTEMS}, got {sorted(tutors)}.")
    if any(tutor.stage != "rq4" for tutor in tutors.values()):
        raise ValueError("RQ4 tutors must run at stage 'rq4' (predicted answers only).")
    rq4 = [record for record in records if record.get("rq4")]
    if not rq4:
        raise ValueError("No RQ4 queries in the query file.")

    labels = predict([{"id": str(r["question_id"]), **mcq_row(r)} for r in rq4])
    usable = [(record, label) for record, label in zip(rq4, labels, strict=True) if label is not None]
    turns_by_system = {
        system: [
            {**turn, "system": system}
            for record, label in usable
            for turn in tutors[system].converse(record, predicted_label=label)
        ]
        for system in SYSTEMS
    }

    output_dir.mkdir(parents=True)
    with (output_dir / "mcq_predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record, label in zip(rq4, labels, strict=True):
            handle.write(json.dumps({
                "query_id": record["query_id"], "question_id": record["question_id"],
                "predicted_label": label or "MALFORMED", "gold_label": record["gold_label"],
            }, sort_keys=True) + "\n")
    for system, turns in turns_by_system.items():
        (output_dir / system).mkdir()
        with (output_dir / system / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for turn in turns:
                handle.write(json.dumps(turn, sort_keys=True) + "\n")
    summary = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": "rq4",
        "answer_source": "predicted",
        "n_rq4_queries": len(rq4),
        "n_second_turns": sum(bool(r.get("second_turn")) for r in rq4),
        "malformed_predictions": len(rq4) - len(usable),
        "predicted_correct": sum(label == r["gold_label"] for r, label in usable),
        "systems": {
            system: {
                "threshold": tutors[system].gate.threshold if tutors[system].gate else None,
                "turns": len(turns),
                "fallbacks": sum(t["fallback"] for t in turns),
            }
            for system, turns in turns_by_system.items()
        },
        **dict(metadata or {}),
    }
    (output_dir / "config.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8")
    return summary


def load_prediction_file(path: Path) -> dict[str, str | None]:
    """``{"question_id", "predicted_label"}`` lines; anything outside A-D is malformed."""
    predictions: dict[str, str | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            label = record.get("predicted_label")
            predictions[str(record["question_id"])] = label if label in ("A", "B", "C", "D") else None
    return predictions


def _real_components(args: argparse.Namespace) -> tuple[dict[str, TutorPipeline], Callable, dict[str, Any]]:
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
    generator = FlanT5Generator(model, tokenizer, max_input_tokens=data_config.max_source_length, device=device)
    passages, _ = load_corpus(args.corpus_run)
    tutors = {NO_RETRIEVAL: TutorPipeline(generator, stage="rq4")}
    for method_id in METHOD_IDS:
        config = load_embedding_config(method_id)
        retriever = DenseRetriever(
            Embedder(config, SentenceTransformerEncoder(config)),
            DenseIndex.load(Path("outputs/retrieval") / method_id / args.index_run),
            passages,
        )
        tutors[method_id] = TutorPipeline(generator, stage="rq4", retriever=retriever,
                                          gate=Gate.load(method_id), method_id=method_id)

    if args.predictions is not None:
        from_file = load_prediction_file(args.predictions)

        def predict(rows: list[dict[str, Any]]) -> list[str | None]:
            missing = [row["id"] for row in rows if row["id"] not in from_file]
            if missing:
                raise ValueError(f"{args.predictions} has no prediction for {missing[:3]}.")
            return [from_file[row["id"]] for row in rows]
    else:
        predictor = FlanT5Predictor(model, tokenizer, data_config, device)

        def predict(rows: list[dict[str, Any]]) -> list[str | None]:
            return [p["predicted_answer"] for p in predictor.generate_batch(rows)]

    metadata = {
        "adapter_dir": Path(args.adapter_dir).as_posix(),
        "model_name": adapter_metadata["model_name"],
        "max_source_length": data_config.max_source_length,
        "max_new_tokens": generator.max_new_tokens,
        "predictions_file": args.predictions.as_posix() if args.predictions else None,
        "index_run": args.index_run,
        "corpus_run": Path(args.corpus_run).as_posix(),
        "queries": QUERIES_PATH.as_posix(),
        "queries_sha256": sha256(QUERIES_PATH.read_bytes()).hexdigest(),
        "dry_run": args.dry_run,
    }
    return tutors, predict, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True, help="Frozen FLAN-T5 + LoRA adapter directory.")
    parser.add_argument("--predictions", type=Path, help="Optional MCQ predictions from that same checkpoint.")
    parser.add_argument("--index-run", default="full-20261016")
    parser.add_argument("--corpus-run", type=Path, default=Path("outputs/retrieval/corpus/20261016-142815"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mark the run as a pipeline check (e.g. with a smoke checkpoint), not an RQ4 result.")
    args = parser.parse_args()
    records = [json.loads(line) for line in QUERIES_PATH.read_text("utf-8").splitlines() if line.strip()]
    tutors, predict, metadata = _real_components(args)
    summary = run(records=records, tutors=tutors, predict=predict, output_dir=OUTPUT_ROOT / args.run_id,
                  metadata=metadata)
    print(json.dumps({k: v for k, v in summary.items() if k != "systems"} | {"systems": summary["systems"]}, indent=2))


if __name__ == "__main__":
    main()
