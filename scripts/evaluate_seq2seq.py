"""Evaluate a saved Seq2Seq checkpoint on internal_dev without retraining.

Generation sees only the question and options. Gold labels and reference
explanations are read *after* generation, purely for scoring and export.

    uv run python scripts/evaluate_seq2seq.py \
        --checkpoint-dir checkpoints/seq2seq/<run_id>/best \
        --output-dir outputs/seq2seq/<run_id>/eval-internal-dev
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import torch
from rouge_score import rouge_scorer

from clearmed.data.preprocessing import (
    get_answer_label,
    is_nonblank_text,
    is_valid_qa_example,
)
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.training.seq2seq import Seq2SeqPredictor, load_checkpoint


def _load_train_script():
    path = Path(__file__).with_name("train_seq2seq.py")
    spec = spec_from_file_location("_clearmed_train_seq2seq", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_train = _load_train_script()


def explanation_overlap(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """ROUGE-L F1 on rows that have a reference explanation.

    Reference overlap is a supplementary signal only; it does not measure
    medical correctness (see AGENTS.md section 5).
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(p["reference_explanation"], p["generated_explanation"])["rougeL"].fmeasure
        for p in predictions
        if p["reference_explanation"]
    ]
    return {
        "rows_with_reference": len(scores),
        "rouge_l_f1_mean": mean(scores) if scores else None,
        "empty_generated_explanation_rate": (
            sum(not p["generated_explanation"] for p in predictions) / len(predictions) if predictions else 0.0
        ),
        "note": "Reference overlap only; BERTScore and blind human rating are run separately for RQ2.",
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    complete_train = _train.load_partition(args.train_dir, "part1_train")
    complete_dev = _train.load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(complete_train, complete_dev, train_name="part1_train", development_name="internal_dev")
    manifest_verification = _train.verify_manifest(
        train=complete_train, development=complete_dev,
        train_dir=args.train_dir, dev_dir=args.dev_dir, manifest_path=args.manifest,
    )
    dev = _train.subset(complete_dev, args.max_dev_questions, args.seed)
    _train.prepare_directory(args.output_dir, overwrite=args.overwrite)

    device = _train.resolve_device(args.device)
    model, vocab, data_config, metadata = load_checkpoint(args.checkpoint_dir, device)
    predictor = Seq2SeqPredictor(model, vocab, data_config, device)
    rows = [row for row in dev if is_valid_qa_example(row)]

    start = perf_counter()
    predict, raw_predictions = predictor.cached_predictor(rows, batch_size=args.batch_size)
    generation_seconds = perf_counter() - start
    evaluation = evaluate_mcq_predictions(rows, predict, error_limit=args.error_limit)

    # Gold/reference fields are attached only now, after generation, for export.
    predictions = []
    for row, generated in zip(rows, raw_predictions, strict=True):
        reference = row.get("exp").strip() if is_nonblank_text(row.get("exp")) else ""
        predictions.append(
            {
                "example_id": generated["example_id"],
                "gold_label": get_answer_label(row),
                "predicted_label": generated["predicted_answer"] or "MALFORMED",
                "raw_output": generated["raw_output"],
                "generated_explanation": generated["generated_explanation"],
                "answer_parse_mode": generated["answer_parse_mode"],
                "answer_explanation_format_compliant": generated[
                    "answer_explanation_format_compliant"
                ],
                "reference_explanation": reference,
                "subject_name": row.get("subject_name"),
            }
        )

    metrics = {
        "model_id": "seq2seq",
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_extra": metadata.get("extra", {}),
        "evaluated_partition": "internal_dev",
        "selected_dev_questions": len(dev),
        "selected_dev_fingerprint": membership_fingerprint(dev),
        "manifest_verification": manifest_verification,
        "device": str(device),
        "decoding": {"strategy": "greedy", "max_target_len": data_config.max_target_len},
        "generation_seconds": generation_seconds,
        "mcq_metrics": {k: v for k, v in evaluation.items() if k not in {"errors", "subject_accuracy"}},
        "explanation_overlap": explanation_overlap(predictions),
    }
    out = args.output_dir
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for error in evaluation["errors"]:
            handle.write(json.dumps(error, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "subject_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_name", "total", "correct", "accuracy"])
        writer.writeheader()
        writer.writerows(evaluation["subject_accuracy"])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--max-dev-questions", type=_train.positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=_train.positive_int, default=64)
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_grad_enabled(False)
    metrics = run_evaluation(args)
    mcq = metrics["mcq_metrics"]
    print(
        f"accuracy={mcq['accuracy']:.4f} ({mcq['correct']}/{mcq['total_evaluated']}), "
        f"malformed_rate={mcq['malformed_output_rate']:.4f}, "
        f"rougeL={metrics['explanation_overlap']['rouge_l_f1_mean']}"
    )
    print(f"Saved evaluation to {args.output_dir}")


if __name__ == "__main__":
    main()
