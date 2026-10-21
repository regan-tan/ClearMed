"""Independently evaluate a selected FLAN-T5 LoRA adapter on ``internal_dev``."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from train_flan_t5_lora import (
    load_partition,
    positive_int,
    resolve_device,
    subset,
    verify_manifest,
)

from clearmed.data.preprocessing import is_valid_qa_example
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.flan_t5 import (
    explanation_metrics,
    generation_parse_distribution,
    prediction_records,
)
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.training.flan_t5_lora import FlanT5Predictor, load_adapter_checkpoint


def _claim_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    complete_train = load_partition(args.train_dir, "part1_train")
    complete_dev = load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(complete_train, complete_dev, train_name="part1_train", development_name="internal_dev")
    manifest_verification = verify_manifest(train=complete_train, development=complete_dev, train_dir=args.train_dir, dev_dir=args.dev_dir, manifest_path=args.manifest)
    dev = subset(complete_dev, args.max_dev_questions, args.seed)
    _claim_output(args.output_dir)
    device = resolve_device(args.device, require_cuda=args.require_cuda)
    model, tokenizer, data_config, metadata = load_adapter_checkpoint(args.checkpoint_dir, device)
    rows = [row for row in dev if is_valid_qa_example(row)]
    start = perf_counter()
    predictor = FlanT5Predictor(model, tokenizer, data_config, device)
    predict, generated = predictor.cached_predictor(rows, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    generation_seconds = perf_counter() - start
    mcq = evaluate_mcq_predictions(rows, predict, error_limit=args.error_limit)
    records = prediction_records(rows, generated)
    explanation = explanation_metrics(records, compute_bertscore=args.bertscore, bertscore_device=str(device) if args.bertscore else None)
    metrics = {
        "model_id": "flan_t5_lora", "created_utc": datetime.now(UTC).isoformat(), "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_metadata": metadata, "evaluated_partition": "internal_dev", "complete_dev_questions": len(complete_dev),
        "selected_dev_questions": len(dev), "qa_valid_evaluated_questions": len(rows), "selected_dev_fingerprint": membership_fingerprint(dev),
        "manifest_verification": manifest_verification, "device": str(device), "generation_seconds": generation_seconds,
        "generation": {"strategy": "greedy", "do_sample": False, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size},
        "mcq_metrics": {key: value for key, value in mcq.items() if key != "errors"},
        "generation_parse_distribution": generation_parse_distribution(records), "explanation_metrics": explanation,
    }
    (args.output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "explanation_metrics.json").write_text(json.dumps(explanation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "errors.jsonl").open("w", encoding="utf-8") as handle:
        for item in mcq["errors"]:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Selected inference adapter, normally checkpoints/flan_t5_lora/<run-id>/best")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--max-dev-questions", type=positive_int, help="Preliminary diagnostic cap; omit for complete internal_dev evaluation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--error-limit", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--bertscore", action="store_true", help="Compute costly BERTScore for every explanation-eligible selected row.")
    return parser


def main() -> None:
    metrics = run_evaluation(build_parser().parse_args())
    mcq = metrics["mcq_metrics"]
    print(f"accuracy={mcq['accuracy']:.4f} ({mcq['correct']}/{mcq['total_evaluated']})")


if __name__ == "__main__":
    main()
