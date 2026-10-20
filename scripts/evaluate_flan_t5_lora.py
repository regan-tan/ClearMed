"""Generate and evaluate a saved FLAN-T5 LoRA adapter on internal_dev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_flan_t5_lora import (
    explanation_overlap,
    load_partition,
    positive_int,
    resolve_device,
    subset,
    verify_manifest,
)

from clearmed.data.preprocessing import (
    get_answer_label,
    is_nonblank_text,
    is_valid_qa_example,
)
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.training.flan_t5_lora import FlanT5Predictor, load_adapter_checkpoint


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    complete_train = load_partition(args.train_dir, "part1_train")
    complete_dev = load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(complete_train, complete_dev, train_name="part1_train", development_name="internal_dev")
    manifest_verification = verify_manifest(train=complete_train, development=complete_dev, train_dir=args.train_dir, dev_dir=args.dev_dir, manifest_path=args.manifest)
    dev = subset(complete_dev, args.max_dev_questions, args.seed)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, data_config, metadata = load_adapter_checkpoint(args.checkpoint_dir, resolve_device(args.device))
    rows = [row for row in dev if is_valid_qa_example(row)]
    predict, generated = FlanT5Predictor(model, tokenizer, data_config, resolve_device(args.device)).cached_predictor(rows, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    evaluation = evaluate_mcq_predictions(rows, predict, error_limit=args.error_limit)
    predictions = [
        {**item, "gold_label": get_answer_label(row), "predicted_label": item["predicted_answer"] or "MALFORMED",
         "reference_explanation": row.get("exp").strip() if is_nonblank_text(row.get("exp")) else "", "subject_name": row.get("subject_name")}
        for row, item in zip(rows, generated, strict=True)
    ]
    metrics = {
        "model_id": "flan_t5_lora", "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_metadata": metadata, "evaluated_partition": "internal_dev",
        "selected_dev_questions": len(dev), "selected_dev_fingerprint": membership_fingerprint(dev),
        "manifest_verification": manifest_verification,
        "mcq_metrics": {key: value for key, value in evaluation.items() if key not in {"errors", "subject_accuracy"}},
        "explanation_overlap": explanation_overlap(predictions),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--max-dev-questions", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    metrics = run_evaluation(build_parser().parse_args())
    mcq = metrics["mcq_metrics"]
    print(f"accuracy={mcq['accuracy']:.4f} ({mcq['correct']}/{mcq['total_evaluated']})")


if __name__ == "__main__":
    main()
