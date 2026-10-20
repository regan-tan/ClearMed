"""Fine-tune FLAN-T5 with LoRA on saved ClearMed Part I partitions.

The complete canonical partitions are verified before optional smoke-test
subsetting.  Example smoke run (requires a locally cached/downloadable model):

    uv run python scripts/train_flan_t5_lora.py --max-train-questions 8 \
        --max-dev-questions 4 --batch-size 2 --max-steps 1 --device cpu \
        --run-id flan-smoke-v1
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import peft
import torch
import transformers
from rouge_score import rouge_scorer

from clearmed.data.flan_t5 import FlanT5DataConfig, FlanT5MCQDataset
from clearmed.data.preprocessing import (
    get_answer_label,
    is_nonblank_text,
    is_valid_qa_example,
)
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.models.flan_t5_lora import (
    FlanT5LoraConfig,
    load_flan_t5_lora,
    parameter_counts,
)
from clearmed.training.flan_t5_lora import (
    FlanT5Predictor,
    FlanT5TrainConfig,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
    set_seed,
    train_steps,
)


def _load_tfidf_helpers() -> Any:
    path = Path(__file__).with_name("train_tfidf.py")
    spec = spec_from_file_location("_clearmed_tfidf_helpers_for_flan", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_helpers = _load_tfidf_helpers()
load_partition = _helpers._load_partition
verify_manifest = _helpers._verify_manifest
subset = _helpers._subset
prepare_directory = _helpers._prepare_directory
positive_int = _helpers._positive_int


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    return torch.device(name)


def explanation_overlap(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(item["reference_explanation"], item["generated_explanation"])["rougeL"].fmeasure
        for item in predictions if item["reference_explanation"]
    ]
    return {
        "rows_with_reference": len(scores), "rouge_l_f1_mean": mean(scores) if scores else None,
        "note": "Reference overlap is supplementary; RQ2 also needs factual and human-quality assessment.",
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    complete_train = load_partition(args.train_dir, "part1_train")
    complete_dev = load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(complete_train, complete_dev, train_name="part1_train", development_name="internal_dev")
    manifest_verification = verify_manifest(
        train=complete_train, development=complete_dev, train_dir=args.train_dir,
        dev_dir=args.dev_dir, manifest_path=args.manifest,
    )
    train = subset(complete_train, args.max_train_questions, args.seed)
    dev = subset(complete_dev, args.max_dev_questions, args.seed)
    checkpoint_dir, output_dir = args.checkpoint_root / args.run_id, args.output_root / args.run_id
    prepare_directory(checkpoint_dir, overwrite=args.overwrite)
    prepare_directory(output_dir, overwrite=args.overwrite)

    set_seed(args.seed)
    device = resolve_device(args.device)
    lora_config = FlanT5LoraConfig(
        model_name=args.model_name, rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout, target_modules=tuple(args.lora_target_modules),
    )
    data_config = FlanT5DataConfig(args.max_source_length, args.max_target_length)
    train_config = FlanT5TrainConfig(
        seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, gradient_clip=args.gradient_clip, max_steps=args.max_steps,
    )
    model, tokenizer = load_flan_t5_lora(lora_config)
    model.to(device)
    train_dataset = FlanT5MCQDataset(list(train), tokenizer, data_config)
    dev_rows = [row for row in dev if is_valid_qa_example(row)]
    if not train_dataset.examples:
        raise ValueError("No QA-valid training rows remain after validation.")

    start = perf_counter()
    losses = train_steps(model, train_dataset.examples, tokenizer, data_config, train_config, device)
    training_seconds = perf_counter() - start
    data_identity = {
        "train_partition": "part1_train", "dev_partition": "internal_dev",
        "selected_train_fingerprint": membership_fingerprint(train),
        "selected_dev_fingerprint": membership_fingerprint(dev),
    }
    save_adapter_checkpoint(
        checkpoint_dir, model=model, tokenizer=tokenizer, model_name=args.model_name,
        data_config=data_config, train_config=train_config,
        extra={"run_id": args.run_id, "lora_config": lora_config.to_dict(), "data_identity": data_identity},
    )

    predictor = FlanT5Predictor(model, tokenizer, data_config, device)
    predict, generated = predictor.cached_predictor(dev_rows, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    reloaded_model, reloaded_tokenizer, reloaded_data_config, _ = load_adapter_checkpoint(checkpoint_dir, device)
    reloaded_generated = FlanT5Predictor(
        reloaded_model, reloaded_tokenizer, reloaded_data_config, device
    ).generate_batch(dev_rows, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    reload_verified = [item["raw_output"] for item in generated] == [
        item["raw_output"] for item in reloaded_generated
    ]
    if not reload_verified:
        raise RuntimeError("Saved and reloaded LoRA adapter produced different greedy outputs.")
    evaluation = evaluate_mcq_predictions(dev_rows, predict, error_limit=args.error_limit)
    predictions = []
    for row, item in zip(dev_rows, generated, strict=True):
        reference = row.get("exp").strip() if is_nonblank_text(row.get("exp")) else ""
        predictions.append({
            **item, "gold_label": get_answer_label(row),
            "predicted_label": item["predicted_answer"] or "MALFORMED",
            "reference_explanation": reference, "subject_name": row.get("subject_name"),
        })
    metrics = {
        "model_id": "flan_t5_lora", "model_name": args.model_name, "run_id": args.run_id,
        "created_utc": datetime.now(UTC).isoformat(), "device": str(device),
        "lora_config": lora_config.to_dict(), "parameter_counts": parameter_counts(model),
        "data_config": vars(data_config), "train_config": vars(train_config), "data_identity": data_identity,
        "manifest_verification": manifest_verification, "train_data_statistics": train_dataset.statistics(),
        "selected_dev_questions": len(dev), "training_steps": len(losses),
        "training_loss_mean": mean(losses), "training_seconds": training_seconds,
        "saved_adapter_reload_verified": reload_verified,
        "decoding": {"strategy": "greedy", "do_sample": False, "max_new_tokens": args.max_new_tokens},
        "mcq_metrics": {key: value for key, value in evaluation.items() if key not in {"errors", "subject_accuracy"}},
        "explanation_overlap": explanation_overlap(predictions),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    (output_dir / "config.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "training_history.json").write_text(json.dumps({"losses": losses}, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--max-train-questions", type=positive_int)
    parser.add_argument("--max-dev-questions", type=positive_int)
    parser.add_argument("--model-name", default="google/flan-t5-small")
    parser.add_argument("--max-source-length", type=positive_int, default=512)
    parser.add_argument("--max-target-length", type=positive_int, default=192)
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q", "v"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=positive_int, default=1000)
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/flan_t5_lora"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/flan_t5_lora"))
    parser.add_argument("--error-limit", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    metrics = run_training(build_parser().parse_args())
    result = metrics["mcq_metrics"]
    print(f"accuracy={result['accuracy']:.4f} ({result['correct']}/{result['total_evaluated']})")


if __name__ == "__main__":
    main()
