"""Run a reproducible, resumable FLAN-T5 + LoRA Part I training experiment.

The default is a real training budget (three epochs, no sample cap), not the
old diagnostic cap.  Use ``--max-*-questions`` and ``--max-steps`` only for a
bounded smoke run.  Full generated evaluation is intentionally a separate
``evaluate_flan_t5_lora.py`` stage.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter
from typing import Any

import peft
import torch
import transformers

from clearmed.data.flan_t5 import FlanT5DataConfig, FlanT5MCQDataset
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.models.flan_t5_lora import (
    FlanT5LoraConfig,
    load_flan_t5_lora,
    load_saved_adapter,
    parameter_counts,
)
from clearmed.training.flan_t5_lora import (
    FlanT5TrainConfig,
    run_resumable_training,
    set_seed,
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
positive_int = _helpers._positive_int


def resolve_device(name: str, *, require_cuda: bool = False) -> torch.device:
    if name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(name)
    if (name.startswith("cuda") or require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run but is not available; refusing a silent CPU fallback.")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda requires --device cuda or --device auto with CUDA available.")
    return device


def _claim_fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {path}. Choose a new --run-id or use --resume.")
    path.mkdir(parents=True, exist_ok=True)


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _validation_subset(dev: Any, limit: int | None, seed: int) -> Any:
    if limit is None or limit >= len(dev):
        return dev
    return subset(dev, limit, seed)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Train only from canonical Part I data and save a resumable experiment."""
    complete_train = load_partition(args.train_dir, "part1_train")
    complete_dev = load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(complete_train, complete_dev, train_name="part1_train", development_name="internal_dev")
    manifest_verification = verify_manifest(train=complete_train, development=complete_dev, train_dir=args.train_dir, dev_dir=args.dev_dir, manifest_path=args.manifest)
    train = subset(complete_train, args.max_train_questions, args.seed)
    dev = subset(complete_dev, args.max_dev_questions, args.seed)
    validation = _validation_subset(dev, args.validation_questions, args.seed)
    checkpoint_dir, output_dir = args.checkpoint_root / args.run_id, args.output_root / args.run_id
    if args.resume:
        if not checkpoint_dir.is_dir() or not output_dir.is_dir():
            raise FileNotFoundError("--resume requires existing checkpoint and output run directories.")
    else:
        _claim_fresh_directory(checkpoint_dir)
        _claim_fresh_directory(output_dir)

    device = resolve_device(args.device, require_cuda=args.require_cuda)
    set_seed(args.seed)
    lora_config = FlanT5LoraConfig(model_name=args.model_name, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout, target_modules=tuple(args.lora_target_modules))
    data_config = FlanT5DataConfig(args.max_source_length, args.max_target_length)
    train_config = FlanT5TrainConfig(seed=args.seed, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, gradient_clip=args.gradient_clip, max_steps=args.max_steps, epochs=args.epochs, gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_steps=args.warmup_steps, precision=args.precision)
    latest_adapter = checkpoint_dir / "latest" / "adapter"
    if args.resume:
        model, tokenizer = load_saved_adapter(str(latest_adapter), args.model_name, is_trainable=True)
    else:
        model, tokenizer = load_flan_t5_lora(lora_config)
    model.to(device)
    train_dataset = FlanT5MCQDataset(list(train), tokenizer, data_config)
    validation_dataset = FlanT5MCQDataset(list(validation), tokenizer, data_config)
    if not train_dataset.examples or not validation_dataset.examples:
        raise ValueError("No QA-valid training or validation rows remain after validation.")
    data_identity = {
        "train_partition": "part1_train", "development_partition": "internal_dev",
        "complete_train_fingerprint": membership_fingerprint(complete_train), "complete_dev_fingerprint": membership_fingerprint(complete_dev),
        "selected_train_fingerprint": membership_fingerprint(train), "selected_development_fingerprint": membership_fingerprint(dev),
        "validation_subset_fingerprint": membership_fingerprint(validation), "validation_subset_selection": {"method": "deterministic_shuffle_then_prefix", "seed": args.seed, "requested_size": args.validation_questions, "actual_rows": len(validation)},
    }
    metadata = {
        "model_id": "flan_t5_lora", "run_id": args.run_id, "created_utc": datetime.now(UTC).isoformat(), "requested_arguments": _serializable_args(args),
        "lora_config": lora_config.to_dict(), "data_config": asdict(data_config), "train_config": asdict(train_config), "data_identity": data_identity,
        "manifest_verification": manifest_verification, "train_data_statistics": train_dataset.statistics(), "validation_data_statistics": validation_dataset.statistics(),
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__, "device": str(device), "cuda_available": torch.cuda.is_available(), "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "checkpoint_selection": {"metric": "minimum_teacher_forced_validation_loss", "partition": "deterministic internal_dev subset", "generation_during_training": False},
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    start = perf_counter()
    training = run_resumable_training(model=model, tokenizer=tokenizer, train_examples=train_dataset.examples, validation_examples=validation_dataset.examples, data_config=data_config, train_config=train_config, device=device, run_checkpoint_dir=checkpoint_dir, model_name=args.model_name, lora_config=lora_config.to_dict(), data_identity=data_identity, validation_interval_steps=args.validation_interval_steps, checkpoint_interval_steps=args.checkpoint_interval_steps, early_stopping_patience=args.early_stopping_patience, resume=args.resume)
    elapsed = perf_counter() - start
    history = training["history"]
    with (output_dir / "training_history.jsonl").open("w", encoding="utf-8") as handle:
        for event in history:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    validation_events = [event for event in history if event["event"] == "validation"]
    result = {**metadata, "parameter_counts": parameter_counts(model), "training": {key: value for key, value in training.items() if key != "history"}, "training_duration_seconds_this_invocation": elapsed, "validation_events": validation_events, "best_adapter_dir": str(checkpoint_dir / "best"), "latest_training_state_dir": str(checkpoint_dir / "latest")}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    parser.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    parser.add_argument("--max-train-questions", type=positive_int, help="Diagnostic-only deterministic cap; omit for all eligible part1_train rows.")
    parser.add_argument("--max-dev-questions", type=positive_int, help="Diagnostic-only deterministic cap; omit for all internal_dev rows.")
    parser.add_argument("--validation-questions", type=positive_int, default=512, help="Deterministic internal_dev subset used for teacher-forced checkpoint selection.")
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--max-source-length", type=positive_int, default=512)
    parser.add_argument("--max-target-length", type=positive_int, default=192)
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=["q", "v"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--max-steps", type=positive_int, help="Optional optimizer-step ceiling; omit for the epoch budget.")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--validation-interval-steps", type=positive_int, default=500)
    parser.add_argument("--checkpoint-interval-steps", type=positive_int, default=250)
    parser.add_argument("--early-stopping-patience", type=positive_int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/flan_t5_lora"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/flan_t5_lora"))
    parser.add_argument("--resume", action="store_true", help="Continue exactly the named run from checkpoints/.../<run-id>/latest.")
    return parser


def main() -> None:
    metrics = run_training(build_parser().parse_args())
    progress = metrics["training"]["progress"]
    print(f"complete global_step={progress['global_step']} best_validation_loss={progress['best_validation_loss']}")


if __name__ == "__main__":
    main()
