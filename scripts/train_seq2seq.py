"""Train the scratch Seq2Seq (bi-GRU + additive attention) MCQ model.

Reads only the saved Step 1 partitions. Canonical partitions are verified
against ``data/splits/manifest.json`` *before* any --max-* subsetting, using the
same verification code as the TF-IDF baseline.

Local smoke test:
    uv run python scripts/train_seq2seq.py --max-train-questions 500 \
        --max-dev-questions 100 --max-epochs 3 --run-id smoke

Full run (GPU cluster):
    uv run python scripts/train_seq2seq.py --device cuda --run-id full-seed42
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from clearmed.data.preprocessing import is_valid_qa_example
from clearmed.data.seq2seq import Seq2SeqDataConfig, Seq2SeqMCQDataset, build_vocabulary
from clearmed.data.splitting import assert_train_dev_disjoint, membership_fingerprint
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.models.seq2seq import Seq2SeqModel, Seq2SeqModelConfig, count_parameters
from clearmed.training.seq2seq import (
    Seq2SeqPredictor,
    Seq2SeqTrainConfig,
    fit,
    set_seed,
)


def _load_tfidf_script_helpers():
    """Reuse partition loading + manifest verification from the TF-IDF runner."""
    path = Path(__file__).with_name("train_tfidf.py")
    spec = spec_from_file_location("_clearmed_train_tfidf_helpers", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


_helpers = _load_tfidf_script_helpers()
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


def accuracy_on_rows(
    predictor: Seq2SeqPredictor, rows: Sequence[Mapping[str, Any]], batch_size: int = 64
) -> dict[str, Any]:
    """Dev accuracy through the shared question-level evaluator."""
    predict, _ = predictor.cached_predictor(rows, batch_size=batch_size)
    return evaluate_mcq_predictions(rows, predict, error_limit=0)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    complete_train = load_partition(args.train_dir, "part1_train")
    complete_dev = load_partition(args.dev_dir, "internal_dev")
    assert_train_dev_disjoint(
        complete_train, complete_dev, train_name="part1_train", development_name="internal_dev"
    )
    manifest_verification = verify_manifest(
        train=complete_train,
        development=complete_dev,
        train_dir=args.train_dir,
        dev_dir=args.dev_dir,
        manifest_path=args.manifest,
    )
    train = subset(complete_train, args.max_train_questions, args.seed)
    dev = subset(complete_dev, args.max_dev_questions, args.seed)

    checkpoint_dir = args.checkpoint_root / args.run_id
    output_dir = args.output_root / args.run_id
    if not args.resume:
        prepare_directory(checkpoint_dir, overwrite=args.overwrite)
        prepare_directory(output_dir, overwrite=args.overwrite)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)
    data_config = Seq2SeqDataConfig(
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_question_tokens=args.max_question_tokens,
        max_option_tokens=args.max_option_tokens,
        max_explanation_tokens=args.max_explanation_tokens,
        include_explanations=not args.answer_only,
    )
    train_rows = list(train)
    dev_rows = list(dev)
    # Vocabulary is fit on the (possibly subsetted) training partition ONLY.
    vocab = build_vocabulary(train_rows, data_config)
    train_dataset = Seq2SeqMCQDataset(train_rows, vocab, data_config)
    dev_dataset = Seq2SeqMCQDataset(dev_rows, vocab, data_config)
    valid_dev_rows = [row for row in dev_rows if is_valid_qa_example(row)]

    model_config = Seq2SeqModelConfig(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    train_config = Seq2SeqTrainConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        gradient_clip=args.gradient_clip,
        teacher_forcing_ratio=args.teacher_forcing_ratio,
        answer_loss_weight=args.answer_loss_weight,
        early_stopping_patience=args.patience,
        log_every=args.log_every,
    )
    model = Seq2SeqModel(model_config)
    data_identity = {
        "train_partition": "part1_train",
        "dev_partition": "internal_dev",
        "selected_train_fingerprint": membership_fingerprint(train),
        "selected_dev_fingerprint": membership_fingerprint(dev),
    }
    run_config = {
        "model_id": "seq2seq",
        "run_id": args.run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "cli_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_config": asdict(data_config),
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "vocabulary_size": len(vocab),
        "trainable_parameters": count_parameters(model),
        "pretrained_weights": None,
        "device": str(device),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "manifest_verification": manifest_verification,
        "data_identity": data_identity,
        "train_data_statistics": train_dataset.statistics(vocab),
        "dev_data_statistics": dev_dataset.statistics(vocab),
        "selection_criterion": "highest internal_dev accuracy, tie-break lowest dev token loss",
    }
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Training seq2seq on {len(train_dataset):,} MCQs, selecting on {len(dev_dataset):,} dev MCQs; "
        f"vocab={len(vocab):,}, params={run_config['trainable_parameters']:,}, device={device}"
    )

    start = perf_counter()
    result = fit(
        model=model,
        vocab=vocab,
        data_config=data_config,
        train_config=train_config,
        train_examples=train_dataset.examples,
        dev_examples=dev_dataset.examples,
        dev_rows=valid_dev_rows,
        checkpoint_dir=checkpoint_dir,
        device=device,
        evaluate_accuracy=lambda predictor, rows: accuracy_on_rows(predictor, rows, args.batch_size),
        resume=args.resume,
        checkpoint_extra={"run_id": args.run_id, "data_identity": data_identity},
    )
    elapsed = perf_counter() - start
    (output_dir / "training_history.json").write_text(
        json.dumps(result["history"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "best": result["best"],
        "stopped_early": result["stopped_early"],
        "epochs_run": len(result["history"]),
        "training_seconds_this_invocation": elapsed,
        "best_checkpoint": str(checkpoint_dir / "best"),
        "last_checkpoint": str(checkpoint_dir / "last"),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**run_config, "training_summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    data = parser.add_argument_group("data")
    data.add_argument("--train-dir", type=Path, default=Path("data/splits/part1_train"))
    data.add_argument("--dev-dir", type=Path, default=Path("data/splits/internal_dev"))
    data.add_argument("--manifest", type=Path, default=Path("data/splits/manifest.json"))
    data.add_argument("--max-train-questions", type=positive_int)
    data.add_argument("--max-dev-questions", type=positive_int)
    data.add_argument("--min-freq", type=positive_int, default=2)
    data.add_argument("--max-vocab-size", type=positive_int, default=30_000)
    data.add_argument("--max-question-tokens", type=positive_int, default=96)
    data.add_argument("--max-option-tokens", type=positive_int, default=24)
    data.add_argument("--max-explanation-tokens", type=positive_int, default=128)
    data.add_argument("--answer-only", action="store_true", help="Ablation: train on answer labels only.")

    model = parser.add_argument_group("model")
    model.add_argument("--embedding-dim", type=positive_int, default=256)
    model.add_argument("--hidden-dim", type=positive_int, default=256)
    model.add_argument("--attention-dim", type=positive_int, default=256)
    model.add_argument("--num-layers", type=positive_int, default=1)
    model.add_argument("--dropout", type=float, default=0.2)

    train = parser.add_argument_group("training")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--batch-size", type=positive_int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--max-epochs", type=positive_int, default=20)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--teacher-forcing-ratio", type=float, default=1.0)
    train.add_argument("--answer-loss-weight", type=float, default=1.0)
    train.add_argument("--patience", type=positive_int, default=3)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    train.add_argument(
        "--log-every", type=int, default=200, help="Progress line every N batches (0 = per epoch only)."
    )

    out = parser.add_argument_group("artifacts")
    out.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    out.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/seq2seq"))
    out.add_argument("--output-root", type=Path, default=Path("outputs/seq2seq"))
    out.add_argument("--resume", action="store_true", help="Continue this --run-id from its 'last' checkpoint.")
    out.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_training(args)
    best = result["training_summary"]["best"]
    print(f"Best epoch {best['epoch']}: dev_accuracy={best['dev_accuracy']:.4f}" if best else "No epochs run.")
    print(f"Checkpoints: {result['training_summary']['best_checkpoint']}")


if __name__ == "__main__":
    main()