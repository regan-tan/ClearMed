"""Training, checkpointing, and prediction for the scratch Seq2Seq model."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from clearmed.data.seq2seq import (
    EncodedExample,
    Seq2SeqBatch,
    Seq2SeqDataConfig,
    Vocabulary,
    collate_examples,
    encode_example,
    parse_answer_label,
    parse_explanation,
    render_output,
)
from clearmed.models.seq2seq import Seq2SeqModel, Seq2SeqModelConfig

CHECKPOINT_FILENAME = "model.pt"
METADATA_FILENAME = "checkpoint_config.json"


@dataclass(frozen=True)
class Seq2SeqTrainConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 20
    gradient_clip: float = 1.0
    teacher_forcing_ratio: float = 1.0
    # Loss weight on the first target position (the answer label) relative to
    # explanation tokens. 1.0 = plain token-level cross-entropy.
    answer_loss_weight: float = 1.0
    early_stopping_patience: int = 3
    # Print a progress line every N training batches (0 = only per epoch).
    log_every: int = 200


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(
    examples: Sequence[EncodedExample], vocab: Vocabulary, *, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        list(examples),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=partial(collate_examples, vocab=vocab),
    )


def sequence_loss(
    logits: torch.Tensor, target: torch.Tensor, *, pad_id: int, answer_loss_weight: float = 1.0
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted mean token cross-entropy; pads ignored, position 0 = answer label."""
    per_token = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=pad_id, reduction="none"
    ).view_as(target)
    mask = (target != pad_id).float()
    weights = mask.clone()
    weights[:, 0] = weights[:, 0] * answer_loss_weight
    loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
    answer_loss = (per_token[:, 0] * mask[:, 0]).sum() / mask[:, 0].sum().clamp_min(1.0)
    return loss, {"answer_token_loss": float(answer_loss.detach())}


def train_epoch(
    model: Seq2SeqModel,
    loader: Iterable[Seq2SeqBatch],
    optimizer: torch.optim.Optimizer,
    config: Seq2SeqTrainConfig,
    device: torch.device,
    *,
    log: Callable[[str], None] | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = total_answer_loss = 0.0
    batches = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None  # type: ignore[arg-type]
    for batch in loader:
        batch = batch.to(device)
        assert batch.decoder_input is not None and batch.decoder_target is not None
        optimizer.zero_grad()
        logits = model(
            batch.source,
            batch.source_lengths,
            batch.decoder_input,
            teacher_forcing_ratio=config.teacher_forcing_ratio,
        )
        loss, parts = sequence_loss(
            logits,
            batch.decoder_target,
            pad_id=model.config.pad_id,
            answer_loss_weight=config.answer_loss_weight,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total_loss += float(loss.detach())
        total_answer_loss += parts["answer_token_loss"]
        batches += 1
        if log is not None and config.log_every and batches % config.log_every == 0:
            of_total = f"/{total_batches}" if total_batches else ""
            prefix = f"epoch {epoch} " if epoch is not None else ""
            log(
                f"  {prefix}batch {batches}{of_total}: "
                f"loss={total_loss / batches:.4f} answer_loss={total_answer_loss / batches:.4f}"
            )
    return {
        "train_loss": total_loss / max(batches, 1),
        "train_answer_token_loss": total_answer_loss / max(batches, 1),
    }


@torch.no_grad()
def evaluate_loss(
    model: Seq2SeqModel, loader: Iterable[Seq2SeqBatch], device: torch.device
) -> float:
    """Teacher-forced token loss (unweighted) — a cheap dev signal."""
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.source, batch.source_lengths, batch.decoder_input)
        loss, _ = sequence_loss(logits, batch.decoder_target, pad_id=model.config.pad_id)
        total += float(loss)
        batches += 1
    return total / max(batches, 1)


class Seq2SeqPredictor:
    """Inference wrapper. Encodes **source only** — gold fields are never read."""

    def __init__(
        self,
        model: Seq2SeqModel,
        vocab: Vocabulary,
        data_config: Seq2SeqDataConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.vocab = vocab
        self.data_config = data_config
        self.device = torch.device(device)

    def generate_batch(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        examples = [encode_example(row, self.vocab, self.data_config, with_target=False) for row in rows]
        batch = collate_examples(examples, self.vocab).to(self.device)
        generated = self.model.generate(
            batch.source,
            batch.source_lengths,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            max_len=self.data_config.max_target_len,
        )
        results = []
        for example, ids in zip(examples, generated.tolist(), strict=True):
            raw_output = render_output(ids, self.vocab)
            results.append(
                {
                    "example_id": example.example_id,
                    "predicted_answer": parse_answer_label(raw_output),
                    "raw_output": raw_output,
                    "generated_explanation": parse_explanation(raw_output),
                }
            )
        return results

    def predict_mcq(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Adapter for ``clearmed.evaluation.qa.evaluate_mcq_predictions``."""
        return self.generate_batch([row])[0]

    def cached_predictor(
        self, rows: Sequence[Mapping[str, Any]], batch_size: int = 64
    ) -> tuple[Callable[[Mapping[str, Any]], dict[str, Any]], list[dict[str, Any]]]:
        """Batch-generate once, then serve per-row results to the shared evaluator.

        The evaluator calls ``predict(row)`` one row at a time; doing real
        generation that way would be slow, so we precompute in batches and look
        results up by row id.
        """
        predictions: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            predictions.extend(self.generate_batch(rows[start : start + batch_size]))
        by_id = {prediction["example_id"]: prediction for prediction in predictions}
        if len(by_id) != len(predictions):
            raise ValueError("Row ids must be unique for cached prediction lookup.")
        return (lambda row: by_id[str(row.get("id"))]), predictions


def save_checkpoint(
    directory: Path,
    *,
    model: Seq2SeqModel,
    vocab: Vocabulary,
    data_config: Seq2SeqDataConfig,
    train_config: Seq2SeqTrainConfig,
    extra: Mapping[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / CHECKPOINT_FILENAME)
    metadata = {
        "model_id": "seq2seq",
        "model_config": model.config.to_dict(),
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "vocabulary": vocab.to_dict(),
        "extra": dict(extra or {}),
    }
    (directory / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_checkpoint(
    directory: Path, device: torch.device | str = "cpu"
) -> tuple[Seq2SeqModel, Vocabulary, Seq2SeqDataConfig, dict[str, Any]]:
    metadata = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))
    vocab = Vocabulary.from_dict(metadata["vocabulary"])
    model = Seq2SeqModel(Seq2SeqModelConfig(**metadata["model_config"]))
    state = torch.load(directory / CHECKPOINT_FILENAME, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocab, Seq2SeqDataConfig(**metadata["data_config"]), metadata


# ---------------------------------------------------------------------------
# Full training loop: early stopping, best checkpoint, resume
# ---------------------------------------------------------------------------

TRAINING_STATE_FILENAME = "training_state.pt"


def save_training_state(
    directory: Path,
    *,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[dict[str, Any]],
    best: dict[str, Any] | None,
    epochs_without_improvement: int,
) -> None:
    """Extra state needed to resume (lives beside a normal checkpoint)."""
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "history": history,
            "best": best,
            "epochs_without_improvement": epochs_without_improvement,
            "torch_rng": torch.get_rng_state(),
            # GPU random state too, so dropout replays identically after a resume.
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        directory / TRAINING_STATE_FILENAME,
    )


def load_training_state(directory: Path) -> dict[str, Any]:
    # weights_only=False: the file holds our own optimizer state + history, not
    # untrusted data. Only resume from checkpoints you produced yourself.
    return torch.load(directory / TRAINING_STATE_FILENAME, map_location="cpu", weights_only=False)


def _is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Model selection: higher dev accuracy, then lower dev loss as tie-break."""
    if best is None:
        return True
    if candidate["dev_accuracy"] != best["dev_accuracy"]:
        return candidate["dev_accuracy"] > best["dev_accuracy"]
    return candidate["dev_loss"] < best["dev_loss"]


def fit(
    *,
    model: Seq2SeqModel,
    vocab: Vocabulary,
    data_config: Seq2SeqDataConfig,
    train_config: Seq2SeqTrainConfig,
    train_examples: Sequence[EncodedExample],
    dev_examples: Sequence[EncodedExample],
    dev_rows: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    device: torch.device,
    evaluate_accuracy: Callable[[Seq2SeqPredictor, Sequence[Mapping[str, Any]]], dict[str, Any]],
    resume: bool = False,
    log: Callable[[str], None] = print,
    checkpoint_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train with per-epoch dev evaluation, early stopping and best-checkpoint saving.

    * ``checkpoint_dir/best`` — weights with the best dev accuracy (selection).
    * ``checkpoint_dir/last`` — latest weights + optimizer state (for resume).

    Shuffling uses ``seed + epoch`` so a resumed run sees the same batch order
    as an uninterrupted one.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    best_dir, last_dir = checkpoint_dir / "best", checkpoint_dir / "last"
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    epochs_without_improvement = 0
    start_epoch = 1

    if resume:
        if not (last_dir / TRAINING_STATE_FILENAME).is_file():
            raise FileNotFoundError(f"Cannot resume: no training state in {last_dir}.")
        restored, restored_vocab, _, _ = load_checkpoint(last_dir, device)
        if restored_vocab.itos != vocab.itos:
            raise ValueError("Cannot resume: vocabulary differs from the saved run (were data/limit args changed?).")
        model.load_state_dict(restored.state_dict())
        state = load_training_state(last_dir)
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
        epochs_without_improvement = state["epochs_without_improvement"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        log(f"Resumed after epoch {state['epoch']}.")

    dev_loader = make_loader(dev_examples, vocab, batch_size=train_config.batch_size, shuffle=False, seed=0)
    stopped_early = False
    for epoch in range(start_epoch, train_config.max_epochs + 1):
        if epochs_without_improvement >= train_config.early_stopping_patience:
            stopped_early = True
            break
        loader = make_loader(
            train_examples, vocab, batch_size=train_config.batch_size, shuffle=True,
            seed=train_config.seed + epoch,
        )
        record: dict[str, Any] = {"epoch": epoch}
        record.update(train_epoch(model, loader, optimizer, train_config, device, log=log, epoch=epoch))
        record["dev_loss"] = evaluate_loss(model, dev_loader, device)
        predictor = Seq2SeqPredictor(model, vocab, data_config, device)
        dev_metrics = evaluate_accuracy(predictor, dev_rows)
        record["dev_accuracy"] = dev_metrics["accuracy"]
        record["dev_malformed_rate"] = dev_metrics["malformed_output_rate"]

        improved = _is_better(record, best)
        record["is_best"] = improved
        history.append(record)
        if improved:
            best = {key: record[key] for key in ("epoch", "dev_accuracy", "dev_loss")}
            epochs_without_improvement = 0
            save_checkpoint(
                best_dir, model=model, vocab=vocab, data_config=data_config,
                train_config=train_config, extra={**(checkpoint_extra or {}), "selected_epoch": epoch},
            )
        else:
            epochs_without_improvement += 1
        save_checkpoint(
            last_dir, model=model, vocab=vocab, data_config=data_config,
            train_config=train_config, extra={**(checkpoint_extra or {}), "epoch": epoch},
        )
        save_training_state(
            last_dir, optimizer=optimizer, epoch=epoch, history=history, best=best,
            epochs_without_improvement=epochs_without_improvement,
        )
        log(
            f"epoch {epoch}: train_loss={record['train_loss']:.4f} "
            f"answer_loss={record['train_answer_token_loss']:.4f} dev_loss={record['dev_loss']:.4f} "
            f"dev_acc={record['dev_accuracy']:.4f} malformed={record['dev_malformed_rate']:.4f}"
            + (" *best*" if improved else "")
        )
    else:
        stopped_early = epochs_without_improvement >= train_config.early_stopping_patience

    return {"history": history, "best": best, "stopped_early": stopped_early}