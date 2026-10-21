"""Training, generation, and resumable checkpointing for FLAN-T5 LoRA.

The small helpers at the top remain useful for unit tests and diagnostics.  The
``run_resumable_training`` path is deliberately a compact PyTorch loop rather
than a second training framework: it records every optimizer boundary, keeps a
best inference adapter separate from the latest resumable state, and can be
continued by a later Slurm job.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from clearmed.data.flan_t5 import (
    FlanT5DataConfig,
    FlanT5Example,
    collate_flan_t5_examples,
    format_flan_t5_input,
)
from clearmed.data.seq2seq import parse_generation_output
from clearmed.models.flan_t5_lora import load_saved_adapter

METADATA_FILENAME = "adapter_metadata.json"
TRAINING_STATE_FILENAME = "training_state.pt"
TRAINING_METADATA_FILENAME = "training_metadata.json"


@dataclass(frozen=True)
class FlanT5TrainConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_steps: int | None = None
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    precision: str = "fp32"

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "gradient_accumulation_steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied.")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be one of fp32, fp16, or bf16.")


@dataclass(frozen=True)
class TrainingProgress:
    epoch: int = 0
    batch_in_epoch: int = 0
    global_step: int = 0
    best_validation_loss: float | None = None
    validation_events_without_improvement: int = 0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_inputs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "labels"}
    }


def training_step(
    model: Any, batch: Mapping[str, Any], optimizer: torch.optim.Optimizer,
    config: FlanT5TrainConfig, device: torch.device,
) -> float:
    """Perform one LoRA-only optimizer update and return the finite loss.

    This intentionally remains a one-update utility for the existing small
    tests; production gradient accumulation is handled by the resumable loop.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**_model_inputs(batch, device))
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite FLAN-T5 loss: {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], config.gradient_clip
    )
    optimizer.step()
    return float(loss.detach().cpu())


def train_steps(
    model: Any, examples: Sequence[FlanT5Example], tokenizer: Any, data_config: FlanT5DataConfig,
    train_config: FlanT5TrainConfig, device: torch.device,
) -> list[float]:
    """Run deterministic batches, stopping after ``max_steps`` when requested."""
    if not examples:
        raise ValueError("No QA-valid training examples are available.")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_config.learning_rate, weight_decay=train_config.weight_decay,
    )
    losses: list[float] = []
    for offset in range(0, len(examples), train_config.batch_size):
        batch = collate_flan_t5_examples(examples[offset : offset + train_config.batch_size], tokenizer, data_config)
        losses.append(training_step(model, batch, optimizer, train_config, device))
        if train_config.max_steps is not None and len(losses) >= train_config.max_steps:
            break
    return losses


def teacher_forced_loss(
    model: Any, examples: Sequence[FlanT5Example], tokenizer: Any,
    data_config: FlanT5DataConfig, device: torch.device, *, batch_size: int,
    precision: str = "fp32",
) -> float:
    """Return mean per-batch teacher-forced validation loss without generation."""
    if not examples:
        raise ValueError("No QA-valid validation examples are available.")
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            batch = collate_flan_t5_examples(examples[offset : offset + batch_size], tokenizer, data_config)
            with _autocast_context(device, precision):
                loss = model(**_model_inputs(batch, device)).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite FLAN-T5 validation loss: {loss.item()}")
            losses.append(float(loss.detach().cpu()))
    return sum(losses) / len(losses)


def resolve_precision(requested: str, device: torch.device) -> str:
    """Resolve ``auto`` deterministically and reject unsupported CUDA modes."""
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16.")
    if requested == "auto":
        if device.type != "cuda":
            return "fp32"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested in {"fp16", "bf16"} and device.type != "cuda":
        raise RuntimeError(f"--precision {requested} requires CUDA.")
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--precision bf16 requested but this CUDA GPU does not support bf16.")
    return requested


def _autocast_context(device: torch.device, precision: str) -> Any:
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(total_steps - step, 0)
        denominator = max(total_steps - warmup_steps, 1)
        return remaining / denominator

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def optimizer_steps_per_epoch(example_count: int, batch_size: int, accumulation_steps: int) -> int:
    batches = (example_count + batch_size - 1) // batch_size
    return (batches + accumulation_steps - 1) // accumulation_steps


def total_optimizer_steps(example_count: int, config: FlanT5TrainConfig) -> int:
    epoch_steps = optimizer_steps_per_epoch(example_count, config.batch_size, config.gradient_accumulation_steps)
    configured = epoch_steps * config.epochs
    return min(configured, config.max_steps) if config.max_steps is not None else configured


def _adapter_metadata(
    *, model_name: str, data_config: FlanT5DataConfig, train_config: FlanT5TrainConfig,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_id": "flan_t5_lora", "model_name": model_name,
        "data_config": asdict(data_config), "train_config": asdict(train_config), "extra": dict(extra or {}),
    }


def save_adapter_checkpoint(
    path: Path, *, model: Any, tokenizer: Any, model_name: str,
    data_config: FlanT5DataConfig, train_config: FlanT5TrainConfig, extra: Mapping[str, Any] | None = None,
) -> None:
    """Save PEFT adapter/tokenizer plus enough metadata for independent reload."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Adapter directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / METADATA_FILENAME).write_text(
        json.dumps(_adapter_metadata(model_name=model_name, data_config=data_config, train_config=train_config, extra=extra), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_adapter_checkpoint(path: Path, device: torch.device, *, is_trainable: bool = False) -> tuple[Any, Any, FlanT5DataConfig, dict[str, Any]]:
    """Reload an adapter checkpoint for training or inference."""
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing adapter metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model, tokenizer = load_saved_adapter(str(path), metadata["model_name"], is_trainable=is_trainable)
    model.to(device)
    return model, tokenizer, FlanT5DataConfig(**metadata["data_config"]), metadata


def _resume_identity(
    *, model_name: str, lora_config: Mapping[str, Any], data_config: FlanT5DataConfig,
    train_config: FlanT5TrainConfig, data_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model_name": model_name, "lora_config": dict(lora_config), "data_config": asdict(data_config),
        "train_config": asdict(train_config), "data_identity": dict(data_identity),
    }


def _validate_resume_identity(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("model_name", "lora_config", "data_config", "data_identity"):
        if saved.get(key) != current.get(key):
            raise ValueError(f"Resume checkpoint is incompatible: {key} differs from the requested run.")
    saved_train, current_train = dict(saved.get("train_config", {})), dict(current.get("train_config", {}))
    saved_max_steps = saved_train.pop("max_steps", None)
    current_max_steps = current_train.pop("max_steps", None)
    if saved_train != current_train:
        raise ValueError("Resume checkpoint is incompatible: optimizer/training settings differ.")
    if current_max_steps is not None and saved_max_steps is not None and current_max_steps < saved_max_steps:
        raise ValueError("A resumed --max-steps cannot be lower than the previous configured budget.")


def save_latest_training_checkpoint(
    path: Path, *, model: Any, tokenizer: Any, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: Any, progress: TrainingProgress,
    model_name: str, data_config: FlanT5DataConfig, train_config: FlanT5TrainConfig,
    lora_config: Mapping[str, Any], data_identity: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
) -> None:
    """Update a known run's resumable state at an optimizer boundary.

    ``path`` is created only after the caller has claimed a fresh run directory.
    Updating this one ``latest`` location is intentional and never replaces a
    different run's artifacts.
    """
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    tokenizer.save_pretrained(path / "adapter")
    identity = _resume_identity(
        model_name=model_name, lora_config=lora_config, data_config=data_config,
        train_config=train_config, data_identity=data_identity,
    )
    metadata = {"format_version": 1, "identity": identity, "progress": asdict(progress)}
    (path / TRAINING_METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "progress": asdict(progress), "rng_state": _capture_rng_state(), "history": list(history),
        },
        path / TRAINING_STATE_FILENAME,
    )


def load_latest_training_checkpoint(
    path: Path, *, model: Any, optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: Any, current_identity: Mapping[str, Any],
    device: torch.device,
) -> tuple[TrainingProgress, list[dict[str, Any]]]:
    """Restore a complete training state after validating its run identity."""
    metadata_path, state_path = path / TRAINING_METADATA_FILENAME, path / TRAINING_STATE_FILENAME
    if not metadata_path.is_file() or not state_path.is_file() or not (path / "adapter").is_dir():
        raise FileNotFoundError("Resume requires latest/adapter, training_metadata.json, and training_state.pt.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _validate_resume_identity(metadata.get("identity", {}), current_identity)
    state = torch.load(state_path, map_location=device, weights_only=False)
    adapter_state = torch.load(path / "adapter" / "adapter_model.safetensors", map_location=device) if False else None
    # PeftModel has already loaded the adapter before this function is called;
    # optimizer and scheduler carry the remainder of the resumable state.
    del adapter_state
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    _restore_rng_state(state["rng_state"])
    return TrainingProgress(**state["progress"]), list(state.get("history", []))


def _epoch_order(size: int, *, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(size, generator=generator).tolist()


def _batches_for_epoch(examples: Sequence[FlanT5Example], config: FlanT5TrainConfig, epoch: int) -> list[list[FlanT5Example]]:
    order = _epoch_order(len(examples), seed=config.seed, epoch=epoch)
    shuffled = [examples[index] for index in order]
    return [shuffled[offset : offset + config.batch_size] for offset in range(0, len(shuffled), config.batch_size)]


def run_resumable_training(
    *, model: Any, tokenizer: Any, train_examples: Sequence[FlanT5Example],
    validation_examples: Sequence[FlanT5Example], data_config: FlanT5DataConfig,
    train_config: FlanT5TrainConfig, device: torch.device, run_checkpoint_dir: Path,
    model_name: str, lora_config: Mapping[str, Any], data_identity: Mapping[str, Any],
    validation_interval_steps: int, checkpoint_interval_steps: int, early_stopping_patience: int | None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train through epochs with safe latest/best artifacts and optional resume."""
    if not train_examples or not validation_examples:
        raise ValueError("Training and validation both require at least one QA-valid example.")
    if validation_interval_steps <= 0 or checkpoint_interval_steps <= 0:
        raise ValueError("Validation and checkpoint intervals must be positive.")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when supplied.")
    precision = resolve_precision(train_config.precision, device)
    effective_config = FlanT5TrainConfig(**{**asdict(train_config), "precision": precision})
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=effective_config.learning_rate, weight_decay=effective_config.weight_decay,
    )
    budget = total_optimizer_steps(len(train_examples), effective_config)
    scheduler = _build_scheduler(optimizer, budget, effective_config.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    identity = _resume_identity(
        model_name=model_name, lora_config=lora_config, data_config=data_config,
        train_config=effective_config, data_identity=data_identity,
    )
    progress, history = TrainingProgress(), []
    latest_dir, best_dir = run_checkpoint_dir / "latest", run_checkpoint_dir / "best"
    if resume:
        progress, history = load_latest_training_checkpoint(
            latest_dir, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            current_identity=identity, device=device,
        )
    elif run_checkpoint_dir.exists() and any(run_checkpoint_dir.iterdir()):
        raise FileExistsError(f"Run checkpoint directory is not empty: {run_checkpoint_dir}. Use --resume for its latest state.")
    else:
        run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if progress.global_step >= budget:
        return {"progress": asdict(progress), "history": history, "total_optimizer_steps": budget, "completed": True}

    stop = False
    for epoch in range(progress.epoch, effective_config.epochs):
        batches = _batches_for_epoch(train_examples, effective_config, epoch)
        batch_start = progress.batch_in_epoch if epoch == progress.epoch else 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(batch_start, len(batches)):
            batch = collate_flan_t5_examples(batches[batch_index], tokenizer, data_config)
            with _autocast_context(device, precision):
                loss = model(**_model_inputs(batch, device)).loss
                scaled_loss = loss / effective_config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite FLAN-T5 loss: {loss.item()}")
            scaler.scale(scaled_loss).backward()
            is_update = (batch_index + 1) % effective_config.gradient_accumulation_steps == 0 or batch_index + 1 == len(batches)
            if not is_update:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], effective_config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            next_epoch = epoch + 1 if batch_index + 1 == len(batches) else epoch
            next_batch = 0 if next_epoch != epoch else batch_index + 1
            progress = TrainingProgress(
                epoch=next_epoch, batch_in_epoch=next_batch, global_step=progress.global_step + 1,
                best_validation_loss=progress.best_validation_loss,
                validation_events_without_improvement=progress.validation_events_without_improvement,
            )
            history.append({"event": "train", "global_step": progress.global_step, "epoch": epoch + 1, "loss": float(loss.detach().cpu()), "learning_rate": optimizer.param_groups[0]["lr"]})
            should_validate = progress.global_step % validation_interval_steps == 0 or progress.global_step >= budget
            if should_validate:
                validation_loss = teacher_forced_loss(model, validation_examples, tokenizer, data_config, device, batch_size=effective_config.batch_size, precision=precision)
                improved = progress.best_validation_loss is None or validation_loss < progress.best_validation_loss
                misses = 0 if improved else progress.validation_events_without_improvement + 1
                progress = TrainingProgress(
                    epoch=progress.epoch, batch_in_epoch=progress.batch_in_epoch, global_step=progress.global_step,
                    best_validation_loss=validation_loss if improved else progress.best_validation_loss,
                    validation_events_without_improvement=misses,
                )
                history.append({"event": "validation", "global_step": progress.global_step, "epoch": epoch + 1, "teacher_forced_loss": validation_loss, "improved": improved})
                if improved:
                    if best_dir.exists() and any(best_dir.iterdir()):
                        # Best is internal to the current claimed run; PEFT safely replaces its adapter files.
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                        (best_dir / METADATA_FILENAME).write_text(json.dumps(_adapter_metadata(model_name=model_name, data_config=data_config, train_config=effective_config, extra={"selection_metric": "teacher_forced_validation_loss", "selected_global_step": progress.global_step, "selected_validation_loss": validation_loss, "data_identity": dict(data_identity), "lora_config": dict(lora_config)}), indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    else:
                        save_adapter_checkpoint(best_dir, model=model, tokenizer=tokenizer, model_name=model_name, data_config=data_config, train_config=effective_config, extra={"selection_metric": "teacher_forced_validation_loss", "selected_global_step": progress.global_step, "selected_validation_loss": validation_loss, "data_identity": dict(data_identity), "lora_config": dict(lora_config)})
                if early_stopping_patience is not None and misses >= early_stopping_patience:
                    stop = True
            if progress.global_step % checkpoint_interval_steps == 0 or should_validate or stop:
                save_latest_training_checkpoint(latest_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scheduler=scheduler, scaler=scaler, progress=progress, model_name=model_name, data_config=data_config, train_config=effective_config, lora_config=lora_config, data_identity=data_identity, history=history)
            if stop or progress.global_step >= budget:
                break
        if stop or progress.global_step >= budget:
            break
    save_latest_training_checkpoint(latest_dir, model=model, tokenizer=tokenizer, optimizer=optimizer, scheduler=scheduler, scaler=scaler, progress=progress, model_name=model_name, data_config=data_config, train_config=effective_config, lora_config=lora_config, data_identity=data_identity, history=history)
    return {"progress": asdict(progress), "history": history, "total_optimizer_steps": budget, "completed": progress.global_step >= budget or stop, "precision": precision}


class FlanT5Predictor:
    """Greedy generator whose source formatting never reads gold fields."""

    def __init__(self, model: Any, tokenizer: Any, data_config: FlanT5DataConfig, device: torch.device) -> None:
        self.model, self.tokenizer, self.data_config, self.device = model, tokenizer, data_config, device
        self.model.to(device)

    @torch.inference_mode()
    def generate_batch(self, rows: Sequence[Mapping[str, Any]], *, batch_size: int = 8, max_new_tokens: int = 128) -> list[dict[str, Any]]:
        generated: list[dict[str, Any]] = []
        self.model.eval()
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            sources = [format_flan_t5_input(row, self.tokenizer, self.data_config)[0] for row in chunk]
            inputs = self.tokenizer(sources, padding=True, truncation=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] > self.data_config.max_source_length:
                raise RuntimeError("FLAN-T5 source formatting exceeded its configured maximum length.")
            output_ids = self.model.generate(**{key: value.to(self.device) for key, value in inputs.items()}, do_sample=False, max_new_tokens=max_new_tokens)
            eos_id = self.tokenizer.eos_token_id
            for row, output in zip(chunk, output_ids, strict=True):
                ids = output.detach().cpu().tolist()
                generated_tokens = max(len(ids) - 1, 0)  # T5 includes its decoder-start token.
                eos_observed = eos_id is not None and eos_id in ids
                raw = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                parsed = parse_generation_output(raw)
                generated.append({
                    "example_id": str(row.get("id")), "raw_output": raw,
                    "predicted_answer": parsed.predicted_answer, "generated_explanation": parsed.generated_explanation,
                    "answer_parse_mode": parsed.answer_parse_mode,
                    "answer_explanation_format_compliant": parsed.answer_explanation_format_compliant,
                    "generation_token_count": generated_tokens, "generation_eos_observed": eos_observed,
                    "generation_length_limited": not eos_observed and generated_tokens >= max_new_tokens,
                })
        return generated

    def cached_predictor(self, rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        predictions = self.generate_batch(rows, **kwargs)
        by_id = {record["example_id"]: record for record in predictions}
        return lambda row: by_id[str(row.get("id"))], predictions
