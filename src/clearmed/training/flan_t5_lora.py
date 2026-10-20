"""Training, greedy generation, and adapter checkpointing for FLAN-T5 LoRA."""

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


@dataclass(frozen=True)
class FlanT5TrainConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_steps: int | None = None


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
    """Perform one LoRA-only optimizer update and return the finite loss."""
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
            # This calls only content-only input formatting; no target fields are read.
            sources = [format_flan_t5_input(row, self.tokenizer, self.data_config)[0] for row in chunk]
            inputs = self.tokenizer(
                sources, padding=True, truncation=False, return_tensors="pt",
            )
            if inputs["input_ids"].shape[1] > self.data_config.max_source_length:
                raise RuntimeError("FLAN-T5 source formatting exceeded its configured maximum length.")
            output_ids = self.model.generate(
                **{key: value.to(self.device) for key, value in inputs.items()},
                do_sample=False, max_new_tokens=max_new_tokens,
            )
            for row, output in zip(chunk, output_ids, strict=True):
                raw = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                parsed = parse_generation_output(raw)
                generated.append(
                    {
                        "example_id": str(row.get("id")), "raw_output": raw,
                        "predicted_answer": parsed.predicted_answer,
                        "generated_explanation": parsed.generated_explanation,
                        "answer_parse_mode": parsed.answer_parse_mode,
                        "answer_explanation_format_compliant": parsed.answer_explanation_format_compliant,
                    }
                )
        return generated

    def cached_predictor(self, rows: Sequence[Mapping[str, Any]], **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        predictions = self.generate_batch(rows, **kwargs)
        by_id = {record["example_id"]: record for record in predictions}
        return lambda row: by_id[str(row.get("id"))], predictions


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
    payload = {
        "model_id": "flan_t5_lora", "model_name": model_name,
        "data_config": asdict(data_config), "train_config": asdict(train_config), "extra": dict(extra or {}),
    }
    (path / METADATA_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_adapter_checkpoint(path: Path, device: torch.device, *, is_trainable: bool = False) -> tuple[Any, Any, FlanT5DataConfig, dict[str, Any]]:
    """Reload an adapter checkpoint for training or inference."""
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing adapter metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model, tokenizer = load_saved_adapter(str(path), metadata["model_name"], is_trainable=is_trainable)
    model.to(device)
    return model, tokenizer, FlanT5DataConfig(**metadata["data_config"]), metadata
