"""FLAN-T5 loading and verified PEFT/LoRA attachment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass(frozen=True)
class FlanT5LoraConfig:
    model_name: str = "google/flan-t5-small"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q", "v")

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.target_modules:
            raise ValueError("At least one LoRA target module is required.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_modules"] = list(self.target_modules)
        return payload


def verify_lora_target_modules(model: Any, target_modules: Sequence[str]) -> None:
    """Reject target names that do not exist in the loaded T5 architecture."""
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = sorted(set(target_modules) - suffixes)
    if missing:
        raise ValueError(
            f"LoRA target module(s) {', '.join(missing)} were not found in this model. "
            f"Available module suffixes include: {', '.join(sorted(suffixes)[:20])}."
        )


def parameter_counts(model: Any) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def assert_lora_trainability(model: Any) -> None:
    """Ensure PEFT has frozen the base weights and exposes trainable adapters."""
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoRA attachment produced no trainable parameters.")
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        raise RuntimeError("Expected frozen base model weights; trainable non-LoRA parameters: " + ", ".join(unexpected[:5]))


def attach_lora(base_model: Any, config: FlanT5LoraConfig) -> Any:
    """Attach a Seq2Seq LoRA adapter after architecture and freezing checks."""
    verify_lora_target_modules(base_model, config.target_modules)
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, r=config.rank, lora_alpha=config.alpha,
            lora_dropout=config.dropout, bias="none", target_modules=list(config.target_modules),
        ),
    )
    assert_lora_trainability(model)
    return model


def load_flan_t5_lora(config: FlanT5LoraConfig) -> tuple[Any, Any]:
    """Download/load the configured pretrained FLAN-T5 model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    return attach_lora(base_model, config), tokenizer


def load_saved_adapter(adapter_dir: str, model_name: str, *, is_trainable: bool = False) -> tuple[Any, Any]:
    """Restore a saved adapter without saving a redundant pretrained-model copy."""
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=is_trainable)
    if is_trainable:
        assert_lora_trainability(model)
    return model, tokenizer
