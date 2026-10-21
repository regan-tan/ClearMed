from pathlib import Path

import torch
from test_flan_t5_data import ToyTokenizer, row
from transformers import T5Config, T5ForConditionalGeneration

from clearmed.data.flan_t5 import FlanT5DataConfig, FlanT5MCQDataset
from clearmed.data.seq2seq import parse_answer_label, parse_generation_output
from clearmed.models.flan_t5_lora import FlanT5LoraConfig, attach_lora
from clearmed.training.flan_t5_lora import (
    FlanT5TrainConfig,
    optimizer_steps_per_epoch,
    run_resumable_training,
    training_step,
)


class SavingToyTokenizer(ToyTokenizer):
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "toy-tokenizer.txt").write_text("toy", encoding="utf-8")


def test_training_step_updates_lora_parameters():
    tokenizer = ToyTokenizer()
    base = T5ForConditionalGeneration(T5Config(vocab_size=128, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model = attach_lora(base, FlanT5LoraConfig(model_name="tiny", dropout=0.0))
    dataset = FlanT5MCQDataset([row(), row(exp="")], tokenizer, FlanT5DataConfig(80, 20))
    from clearmed.data.flan_t5 import collate_flan_t5_examples
    batch = collate_flan_t5_examples(dataset.examples, tokenizer, FlanT5DataConfig(80, 20))
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01)
    before = next(parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad)
    loss = training_step(model, batch, optimizer, FlanT5TrainConfig(batch_size=2), torch.device("cpu"))
    after = next(parameter for parameter in model.parameters() if parameter.requires_grad)
    assert loss > 0 and not torch.equal(before, after)


def test_strict_answer_parser_keeps_malformed_output_malformed():
    assert parse_answer_label("Answer: C\nExplanation: valid") == "C"
    assert parse_answer_label("C.") == "C"
    assert parse_answer_label("The answer is C") is None
    assert parse_answer_label("B12.") is None
    assert not parse_generation_output("A").answer_explanation_format_compliant


def test_resumable_training_continues_optimizer_steps_and_keeps_best_separate(tmp_path):
    tokenizer = SavingToyTokenizer()
    base = T5ForConditionalGeneration(T5Config(vocab_size=128, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
    model = attach_lora(base, FlanT5LoraConfig(model_name="tiny", dropout=0.0))
    examples = FlanT5MCQDataset([row(), {**row(), "id": "example-2"}], tokenizer, FlanT5DataConfig(80, 20)).examples
    common = {"model": model, "tokenizer": tokenizer, "train_examples": examples, "validation_examples": examples, "data_config": FlanT5DataConfig(80, 20), "device": torch.device("cpu"), "run_checkpoint_dir": tmp_path / "run", "model_name": "tiny", "lora_config": FlanT5LoraConfig(model_name="tiny", dropout=0.0).to_dict(), "data_identity": {"train": "synthetic", "dev": "synthetic"}, "validation_interval_steps": 1, "checkpoint_interval_steps": 1, "early_stopping_patience": None}
    first = run_resumable_training(train_config=FlanT5TrainConfig(batch_size=1, max_steps=1, epochs=1), resume=False, **common)
    assert first["progress"]["global_step"] == 1
    assert (tmp_path / "run" / "best" / "adapter_config.json").is_file()
    assert (tmp_path / "run" / "latest" / "training_state.pt").is_file()
    resumed = run_resumable_training(train_config=FlanT5TrainConfig(batch_size=1, max_steps=2, epochs=1), resume=True, **common)
    assert resumed["progress"]["global_step"] == 2
    assert optimizer_steps_per_epoch(5, 2, 2) == 2
