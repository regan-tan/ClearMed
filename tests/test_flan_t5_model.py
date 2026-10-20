import torch
from peft import PeftModel
from transformers import T5Config, T5ForConditionalGeneration

from clearmed.models.flan_t5_lora import (
    FlanT5LoraConfig,
    assert_lora_trainability,
    attach_lora,
    parameter_counts,
    verify_lora_target_modules,
)


def tiny_base():
    return T5ForConditionalGeneration(T5Config(vocab_size=64, d_model=16, d_ff=32, num_layers=1, num_decoder_layers=1, num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))


def test_lora_attaches_to_tiny_t5_and_freezes_base_weights():
    model = attach_lora(tiny_base(), FlanT5LoraConfig(model_name="tiny"))
    counts = parameter_counts(model)
    assert 0 < counts["trainable"] < counts["total"]
    assert_lora_trainability(model)
    assert any("SelfAttention.q.lora_A" in name for name, parameter in model.named_parameters() if parameter.requires_grad)


def test_invalid_lora_target_is_rejected():
    try:
        verify_lora_target_modules(tiny_base(), ("not_a_t5_module",))
    except ValueError as error:
        assert "not_a_t5_module" in str(error)
    else:
        raise AssertionError("Missing target module should fail before PEFT attachment.")


def test_forward_backward_update_and_adapter_reload(tmp_path):
    torch.manual_seed(7)
    base_dir, adapter_dir = tmp_path / "base", tmp_path / "adapter"
    base = tiny_base()
    base.save_pretrained(base_dir)
    model = attach_lora(base, FlanT5LoraConfig(model_name=str(base_dir), dropout=0.0))
    input_ids = torch.tensor([[2, 3, 4], [5, 6, 0]])
    attention_mask = input_ids.ne(0).long()
    labels = torch.tensor([[7, 1, -100], [8, 1, -100]])
    before = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01).step()
    assert any(not torch.equal(old, new) for old, new in zip(before, (p for p in model.parameters() if p.requires_grad), strict=True))
    model.eval()
    model.save_pretrained(adapter_dir)
    reloaded = PeftModel.from_pretrained(tiny_base().from_pretrained(base_dir), adapter_dir).eval()
    decoder_input_ids = torch.tensor([[0, 7, 1], [0, 8, 1]])
    with torch.inference_mode():
        assert torch.allclose(
            model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits,
            reloaded(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits,
        )
