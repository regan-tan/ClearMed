import pytest
import torch

from clearmed.models.seq2seq import Seq2SeqModel, Seq2SeqModelConfig, count_parameters

PAD = 0


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Seq2SeqModel(
        Seq2SeqModelConfig(vocab_size=30, pad_id=PAD, embedding_dim=8, hidden_dim=12, attention_dim=10, num_layers=2, dropout=0.0)
    )


def batch():
    source = torch.tensor([[5, 6, 7, 8, 9], [5, 6, 7, PAD, PAD]])
    lengths = torch.tensor([5, 3])
    decoder_input = torch.tensor([[2, 10, 11], [2, 12, PAD]])
    return source, lengths, decoder_input


def test_forward_shapes(model):
    source, lengths, decoder_input = batch()
    logits = model(source, lengths, decoder_input)
    assert logits.shape == (2, 3, 30)


def test_attention_ignores_padding(model):
    source, lengths, _ = batch()
    _, attention = model.generate(source, lengths, sos_id=2, eos_id=3, max_len=4, return_attention=True)
    assert torch.allclose(attention.sum(-1), torch.ones(attention.shape[:2]))
    assert attention[1, :, 3:].abs().max() == 0  # padded positions get zero weight


def test_padding_content_does_not_change_outputs(model):
    source, lengths, decoder_input = batch()
    garbage = source.clone()
    garbage[1, 3:] = 25  # change padded slots only
    model.eval()
    assert torch.allclose(model(source, lengths, decoder_input)[1], model(garbage, lengths, decoder_input)[1])


def test_backward_reaches_every_parameter(model):
    source, lengths, decoder_input = batch()
    model(source, lengths, decoder_input).sum().backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing


def test_generation_is_deterministic_and_bounded(model):
    source, lengths, _ = batch()
    first = model.generate(source, lengths, sos_id=2, eos_id=3, max_len=6)
    second = model.generate(source, lengths, sos_id=2, eos_id=3, max_len=6)
    assert torch.equal(first, second)
    assert first.shape[0] == 2 and first.shape[1] <= 6


def test_generation_restores_training_mode(model):
    source, lengths, _ = batch()
    model.train()
    model.generate(source, lengths, sos_id=2, eos_id=3, max_len=2)
    assert model.training


def test_config_validation_and_parameter_count(model):
    with pytest.raises(ValueError):
        Seq2SeqModelConfig(vocab_size=10, pad_id=0, dropout=1.0)
    assert count_parameters(model) > 0


def test_teacher_forcing_vs_free_running_decoding(model):
    """Teacher forcing reads the given previous tokens; generation reads only its own."""
    model.eval()
    source, lengths, _ = batch()

    # 1) Teacher-forced step t depends on the supplied token at t, never on later ones.
    gold = torch.tensor([[2, 10, 11, 12], [2, 13, 14, 15]])
    altered = gold.clone()
    altered[:, 2] = 20  # change the token fed at step 2 only
    with torch.no_grad():
        logits_gold = model(source, lengths, gold)
        logits_altered = model(source, lengths, altered)
    assert torch.allclose(logits_gold[:, :2], logits_altered[:, :2])  # steps 0-1 unaffected (causal)
    assert not torch.allclose(logits_gold[:, 2], logits_altered[:, 2])  # step 2 sees the new token

    # 2) Free-running generation == teacher forcing fed with the model's OWN predictions,
    #    so inference needs no reference tokens at all.
    generated = model.generate(source, lengths, sos_id=2, eos_id=3, max_len=5)
    own_inputs = torch.cat([torch.full((2, 1), 2), generated[:, :-1]], dim=1)
    with torch.no_grad():
        replayed = model(source, lengths, own_inputs).argmax(-1)
    for row in range(2):
        tokens = generated[row].tolist()
        stop = tokens.index(3) + 1 if 3 in tokens else len(tokens)
        assert replayed[row, :stop].tolist() == tokens[:stop]