import pytest
import torch

from clearmed.data.seq2seq import (
    ANSWER_TOKENS,
    EOS,
    SPECIAL_TOKENS,
    Seq2SeqDataConfig,
    Seq2SeqMCQDataset,
    Vocabulary,
    build_vocabulary,
    collate_examples,
    encode_source,
    encode_target,
    parse_answer_label,
    parse_explanation,
    render_output,
    tokenize,
)


def make_row(index=0, *, exp="Asthma is reversible airway obstruction.", cop=2, question=None):
    return {
        "id": f"row-{index}",
        "question": question or f"Which drug treats asthma case {index}?",
        "opa": "Aspirin",
        "opb": "Insulin",
        "opc": "Salbutamol inhaler",
        "opd": "Warfarin",
        "cop": cop,
        "exp": exp,
        "subject_name": "Medicine",
    }


@pytest.fixture
def config():
    return Seq2SeqDataConfig(min_freq=1, max_question_tokens=8, max_option_tokens=2, max_explanation_tokens=4)


@pytest.fixture
def vocab(config):
    return build_vocabulary([make_row(0), make_row(1)], config)


def test_tokenize_is_lowercase_and_splits_punctuation():
    assert tokenize("Beta-blocker, 5mg!") == ["beta-blocker", ",", "5mg", "!"]


def test_vocabulary_is_training_only_and_deterministic(config):
    train_vocab = build_vocabulary([make_row(0)], config)
    assert "zebra" not in train_vocab.stoi
    assert train_vocab.itos == build_vocabulary([make_row(0)], config).itos
    assert tuple(train_vocab.itos[: len(SPECIAL_TOKENS)]) == SPECIAL_TOKENS


def test_min_freq_and_max_size_are_respected():
    rows = [make_row(0), make_row(1)]
    vocab = build_vocabulary(rows, Seq2SeqDataConfig(min_freq=2))
    assert "aspirin" in vocab.stoi  # appears in both rows
    assert "0" not in vocab.stoi  # appears once
    capped = build_vocabulary(rows, Seq2SeqDataConfig(min_freq=1, max_vocab_size=len(SPECIAL_TOKENS) + 3))
    assert len(capped) == len(SPECIAL_TOKENS) + 3


def test_source_never_depends_on_gold_fields(vocab, config):
    row = make_row()
    leaked = {**row, "cop": 0, "exp": "completely different explanation"}
    assert encode_source(row, vocab, config) == encode_source(leaked, vocab, config)
    no_gold = {key: value for key, value in row.items() if key not in {"cop", "exp"}}
    assert encode_source(no_gold, vocab, config)[0] == encode_source(row, vocab, config)[0]


def test_truncation_never_drops_an_option(vocab):
    config = Seq2SeqDataConfig(min_freq=1, max_question_tokens=8, max_option_tokens=1)
    long_question = " ".join(["word"] * 50)
    ids, question_truncated, options_truncated = encode_source(make_row(question=long_question), vocab, config)
    tokens = vocab.decode(ids)
    assert question_truncated
    assert options_truncated == 1  # only "Salbutamol inhaler" exceeds 1 token
    for marker in ("<q>", "<A>", "<B>", "<C>", "<D>"):
        assert tokens.count(marker) == 1
    assert len(ids) <= config.max_source_len


def test_target_starts_with_answer_and_handles_missing_explanation(vocab, config):
    ids, has_exp, truncated = encode_target(make_row(cop=2), vocab, config)
    tokens = vocab.decode(ids)
    assert tokens[0] == "<ans_C>" and tokens[-1] == EOS
    assert has_exp and truncated  # 6 explanation tokens > limit of 4
    assert len(ids) == config.max_target_len

    ids, has_exp, _ = encode_target(make_row(cop=1, exp="   "), vocab, config)
    assert vocab.decode(ids) == ["<ans_B>", EOS]  # answer-only, nothing fabricated
    assert not has_exp


def test_dataset_skips_invalid_rows_and_reports_coverage(vocab, config):
    rows = [make_row(0), make_row(1, exp=None), {**make_row(2), "opd": ""}]
    dataset = Seq2SeqMCQDataset(rows, vocab, config)
    stats = dataset.statistics(vocab)
    assert len(dataset) == 2 and stats["skipped_invalid_rows"] == 1
    assert stats["with_explanation"] == 1 and stats["answer_only_targets"] == 1


def test_collate_builds_shifted_teacher_forcing_inputs(vocab, config):
    dataset = Seq2SeqMCQDataset([make_row(0), make_row(1, exp="")], vocab, config)
    batch = collate_examples([dataset[0], dataset[1]], vocab)
    assert batch.decoder_input[:, 0].eq(vocab.sos_id).all()
    assert torch.equal(batch.decoder_input[0, 1:], batch.decoder_target[0, :-1])
    assert batch.decoder_target[1, 2:].eq(vocab.pad_id).all()
    assert batch.source.shape[0] == batch.source_lengths.shape[0] == 2


def test_render_and_strict_parse_round_trip(vocab):
    ids = vocab.encode(["<ans_C>", "asthma", EOS, "ignored"])
    raw = render_output(ids, vocab)
    assert raw == "Answer: C\nExplanation: asthma"
    assert parse_answer_label(raw) == "C"
    assert parse_explanation(raw) == "asthma"


@pytest.mark.parametrize(
    "tokens",
    [["asthma", "<ans_C>"], [EOS], ["<ans_C>"[:-1]]],
)
def test_malformed_outputs_are_not_coerced(vocab, tokens):
    raw = render_output(vocab.encode(tokens), vocab)
    assert parse_answer_label(raw) is None


@pytest.mark.parametrize("text", ["answer: C", "Answer: E", "The Answer: C", "Answer: CD"])
def test_strict_parser_rejects_lookalikes(text):
    assert parse_answer_label(text) is None


def test_vocabulary_round_trips_through_dict(vocab):
    assert Vocabulary.from_dict(vocab.to_dict()).itos == vocab.itos
    assert set(vocab.decode(vocab.answer_ids)) == set(ANSWER_TOKENS)
