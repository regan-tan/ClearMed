import torch

from clearmed.data.seq2seq import Seq2SeqDataConfig, Seq2SeqMCQDataset, build_vocabulary
from clearmed.evaluation.qa import evaluate_mcq_predictions
from clearmed.models.seq2seq import Seq2SeqModel, Seq2SeqModelConfig
from clearmed.training.seq2seq import (
    Seq2SeqPredictor,
    Seq2SeqTrainConfig,
    evaluate_loss,
    load_checkpoint,
    make_loader,
    save_checkpoint,
    sequence_loss,
    set_seed,
    train_epoch,
)

DRUGS = ["aspirin", "insulin", "salbutamol", "warfarin", "metformin", "heparin", "atropine", "digoxin"]


def synthetic_rows(count=8):
    rows = []
    for index in range(count):
        options = [DRUGS[(index + offset) % len(DRUGS)] for offset in range(4)]
        cop = index % 4
        rows.append(
            {
                "id": f"syn-{index}",
                "question": f"Patient {index} needs {options[cop]}. Which drug?",
                "opa": options[0],
                "opb": options[1],
                "opc": options[2],
                "opd": options[3],
                "cop": cop,
                "exp": f"{options[cop]} is indicated." if index % 3 else "",
                "subject_name": "Pharmacology",
            }
        )
    return rows


def build(seed=42):
    set_seed(seed)
    data_config = Seq2SeqDataConfig(min_freq=1, max_explanation_tokens=8)
    rows = synthetic_rows()
    vocab = build_vocabulary(rows, data_config)
    dataset = Seq2SeqMCQDataset(rows, vocab, data_config)
    model = Seq2SeqModel(
        Seq2SeqModelConfig(vocab_size=len(vocab), pad_id=vocab.pad_id, embedding_dim=32, hidden_dim=48, attention_dim=32, dropout=0.0)
    )
    return rows, vocab, data_config, dataset, model


def test_answer_loss_weight_changes_only_weighting():
    logits = torch.randn(2, 3, 5)
    target = torch.tensor([[1, 2, 0], [3, 4, 4]])
    plain, parts = sequence_loss(logits, target, pad_id=0)
    weighted, weighted_parts = sequence_loss(logits, target, pad_id=0, answer_loss_weight=5.0)
    assert parts == weighted_parts
    assert not torch.isclose(plain, weighted)


def test_tiny_batch_overfits_and_evaluates_through_shared_evaluator():
    rows, vocab, data_config, dataset, model = build()
    train_config = Seq2SeqTrainConfig(batch_size=8, learning_rate=5e-3, weight_decay=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    loader = make_loader(dataset.examples, vocab, batch_size=8, shuffle=True, seed=42)
    device = torch.device("cpu")

    initial = evaluate_loss(model, loader, device)
    for _ in range(150):
        train_epoch(model, loader, optimizer, train_config, device)
    final = evaluate_loss(model, loader, device)
    assert final < initial * 0.1

    predictor = Seq2SeqPredictor(model, vocab, data_config)
    predict, predictions = predictor.cached_predictor(rows, batch_size=3)
    metrics = evaluate_mcq_predictions(rows, predict)
    assert metrics["accuracy"] == 1.0
    assert metrics["malformed_output_count"] == 0
    assert all(p["raw_output"].startswith("Answer: ") for p in predictions)


def test_prediction_ignores_gold_fields():
    rows, vocab, data_config, _, model = build()
    predictor = Seq2SeqPredictor(model, vocab, data_config)
    row = rows[0]
    stripped = {key: value for key, value in row.items() if key not in {"cop", "exp"}}
    assert predictor.predict_mcq(row) == predictor.predict_mcq(stripped)


def test_checkpoint_round_trip_reproduces_predictions(tmp_path):
    rows, vocab, data_config, _, model = build()
    save_checkpoint(tmp_path, model=model, vocab=vocab, data_config=data_config, train_config=Seq2SeqTrainConfig())
    restored, restored_vocab, restored_config, metadata = load_checkpoint(tmp_path)
    assert metadata["model_id"] == "seq2seq"
    original = Seq2SeqPredictor(model, vocab, data_config).generate_batch(rows)
    reloaded = Seq2SeqPredictor(restored, restored_vocab, restored_config).generate_batch(rows)
    assert original == reloaded


def test_seeded_training_is_reproducible():
    def one_epoch_loss():
        _, vocab, _, dataset, model = build(seed=7)
        loader = make_loader(dataset.examples, vocab, batch_size=4, shuffle=True, seed=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return train_epoch(model, loader, optimizer, Seq2SeqTrainConfig(), torch.device("cpu"))["train_loss"]

    assert one_epoch_loss() == one_epoch_loss()
