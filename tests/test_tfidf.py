import math

import numpy as np
import pytest
from scipy import sparse

from clearmed.baselines.tfidf import (
    TfidfMCQBaseline,
    TfidfMCQConfig,
    build_candidate_examples,
)


def make_row(index=0, *, question=None, subject="Medicine"):
    return {
        "id": f"row-{index}",
        "question": question or f"Which therapy treats condition {index}?",
        "opa": f"therapy {index} alpha",
        "opb": f"therapy {index} beta",
        "opc": f"therapy {index} gamma",
        "opd": f"therapy {index} delta",
        "cop": index % 4,
        "exp": f"Explanation {index}",
        "subject_name": subject,
        "topic_name": "Topic",
    }


@pytest.fixture
def trained_model():
    rows = [make_row(index) for index in range(8)]
    return TfidfMCQBaseline(TfidfMCQConfig(min_df=1, max_features=100, batch_size=2)).fit(rows)


def test_candidate_construction_uses_four_options_and_one_positive():
    row = make_row(2)
    examples = build_candidate_examples(row)

    assert len(examples) == 4
    assert [example["target"] for example in examples] == [0, 0, 1, 0]
    assert all(set(example) == {"question", "candidate", "target"} for example in examples)
    assert "A" not in examples[0]["candidate"]


@pytest.mark.parametrize("answer_index,label", tuple(enumerate("ABCD")))
def test_prediction_returns_finite_scores_and_canonical_label(trained_model, answer_index, label):
    row = make_row(answer_index)
    prediction = trained_model.predict_mcq(row)

    assert prediction["predicted_answer"] in "ABCD"
    assert prediction["predicted_index"] in range(4)
    assert list(prediction["candidate_scores"]) == list("ABCD")
    assert all(math.isfinite(score) for score in prediction["candidate_scores"].values())
    assert prediction["predicted_answer"] == "ABCD"[prediction["predicted_index"]]


def test_inference_is_gold_and_explanation_independent(trained_model):
    row = make_row(1)
    original = trained_model.predict_mcq(row)
    changed = {**row, "cop": 3, "exp": "Completely different explanation"}

    assert trained_model.predict_mcq(changed) == original
    assert trained_model.predict_mcq({key: value for key, value in row.items() if key not in {"cop", "exp"}}) == original


def test_features_are_sparse_and_development_terms_do_not_expand_vocabulary(trained_model):
    features = trained_model._features(["unseen development only term"], ["new candidate term"])
    same_candidate = "condition therapy"
    first = trained_model._features(["condition therapy"], [same_candidate])
    second = trained_model._features(["condition"], [same_candidate])

    assert sparse.issparse(features)
    assert "unseen" not in trained_model.vectorizer.vocabulary_
    assert "development" not in trained_model.vectorizer.vocabulary_
    vocabulary_size = trained_model.vocabulary_size
    assert (first[:, vocabulary_size : vocabulary_size * 2] != second[:, vocabulary_size : vocabulary_size * 2]).nnz


def test_ties_select_earliest_option(trained_model, monkeypatch):
    monkeypatch.setattr(trained_model, "_positive_scores", lambda matrix: np.array([0.4, 0.4, 0.1, 0.1]))

    assert trained_model.predict_mcq(make_row())["predicted_answer"] == "A"


def test_save_load_and_invalid_input(tmp_path, trained_model):
    row = make_row()
    trained_model.save(tmp_path / "model")
    loaded = TfidfMCQBaseline.load(tmp_path / "model")

    assert loaded.predict_mcq(row) == trained_model.predict_mcq(row)
    with pytest.raises(ValueError, match="question"):
        trained_model.predict_mcq({**row, "question": ""})
    with pytest.raises(FileExistsError):
        trained_model.save(tmp_path / "model")


def test_probability_column_respects_classifier_classes(trained_model):
    scores = trained_model._positive_scores(trained_model._features(["condition 1"], ["therapy 1 beta"]))

    assert scores.shape == (1,)
    assert 0 <= scores[0] <= 1


def test_batch_prediction_matches_single_prediction(trained_model):
    rows = [make_row(0), make_row(1)]

    assert trained_model.predict_batch(rows) == [trained_model.predict_mcq(row) for row in rows]
