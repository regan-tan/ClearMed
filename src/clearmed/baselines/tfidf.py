"""Sparse TF-IDF question--candidate baseline for four-option medical MCQs.

Each MCQ is expanded into four independent binary examples.  A shared
vocabulary represents candidate text and the element-wise interaction of the
question and candidate vectors; a final scalar is their cosine similarity.
No option position, gold answer, explanation, or other options are features.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from clearmed.data.preprocessing import (
    ANSWER_INDEX_TO_LABEL,
    OPTION_LABELS,
    get_answer_index,
    get_question_and_options,
    is_valid_qa_example,
)


@dataclass(frozen=True)
class TfidfMCQConfig:
    """Explicit, reproducible hyperparameters for :class:`TfidfMCQBaseline`."""

    ngram_range: tuple[int, int] = (1, 2)
    max_features: int | None = 50_000
    min_df: float = 2
    sublinear_tf: bool = True
    solver: str = "liblinear"
    max_iter: int = 1000
    random_state: int = 42
    class_weight: str | dict[int, float] | None = "balanced"
    batch_size: int = 2_000


def build_candidate_examples(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand one QA-valid MCQ into four binary candidate-training examples."""
    if not is_valid_qa_example(row):
        raise ValueError("Candidate construction requires a QA-valid MCQ with cop 0 through 3.")
    question, options = get_question_and_options(row)
    correct_index = get_answer_index(row)
    return [
        {"question": question, "candidate": option, "target": int(index == correct_index)}
        for index, option in enumerate(options)
    ]


class TfidfMCQBaseline:
    """A lightweight binary correctness classifier scored once per MCQ option."""

    MODEL_FILENAME = "model.joblib"
    CONFIG_FILENAME = "model_config.json"

    def __init__(self, config: TfidfMCQConfig | None = None) -> None:
        self.config = config or TfidfMCQConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.classifier: LogisticRegression | None = None
        self.effective_min_df: float | None = None
        self.training_question_count = 0
        self.training_candidate_pair_count = 0

    @staticmethod
    def _reiterable(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        """Materialize only a one-shot iterator so fitting can make multiple passes."""
        return list(rows) if isinstance(rows, Iterator) else rows

    def _new_vectorizer(self, min_df: float) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
            min_df=min_df,
            sublinear_tf=self.config.sublinear_tf,
            dtype=np.float32,
        )

    @staticmethod
    def _valid_rows(rows: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        for row in rows:
            if is_valid_qa_example(row):
                yield row

    def _training_texts(self, rows: Iterable[Mapping[str, Any]]) -> Iterator[str]:
        for row in self._valid_rows(rows):
            question, options = get_question_and_options(row)
            yield question
            yield from options

    def _features(self, questions: Sequence[str], candidates: Sequence[str]) -> sparse.csr_matrix:
        if self.vectorizer is None:
            raise RuntimeError("The TF-IDF vectorizer is not fitted.")
        question_matrix = self.vectorizer.transform(questions)
        candidate_matrix = self.vectorizer.transform(candidates)
        interaction = question_matrix.multiply(candidate_matrix)
        question_norms = np.sqrt(question_matrix.multiply(question_matrix).sum(axis=1)).A1
        candidate_norms = np.sqrt(candidate_matrix.multiply(candidate_matrix).sum(axis=1)).A1
        dot_products = interaction.sum(axis=1).A1
        cosine = np.divide(
            dot_products,
            question_norms * candidate_norms,
            out=np.zeros_like(dot_products, dtype=np.float32),
            where=(question_norms * candidate_norms) != 0,
        )
        return sparse.hstack(
            (candidate_matrix, interaction, sparse.csr_matrix(cosine.reshape(-1, 1))),
            format="csr",
            dtype=np.float32,
        )

    def _feature_batches(
        self, rows: Iterable[Mapping[str, Any]], *, with_targets: bool
    ) -> Iterator[tuple[sparse.csr_matrix, np.ndarray | None, int]]:
        questions: list[str] = []
        candidates: list[str] = []
        targets: list[int] = []
        row_count = 0

        def emit() -> tuple[sparse.csr_matrix, np.ndarray | None, int]:
            return (
                self._features(questions, candidates),
                np.asarray(targets, dtype=np.int8) if with_targets else None,
                row_count,
            )

        for row in rows:
            if with_targets and not is_valid_qa_example(row):
                continue
            # Inference intentionally does not inspect ``cop`` or ``exp``.
            question, options = get_question_and_options(row)
            questions.extend([question] * len(options))
            candidates.extend(options)
            if with_targets:
                correct_index = get_answer_index(row)
                targets.extend(int(index == correct_index) for index in range(len(options)))
            row_count += 1
            if row_count >= self.config.batch_size:
                yield emit()
                questions, candidates, targets, row_count = [], [], [], 0
        if row_count:
            yield emit()

    def fit(self, rows: Iterable[Mapping[str, Any]]) -> TfidfMCQBaseline:
        """Fit vocabulary and classifier exclusively on QA-valid training MCQs."""
        source_rows = self._reiterable(rows)
        vectorizer = self._new_vectorizer(self.config.min_df)
        try:
            vectorizer.fit(self._training_texts(source_rows))
            effective_min_df = self.config.min_df
        except ValueError as error:
            if self.config.min_df == 1 or "no terms remain" not in str(error).lower():
                raise
            vectorizer = self._new_vectorizer(1)
            vectorizer.fit(self._training_texts(source_rows))
            effective_min_df = 1
        self.vectorizer = vectorizer
        self.effective_min_df = effective_min_df

        matrices: list[sparse.csr_matrix] = []
        labels: list[np.ndarray] = []
        question_count = 0
        for matrix, targets, count in self._feature_batches(source_rows, with_targets=True):
            matrices.append(matrix)
            assert targets is not None
            labels.append(targets)
            question_count += count
        if not matrices:
            raise ValueError("No QA-valid MCQs were supplied for TF-IDF training.")
        y = np.concatenate(labels)
        if np.unique(y).size != 2:
            raise ValueError("TF-IDF training requires both positive and negative candidate labels.")
        x = sparse.vstack(matrices, format="csr")
        classifier = LogisticRegression(
            solver=self.config.solver,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
        )
        classifier.fit(x, y)
        self.classifier = classifier
        self.training_question_count = question_count
        self.training_candidate_pair_count = int(y.size)
        return self

    def _positive_scores(self, matrix: sparse.csr_matrix) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("The binary classifier is not fitted.")
        positive_positions = np.flatnonzero(self.classifier.classes_ == 1)
        if positive_positions.size != 1:
            raise RuntimeError("The classifier does not expose a positive class labelled 1.")
        scores = self.classifier.predict_proba(matrix)[:, positive_positions[0]]
        if not np.isfinite(scores).all():
            raise RuntimeError("Classifier returned non-finite candidate scores.")
        return scores

    def predict_mcq(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Score four content-only candidates and choose the earliest maximum."""
        question, options = get_question_and_options(row)
        scores = self._positive_scores(self._features([question] * 4, list(options)))
        return self._prediction_from_scores(scores)

    @staticmethod
    def _prediction_from_scores(scores: np.ndarray) -> dict[str, Any]:
        if scores.shape != (4,):
            raise RuntimeError("A valid MCQ must produce exactly four candidate scores.")
        predicted_index = int(np.argmax(scores))
        return {
            "predicted_answer": ANSWER_INDEX_TO_LABEL[predicted_index],
            "predicted_index": predicted_index,
            "candidate_scores": {
                label: float(score) for label, score in zip(OPTION_LABELS, scores, strict=True)
            },
        }

    def predict_batch(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Predict one A/B/C/D label per MCQ using sparse feature batches."""
        predictions: list[dict[str, Any]] = []
        for matrix, _, question_count in self._feature_batches(rows, with_targets=False):
            scores = self._positive_scores(matrix)
            for offset in range(question_count):
                start = offset * 4
                predictions.append(self._prediction_from_scores(scores[start : start + 4]))
        return predictions

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_) if self.vectorizer is not None else 0

    def save(self, directory: str | Path, *, overwrite: bool = False) -> Path:
        """Persist trusted local artifacts without silently replacing an experiment."""
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("Fit the baseline before saving it.")
        target = Path(directory)
        if target.exists() and any(target.iterdir()) and not overwrite:
            raise FileExistsError(f"Model directory {target} is not empty; choose a new directory.")
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target / self.MODEL_FILENAME)
        metadata = {
            "model_type": "tfidf_binary_logistic_regression",
            "config": asdict(self.config),
            "effective_min_df": self.effective_min_df,
            "vocabulary_size": self.vocabulary_size,
            "training_question_count": self.training_question_count,
            "training_candidate_pair_count": self.training_candidate_pair_count,
        }
        (target / self.CONFIG_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, directory: str | Path) -> TfidfMCQBaseline:
        """Load a model artifact produced by :meth:`save` from a trusted path."""
        path = Path(directory) / cls.MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Missing TF-IDF model artifact: {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Artifact is not a TfidfMCQBaseline model.")
        return model
