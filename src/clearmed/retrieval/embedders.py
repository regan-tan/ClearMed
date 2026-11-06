"""One embedding interface for the three Part II retrieval models.

``minilm``, ``e5_base_v2`` and ``biolord_2023`` differ only in their pinned
model and in E5's required ``query: `` / ``passage: `` prefixes, so each is a
configuration file in ``configs/retrieval/`` rather than its own class. Every
vector is L2-normalised here, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

METHOD_IDS = ("minilm", "e5_base_v2", "biolord_2023")
CONFIG_DIR = Path("configs/retrieval")
_NORM_TOLERANCE = 1e-12


@dataclass(frozen=True)
class EmbeddingConfig:
    method_id: str
    model_name: str
    revision: str
    query_prefix: str = ""
    document_prefix: str = ""
    max_seq_length: int | None = None
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.method_id not in METHOD_IDS:
            raise ValueError(f"Unknown method_id {self.method_id!r}; expected one of {METHOD_IDS}.")
        if not self.model_name or not self.revision:
            raise ValueError("model_name and a pinned revision are required.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.max_seq_length is not None and (
            isinstance(self.max_seq_length, bool)
            or not isinstance(self.max_seq_length, int)
            or self.max_seq_length <= 0
        ):
            raise ValueError("max_seq_length must be a positive integer or null.")


def load_embedding_config(method_id: str, config_dir: str | Path = CONFIG_DIR) -> EmbeddingConfig:
    """Read ``<config_dir>/<method_id>.json``; unknown keys are rejected."""
    if method_id not in METHOD_IDS:
        raise ValueError(f"Unknown method_id {method_id!r}; expected one of {METHOD_IDS}.")
    path = Path(config_dir) / f"{method_id}.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    config = EmbeddingConfig(**values)
    if config.method_id != method_id:
        raise ValueError(f"{path} declares method_id {config.method_id!r}, expected {method_id!r}.")
    return config


class TextEncoder(Protocol):
    """The minimal model surface the embedder needs, so tests can use a fake."""

    max_seq_length: int

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray: ...

    def token_counts(self, texts: Sequence[str]) -> list[int]: ...


class SentenceTransformerEncoder:
    """A pinned ``sentence-transformers`` model."""

    def __init__(self, config: EmbeddingConfig, *, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.model_name, revision=config.revision, device=device)
        if config.max_seq_length is not None:
            self._model.max_seq_length = config.max_seq_length
        self.max_seq_length = int(self._model.max_seq_length)

    def encode(self, texts: Sequence[str], *, batch_size: int) -> np.ndarray:
        return self._model.encode(
            list(texts), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=len(texts) > 1000
        )

    def token_counts(self, texts: Sequence[str]) -> list[int]:
        return [len(ids) for ids in self._model.tokenizer(list(texts))["input_ids"]]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return float32 unit vectors; a zero or non-finite vector is an error."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array of vectors, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embeddings contain NaN or infinite values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= _NORM_TOLERANCE):
        raise ValueError("Cannot normalise a zero-length embedding.")
    return array / norms


class Embedder:
    """Applies a method's prefixes and normalisation around any encoder."""

    def __init__(self, config: EmbeddingConfig, encoder: TextEncoder) -> None:
        self.config = config
        self._encoder = encoder

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("Pass a list of texts, not a single string.")
        prefixed = [prefix + text for text in texts]
        if not prefixed:
            raise ValueError("At least one text is required.")
        return l2_normalize(self._encoder.encode(prefixed, batch_size=self.config.batch_size))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config.query_prefix)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.config.document_prefix)

    def truncation_rate(self, documents: Sequence[str]) -> float:
        """Share of documents (with prefix) longer than the model reads."""
        if not documents:
            return 0.0
        counts = self._encoder.token_counts([self.config.document_prefix + text for text in documents])
        return sum(count > self._encoder.max_seq_length for count in counts) / len(counts)

    def describe(self, dimension: int) -> dict[str, Any]:
        """Metadata AGENTS.md requires for every embedding run."""
        return {
            **asdict(self.config),
            "effective_max_seq_length": self._encoder.max_seq_length,
            "dimension": dimension,
            "normalization": "l2",
            "similarity": "cosine (dot product of unit vectors)",
        }
