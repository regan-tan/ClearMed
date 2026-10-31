"""Exact dense retrieval over unit-normalised passage embeddings.

The corpus is small enough (~32k passages) for exact NumPy search, so no
approximate index is needed and results are fully reproducible. Equal scores
are broken by corpus order, making rankings deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDINGS_FILE = "embeddings.npy"
DOC_IDS_FILE = "doc_ids.json"
INDEX_METADATA_FILE = "index.json"
IDENTITY_KEYS = ("method_id", "model_name", "revision", "dimension")
_UNIT_TOLERANCE = 1e-3


def _check_unit_vectors(vectors: np.ndarray, name: str) -> None:
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_UNIT_TOLERANCE):
        raise ValueError(f"{name} must be L2-normalised unit vectors.")


class DenseIndex:
    """Passage vectors, their doc IDs, and the identity of the model that made them."""

    def __init__(
        self, embeddings: np.ndarray, doc_ids: Sequence[str], metadata: Mapping[str, Any]
    ) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("embeddings must be a non-empty 2-D array.")
        if len(doc_ids) != len(vectors):
            raise ValueError(f"{len(doc_ids)} doc IDs for {len(vectors)} embeddings.")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc IDs must be unique.")
        missing = [key for key in IDENTITY_KEYS if key not in metadata]
        if missing:
            raise ValueError(f"Index metadata is missing {missing}.")
        if metadata["dimension"] != vectors.shape[1]:
            raise ValueError(
                f"Metadata dimension {metadata['dimension']} differs from vectors ({vectors.shape[1]})."
            )
        _check_unit_vectors(vectors, "Passage embeddings")
        self.embeddings = vectors
        self.doc_ids = list(doc_ids)
        self.metadata = dict(metadata)

    def __len__(self) -> int:
        return len(self.doc_ids)

    def ensure_compatible(self, description: Mapping[str, Any]) -> None:
        """Refuse queries embedded by a different model, revision or size."""
        mismatched = [
            key for key in IDENTITY_KEYS if description.get(key) != self.metadata.get(key)
        ]
        if mismatched:
            details = ", ".join(
                f"{key}: index={self.metadata.get(key)!r}, query={description.get(key)!r}"
                for key in mismatched
            )
            raise ValueError(f"Query embeddings do not match this index ({details}).")

    def search(self, query_vectors: np.ndarray, k: int) -> list[list[dict[str, Any]]]:
        """Return the top *k* passages per query, best first, with cosine scores."""
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        if queries.ndim != 2 or queries.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query vectors must have dimension {self.embeddings.shape[1]}, got shape {queries.shape}."
            )
        _check_unit_vectors(queries, "Query vectors")
        top = min(k, len(self))
        positions = np.arange(len(self))
        results: list[list[dict[str, Any]]] = []
        for scores in queries @ self.embeddings.T:
            order = np.lexsort((positions, -scores))[:top]
            results.append(
                [
                    {"rank": rank, "doc_id": self.doc_ids[index], "score": float(scores[index])}
                    for rank, index in enumerate(order, start=1)
                ]
            )
        return results

    def save(self, directory: str | Path) -> None:
        """Write the index files; never replaces an existing index."""
        target = Path(directory)
        existing = [name for name in (EMBEDDINGS_FILE, DOC_IDS_FILE, INDEX_METADATA_FILE) if (target / name).exists()]
        if existing:
            raise FileExistsError(f"{target} already holds index files {existing}; use a new run directory.")
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / EMBEDDINGS_FILE, self.embeddings, allow_pickle=False)
        (target / DOC_IDS_FILE).write_text(json.dumps(self.doc_ids), encoding="utf-8")
        (target / INDEX_METADATA_FILE).write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> DenseIndex:
        source = Path(directory)
        for name in (EMBEDDINGS_FILE, DOC_IDS_FILE, INDEX_METADATA_FILE):
            if not (source / name).is_file():
                raise FileNotFoundError(f"Index file {name} is missing from {source}.")
        return cls(
            np.load(source / EMBEDDINGS_FILE, allow_pickle=False),
            json.loads((source / DOC_IDS_FILE).read_text(encoding="utf-8")),
            json.loads((source / INDEX_METADATA_FILE).read_text(encoding="utf-8")),
        )
