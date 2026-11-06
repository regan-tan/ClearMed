import json
import re

import numpy as np
import pytest

from clearmed.retrieval.embedders import (
    METHOD_IDS,
    Embedder,
    EmbeddingConfig,
    l2_normalize,
    load_embedding_config,
)

VOCAB = ("asthma", "propranolol", "beta", "wheezing", "vitamin", "query:", "passage:")


class FakeEncoder:
    """Bag-of-words vectors: deterministic, offline, and easy to reason about."""

    def __init__(self, max_seq_length=4):
        self.max_seq_length = max_seq_length
        self.calls = []

    def encode(self, texts, *, batch_size):
        self.calls.append((list(texts), batch_size))
        return np.array(
            [[text.lower().split().count(word) for word in VOCAB] + [1.0] for text in texts],
            dtype=np.float64,
        )

    def token_counts(self, texts):
        return [len(text.split()) for text in texts]


def config(method_id="minilm", **overrides):
    values = {"method_id": method_id, "model_name": "test/model", "revision": "abc123", **overrides}
    return EmbeddingConfig(**values)


@pytest.mark.parametrize("method_id", METHOD_IDS)
def test_tracked_configs_are_valid_and_pinned(method_id):
    loaded = load_embedding_config(method_id)

    assert loaded.method_id == method_id
    assert re.fullmatch(r"[0-9a-f]{40}", loaded.revision)
    expected_prefixes = ("query: ", "passage: ") if method_id == "e5_base_v2" else ("", "")
    assert (loaded.query_prefix, loaded.document_prefix) == expected_prefixes


def test_config_rejects_unknown_methods_keys_and_mismatches(tmp_path):
    with pytest.raises(ValueError, match="Unknown method_id"):
        config("bm25")
    with pytest.raises(ValueError, match="batch_size"):
        config(batch_size=0)
    with pytest.raises(ValueError, match="max_seq_length"):
        config(max_seq_length=-1)
    (tmp_path / "minilm.json").write_text(
        json.dumps({"method_id": "e5_base_v2", "model_name": "m", "revision": "r"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="declares method_id"):
        load_embedding_config("minilm", tmp_path)
    (tmp_path / "minilm.json").write_text(
        json.dumps({"method_id": "minilm", "model_name": "m", "revision": "r", "pooling": "max"}),
        encoding="utf-8",
    )
    with pytest.raises(TypeError):
        load_embedding_config("minilm", tmp_path)


def test_e5_prefixes_are_applied_only_where_configured():
    e5_encoder, minilm_encoder = FakeEncoder(), FakeEncoder()
    e5 = Embedder(config("e5_base_v2", query_prefix="query: ", document_prefix="passage: "), e5_encoder)
    minilm = Embedder(config("minilm"), minilm_encoder)

    e5.encode_queries(["asthma"])
    e5.encode_documents(["asthma"])
    minilm.encode_queries(["asthma"])
    minilm.encode_documents(["asthma"])

    assert [call[0] for call in e5_encoder.calls] == [["query: asthma"], ["passage: asthma"]]
    assert [call[0] for call in minilm_encoder.calls] == [["asthma"], ["asthma"]]


def test_embeddings_are_float32_unit_vectors_and_batch_size_is_passed():
    encoder = FakeEncoder()
    vectors = Embedder(config(batch_size=7), encoder).encode_documents(
        ["asthma propranolol", "vitamin"]
    )

    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert encoder.calls[0][1] == 7


def test_invalid_inputs_and_vectors_are_rejected():
    embedder = Embedder(config(), FakeEncoder())
    with pytest.raises(TypeError, match="list of texts"):
        embedder.encode_queries("asthma")
    with pytest.raises(ValueError, match="At least one"):
        embedder.encode_queries([])
    with pytest.raises(ValueError, match="zero-length"):
        l2_normalize(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="NaN"):
        l2_normalize(np.array([[np.nan, 1.0]]))


def test_truncation_rate_counts_documents_longer_than_the_model_reads():
    e5 = Embedder(config("e5_base_v2", document_prefix="passage: "), FakeEncoder(max_seq_length=3))

    # With the prefix, "a b" has 3 tokens (fits) and "a b c" has 4 (truncated).
    assert e5.truncation_rate(["a b", "a b c"]) == 0.5
    assert e5.truncation_rate([]) == 0.0


def test_describe_records_identity_and_similarity():
    description = Embedder(config(), FakeEncoder(max_seq_length=256)).describe(dimension=384)

    assert description["method_id"] == "minilm"
    assert description["revision"] == "abc123"
    assert description["dimension"] == 384
    assert description["effective_max_seq_length"] == 256
    assert description["normalization"] == "l2"
    assert "cosine" in description["similarity"]
