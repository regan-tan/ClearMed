import json
import sys
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from test_embedders import FakeEncoder

from clearmed.retrieval.embedders import Embedder, load_embedding_config
from clearmed.retrieval.index import DenseIndex

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_retrieval_index.py"
SCRIPT_SPEC = spec_from_file_location("build_retrieval_index_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

METADATA = {"method_id": "minilm", "model_name": "test/model", "revision": "abc123", "dimension": 2}


def unit(*rows):
    array = np.array(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def small_index():
    return DenseIndex(unit([1, 0], [0.6, 0.8], [0, 1]), ["d0", "d1", "d2"], METADATA)


def test_search_ranks_by_cosine_and_trims_k():
    results = small_index().search(unit([1, 0.1]), k=5)

    assert [hit["doc_id"] for hit in results[0]] == ["d0", "d1", "d2"]
    assert [hit["rank"] for hit in results[0]] == [1, 2, 3]
    scores = [hit["score"] for hit in results[0]]
    assert scores == sorted(scores, reverse=True)


def test_ties_break_by_corpus_order_and_runs_repeat_exactly():
    index = DenseIndex(unit([1, 0], [1, 0], [0, 1]), ["first", "second", "third"], METADATA)
    query = unit([1, 0])

    assert [hit["doc_id"] for hit in index.search(query, k=2)[0]] == ["first", "second"]
    assert index.search(query, k=3) == index.search(query, k=3)


def test_search_accepts_one_vector_or_a_batch():
    index = small_index()

    assert index.search(unit([0, 1])[0], k=1)[0][0]["doc_id"] == "d2"
    batch = index.search(unit([1, 0], [0, 1]), k=1)
    assert [hits[0]["doc_id"] for hits in batch] == ["d0", "d2"]


def test_search_rejects_bad_queries_and_k():
    index = small_index()
    with pytest.raises(ValueError, match="dimension 2"):
        index.search(unit([1, 0, 0]), k=1)
    with pytest.raises(ValueError, match="unit vectors"):
        index.search(np.array([[2.0, 0.0]]), k=1)
    for bad_k in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            index.search(unit([1, 0]), k=bad_k)


def test_constructor_rejects_inconsistent_indexes():
    vectors = unit([1, 0], [0, 1])
    with pytest.raises(ValueError, match="2 embeddings"):
        DenseIndex(vectors, ["only-one"], METADATA)
    with pytest.raises(ValueError, match="unique"):
        DenseIndex(vectors, ["same", "same"], METADATA)
    with pytest.raises(ValueError, match="unit vectors"):
        DenseIndex(np.array([[3.0, 4.0], [0.0, 1.0]]), ["a", "b"], METADATA)
    with pytest.raises(ValueError, match="missing"):
        DenseIndex(vectors, ["a", "b"], {"method_id": "minilm"})
    with pytest.raises(ValueError, match="dimension"):
        DenseIndex(vectors, ["a", "b"], {**METADATA, "dimension": 3})


def test_save_load_round_trip_and_no_overwrite(tmp_path):
    index = small_index()
    index.save(tmp_path / "idx")
    restored = DenseIndex.load(tmp_path / "idx")

    query = unit([0.3, 0.7])
    assert restored.search(query, k=3) == index.search(query, k=3)
    assert restored.metadata == METADATA
    with pytest.raises(FileExistsError, match="already holds"):
        index.save(tmp_path / "idx")
    with pytest.raises(FileNotFoundError, match="missing"):
        DenseIndex.load(tmp_path / "empty")


def test_queries_from_a_different_model_are_refused():
    index = small_index()
    index.ensure_compatible(METADATA)
    with pytest.raises(ValueError, match="revision"):
        index.ensure_compatible({**METADATA, "revision": "other"})
    with pytest.raises(ValueError, match="method_id"):
        index.ensure_compatible({**METADATA, "method_id": "e5_base_v2"})


PASSAGES = [
    {"doc_id": "beta", "text": "propranolol beta wheezing asthma"},
    {"doc_id": "vit", "text": "vitamin vitamin"},
    {"doc_id": "asthma", "text": "asthma"},
]


def write_corpus(directory, passages=PASSAGES):
    directory.mkdir(parents=True)
    body = "".join(json.dumps(passage) + "\n" for passage in passages).encode("utf-8")
    (directory / "passages.jsonl").write_bytes(body)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "run_id": "corpus-test",
                "passages_file": "passages.jsonl",
                "passages_sha256": sha256(body).hexdigest(),
                "passages_out": len(passages),
            }
        ),
        encoding="utf-8",
    )


def test_script_builds_searchable_index_with_metadata(tmp_path):
    write_corpus(tmp_path / "corpus")
    output_dir = tmp_path / "out"

    config = SCRIPT_MODULE.run(
        method_id="e5_base_v2",
        corpus_run=tmp_path / "corpus",
        output_dir=output_dir,
        run_id="test",
        encoder=FakeEncoder(max_seq_length=4),
    )

    assert config["corpus"]["passages_indexed"] == 3
    assert config["embedding"]["query_prefix"] == "query: "
    assert config["truncation_rate"] == pytest.approx(1 / 3)
    assert json.loads((output_dir / "config.json").read_text(encoding="utf-8")) == config
    index = DenseIndex.load(output_dir)
    assert index.metadata["passages_sha256"] == config["corpus"]["passages_sha256"]
    embedder = Embedder(load_embedding_config("e5_base_v2"), FakeEncoder())
    index.ensure_compatible(embedder.describe(dimension=index.embeddings.shape[1]))
    hits = index.search(embedder.encode_queries(["asthma propranolol"]), k=2)[0]
    assert hits[0]["doc_id"] == "beta"


def test_script_limit_overwrite_and_checksum_guards(tmp_path):
    write_corpus(tmp_path / "corpus")
    config = SCRIPT_MODULE.run(
        method_id="minilm",
        corpus_run=tmp_path / "corpus",
        output_dir=tmp_path / "smoke",
        run_id="smoke",
        limit=2,
        encoder=FakeEncoder(),
    )
    assert config["corpus"]["passages_indexed"] == 2
    assert config["corpus"]["limit"] == 2
    with pytest.raises(FileExistsError, match="not empty"):
        SCRIPT_MODULE.run(
            method_id="minilm",
            corpus_run=tmp_path / "corpus",
            output_dir=tmp_path / "smoke",
            run_id="smoke",
            encoder=FakeEncoder(),
        )
    (tmp_path / "corpus" / "passages.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        SCRIPT_MODULE.run(
            method_id="minilm",
            corpus_run=tmp_path / "corpus",
            output_dir=tmp_path / "other",
            run_id="other",
            encoder=FakeEncoder(),
        )
