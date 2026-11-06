import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from test_embedders import FakeEncoder

from clearmed.retrieval.embedders import Embedder, load_embedding_config
from clearmed.retrieval.index import DenseIndex

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_retrieval.py"
SCRIPT_SPEC = spec_from_file_location("run_retrieval_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)

PASSAGES = {"beta": "propranolol beta wheezing asthma", "vit": "vitamin", "asthma": "asthma"}


def query_record(query_id, question_id, follow_up, gold_label="C"):
    return {
        "query_id": query_id,
        "question_id": question_id,
        "query_set": "evaluation",
        "follow_up_type": "why_correct",
        "follow_up": follow_up,
        "question": "Which drug should be avoided in asthma?",
        "options": {"A": "vitamin", "B": "losartan", "C": "propranolol", "D": "thiazide"},
        "gold_label": gold_label,
    }


def setup_run(tmp_path, method_id="minilm"):
    embedder = Embedder(load_embedding_config(method_id), FakeEncoder())
    vectors = embedder.encode_documents(list(PASSAGES.values()))
    description = embedder.describe(dimension=vectors.shape[1])
    DenseIndex(
        vectors,
        list(PASSAGES),
        {key: description[key] for key in ("method_id", "model_name", "revision", "dimension")},
    ).save(tmp_path / "index")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "".join(
            json.dumps(item) + "\n"
            for item in (
                query_record("evaluation-001", "v1", "Why is beta blockade dangerous here?"),
                query_record("evaluation-002", "v2", "Which vitamin is this?", gold_label="A"),
            )
        ),
        encoding="utf-8",
    )
    return queries


def run(tmp_path, queries, **overrides):
    arguments = {
        "method_id": "minilm",
        "queries_path": queries,
        "index_dir": tmp_path / "index",
        "output_dir": tmp_path / "out",
        "run_id": "test",
        "stage": "rq3",
        "answer_source": "gold",
        "k": 2,
        "encoder": FakeEncoder(),
        **overrides,
    }
    return SCRIPT_MODULE.run(**arguments)


def test_rankings_store_top_k_with_scores_and_query_text(tmp_path):
    queries = setup_run(tmp_path)
    config = run(tmp_path, queries)

    rows = [json.loads(line) for line in (tmp_path / "out" / "rankings.jsonl").read_text("utf-8").splitlines()]
    assert [row["query_id"] for row in rows] == ["evaluation-001", "evaluation-002"]
    assert all(len(row["results"]) == 2 for row in rows)
    assert rows[0]["results"][0]["doc_id"] == "beta"
    assert rows[0]["query_text"].startswith("Follow-up: Why is beta blockade dangerous here?\nAnswer: C. propranolol")
    assert rows[1]["query_text"].splitlines()[1] == "Answer: A. vitamin"
    assert config["answer_source"] == "gold" and config["k"] == 2
    assert config["query_count"] == 2
    assert len(config["queries_sha256"]) == 64


def test_predicted_answers_replace_gold_and_are_required(tmp_path):
    queries = setup_run(tmp_path)
    run(tmp_path, queries, stage="rq4", answer_source="predicted", predictions={"v1": "A", "v2": "B"})
    rows = [json.loads(line) for line in (tmp_path / "out" / "rankings.jsonl").read_text("utf-8").splitlines()]
    assert rows[0]["query_text"].splitlines()[1] == "Answer: A. vitamin"

    with pytest.raises(ValueError, match="No predicted answer"):
        run(tmp_path, queries, output_dir=tmp_path / "o2", answer_source="predicted", predictions={"v1": "A"})
    with pytest.raises(ValueError, match="may not use the 'gold' answer"):
        run(tmp_path, queries, output_dir=tmp_path / "o3", stage="rq4")


def test_refuses_mismatched_index_and_overwrites(tmp_path):
    queries = setup_run(tmp_path)
    run(tmp_path, queries)
    with pytest.raises(FileExistsError, match="not empty"):
        run(tmp_path, queries)
    other = tmp_path / "configs"
    other.mkdir()
    values = json.loads((Path("configs/retrieval") / "minilm.json").read_text(encoding="utf-8"))
    (other / "minilm.json").write_text(json.dumps({**values, "revision": "different"}), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match this index"):
        run(tmp_path, queries, output_dir=tmp_path / "o4", config_dir=other)
