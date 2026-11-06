import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

METHODS = ("minilm", "e5_base_v2", "biolord_2023")


def load_script(name):
    spec = spec_from_file_location(f"{name}_for_test", Path(__file__).parents[1] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALIBRATE = load_script("calibrate_thresholds")
EVALUATE = load_script("evaluate_retrieval")

# stage -> query_id -> (follow_up_type, {method: [(doc, score), ...]})
RANKINGS = {
    "A": {
        "c1": ("why_correct", {m: [("a1", 0.9), ("a2", 0.5)] for m in METHODS}),
        "c2": ("mechanism", {m: [("a3", 0.8), ("a4", 0.4)] for m in METHODS}),
        "c3": ("off_topic", {m: [("a5", 0.6), ("a6", 0.3)] for m in METHODS}),
    },
    "B": {
        "e1": ("why_correct", {"minilm": [("b1", 0.9), ("b2", 0.8)], "e5_base_v2": [("b2", 0.9), ("b1", 0.7)],
                               "biolord_2023": [("b3", 0.6), ("b1", 0.5)]}),
        "e2": ("key_clue", {m: [("b4", 0.7), ("b5", 0.6)] for m in METHODS}),
        "e3": ("off_topic", {m: [("b6", 0.85), ("b7", 0.2)] for m in METHODS}),
    },
}
RELEVANT = {"a1", "a3", "b1", "b5"}


def write_fixture(root):
    judgments, stages = [], {}
    for stage, queries in RANKINGS.items():
        sha = f"sha-{stage}"
        run_id = f"run-{stage}"
        stages[stage] = {"queries_sha256": sha, "rankings_run_id": run_id}
        for method in METHODS:
            run_dir = root / "rankings" / method / run_id
            run_dir.mkdir(parents=True)
            rows = [
                {"query_id": q, "follow_up_type": t, "results": [
                    {"rank": i, "doc_id": d, "score": s} for i, (d, s) in enumerate(ranked[method], start=1)
                ]}
                for q, (t, ranked) in queries.items()
            ]
            (run_dir / "rankings.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
            index_dir = root / "index" / method
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "config.json").write_text(json.dumps({"build_seconds": 5, "index_size_bytes": 100}), "utf-8")
            config = {"queries_sha256": sha, "answer_source": "gold", "seconds_per_query": 0.01,
                      "embedding": {"dimension": 8}, "index_dir": index_dir.as_posix()}
            (run_dir / "config.json").write_text(json.dumps(config), "utf-8")
        docs = {(q, d) for q, (_, ranked) in queries.items() for m in METHODS for d, _ in ranked[m]}
        judgments += [{"stage": stage, "query_id": q, "doc_id": d, "relevant": int(d in RELEVANT)}
                      for q, d in sorted(docs)]
    pool_dir = root / "judging" / "pool-1"
    pool_dir.mkdir(parents=True)
    (pool_dir / "pool.json").write_text(json.dumps({"stages": stages}), "utf-8")
    (pool_dir / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments), "utf-8")


def paths(root):
    return {"judging_root": root / "judging", "rankings_root": root / "rankings",
            "thresholds_path": root / "configs" / "gate_thresholds.json"}


def test_calibration_freezes_thresholds_once(tmp_path):
    write_fixture(tmp_path)
    summary = CALIBRATE.run(pool_id="pool-1", run_id="cal", output_root=tmp_path / "cal", **paths(tmp_path))

    # Stage A top-1 scores 0.9 (relevant), 0.8 (relevant), 0.6 (off-topic, not relevant) -> threshold 0.8.
    for method in METHODS:
        result = summary["methods"][method]
        assert result["chosen"]["threshold"] == 0.8 and result["chosen"]["f1"] == 1.0
        assert (result["off_topic_rejected"], result["off_topic_total"]) == (1, 1)
    frozen = json.loads((tmp_path / "configs" / "gate_thresholds.json").read_text("utf-8"))
    assert frozen["thresholds"]["minilm"]["threshold"] == 0.8
    with pytest.raises(FileExistsError, match="frozen"):
        CALIBRATE.run(pool_id="pool-1", run_id="cal2", output_root=tmp_path / "cal", **paths(tmp_path))


def test_evaluation_scores_excludes_and_checks_gate_on_heldout(tmp_path):
    write_fixture(tmp_path)
    CALIBRATE.run(pool_id="pool-1", run_id="cal", output_root=tmp_path / "cal", **paths(tmp_path))
    summary = EVALUATE.run(pool_id="pool-1", run_id="rq3", output_root=tmp_path / "eval", **paths(tmp_path))

    # e3 (off-topic) has no relevant passage and is left out.
    assert summary["n_evaluated"] == 2 and summary["n_excluded"] == 1
    assert summary["excluded_by_type"]["off_topic"] == 1
    minilm, e5, biolord = (summary["methods"][m]["mean"] for m in METHODS)
    # e1: b1 relevant. MiniLM rank 1, E5 rank 2, BioLORD rank 2. e2: b5 at rank 2 for all.
    assert minilm["mrr@5"] == pytest.approx(0.75) and e5["mrr@5"] == pytest.approx(0.5)
    assert biolord["hit@1"] == 0.0 and minilm["hit@1"] == 0.5
    assert "minilm_vs_e5_base_v2" in summary["pairwise"]
    gate = summary["methods"]["minilm"]["gate_heldout"]
    # Held-out top-1: e1 b1 0.9 (relevant, accepted), e2 b4 0.7 (not relevant, rejected), e3 b6 0.85 off-topic accepted.
    assert gate["on_topic"]["tp"] == 1 and gate["on_topic"]["fp"] == 0
    assert (gate["off_topic_rejected"], gate["off_topic_total"]) == (0, 1)
    table = (tmp_path / "eval" / "rq3" / "rq3_table.md").read_text("utf-8")
    assert table.startswith("| Model |") and "MiniLM" in table and "1 of 3 left out" in table
    lines = (tmp_path / "eval" / "rq3" / "per_query.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 2
    with pytest.raises(FileExistsError, match="not empty"):
        EVALUATE.run(pool_id="pool-1", run_id="rq3", output_root=tmp_path / "eval", **paths(tmp_path))


def test_rankings_from_another_query_file_are_rejected(tmp_path):
    write_fixture(tmp_path)
    pool = tmp_path / "judging" / "pool-1" / "pool.json"
    values = json.loads(pool.read_text("utf-8"))
    values["stages"]["B"]["queries_sha256"] = "different"
    pool.write_text(json.dumps(values), "utf-8")
    with pytest.raises(ValueError, match="different query file"):
        EVALUATE.run(pool_id="pool-1", run_id="rq3", output_root=tmp_path / "eval", **paths(tmp_path))
