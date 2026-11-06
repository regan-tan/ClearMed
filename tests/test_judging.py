import csv
import io
import json
import sys
from collections import Counter
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from random import Random

import pytest

from clearmed.retrieval.judging import (
    LABELLERS,
    agreement,
    assign_item_ids,
    build_pool,
    final_judgments,
    pool_query,
    read_labels,
    sheet_rows,
    write_sheet,
)

METHODS = ("minilm", "e5_base_v2", "biolord_2023")


def load_script(name):
    spec = spec_from_file_location(f"{name}_for_test", Path(__file__).parents[1] / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXPORT = load_script("export_judging_pool")
IMPORT = load_script("import_labels")


def query(index, *, query_set="evaluation", follow_up_type="why_correct"):
    return {
        "query_id": f"{query_set}-{index:03d}",
        "query_set": query_set,
        "question_id": f"{query_set}-q{index}",
        "question": f"Question {index}?",
        "options": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
        "gold_label": "B",
        "gold_answer_text": "beta",
        "follow_up_type": follow_up_type,
        "follow_up": "Capital of France?" if follow_up_type == "off_topic" else "Why is that correct?",
    }


def results(doc_ids):
    return [{"rank": rank, "doc_id": doc_id, "score": 1 - rank / 10} for rank, doc_id in enumerate(doc_ids, start=1)]


def fake_rankings(records):
    """Methods return shifted, overlapping passage lists so pooling has duplicates to remove."""
    return {
        method: {
            record["query_id"]: results([f"d{(i + m + s) % 9}" for s in range(5)]) for i, record in enumerate(records)
        }
        for m, method in enumerate(METHODS)
    }


def test_pool_query_unions_and_records_who_retrieved_what():
    pooled = pool_query({"minilm": results(["x", "y"]), "e5_base_v2": results(["y", "z"])}, depth=2)
    assert pooled == {"x": {"minilm": 1}, "y": {"minilm": 2, "e5_base_v2": 1}, "z": {"e5_base_v2": 2}}
    assert set(pool_query({"minilm": results(["x", "y"]), "e5_base_v2": results(["y", "z"])}, depth=1)) == {"x", "y"}


def test_pool_splits_whole_queries_and_shares_only_on_topic_ones():
    records = [query(i, follow_up_type="off_topic" if i < 3 else "why_correct") for i in range(10)]
    entries = build_pool(
        stage="B", queries=records, rankings=fake_rankings(records), depth=5, shared_count=2, rng=Random(1)
    )

    owners = {}
    for entry in entries:
        owners.setdefault(entry["query_id"], set()).add(entry["labeller"])
    shared = {query_id for query_id, who in owners.items() if len(who) == 2}
    assert len(shared) == 2 and not shared & {"evaluation-000", "evaluation-001", "evaluation-002"}
    solo = Counter(next(iter(who)) for who in owners.values() if len(who) == 1)
    assert solo[LABELLERS[0]] == solo[LABELLERS[1]] == 4
    per_query = Counter((entry["labeller"], entry["query_id"]) for entry in entries)
    assert all(count == len(pool_query({m: fake_rankings(records)[m][q] for m in METHODS}, 5))
               for (_, q), count in per_query.items())
    assert all(entry["shared"] == (entry["query_id"] in shared) for entry in entries)


def test_pool_rejects_missing_rankings_and_too_many_shared():
    records = [query(i) for i in range(3)]
    rankings = fake_rankings(records)
    del rankings["minilm"]["evaluation-001"]
    with pytest.raises(ValueError, match="no rankings"):
        build_pool(stage="B", queries=records, rankings=rankings, depth=5, shared_count=0, rng=Random(1))
    with pytest.raises(ValueError, match="only 3 are on-topic"):
        build_pool(
            stage="B", queries=records, rankings=fake_rankings(records), depth=5, shared_count=4, rng=Random(1)
        )


def make_key_and_sheet(records, *, shared=2):
    entries = build_pool(
        stage="B", queries=records, rankings=fake_rankings(records), depth=5, shared_count=shared, rng=Random(3)
    )
    entries = assign_item_ids(entries, Random(3))
    passages = {f"d{i}": {"text": f"passage {i}"} for i in range(9)}
    rows = sheet_rows(
        entries,
        queries={record["query_id"]: record for record in records},
        passages=passages,
        explanations={record["question_id"]: "SECRET REFERENCE" for record in records},
    )
    return entries, rows


def test_sheet_is_blind_and_grouped_by_labeller_then_query():
    records = [query(i) for i in range(6)]
    entries, rows = make_key_and_sheet(records)
    text = write_sheet(rows)

    for method in METHODS:
        assert method not in text
    assert "rank" not in text.splitlines()[0] and "score" not in text.splitlines()[0]
    assert len({row["item_id"] for row in rows}) == len(rows)
    labellers = [row["labeller"] for row in rows]
    assert labellers == sorted(labellers)
    blocks = [(entry["labeller"], entry["query_id"]) for entry in entries]
    runs = [block for i, block in enumerate(blocks) if i == 0 or blocks[i - 1] != block]
    assert len(runs) == len(set(blocks))  # Each query's passages sit together.
    assert rows[0]["correct_answer"] == "B. beta" and rows[0]["question"].endswith("D. delta")


def fill(rows, value_for=lambda row: "1"):
    filled = [{**row, "relevant": value_for(row)} for row in rows]
    return write_sheet(filled)


def test_read_labels_accepts_complete_sheet_and_rejects_bad_ones():
    records = [query(i) for i in range(4)]
    key, rows = make_key_and_sheet(records, shared=1)
    assert set(read_labels(fill(rows), key).values()) == {1}
    assert read_labels("﻿" + fill(rows, lambda row: " 0 "), key)[rows[0]["item_id"]] == 0

    cases = {
        "relevant must be 0 or 1": fill(rows, lambda row: "" if row is rows[1] else "1"),
        "got 'yes'": fill(rows, lambda row: "yes" if row is rows[1] else "1"),
        "missing from the sheet": write_sheet([{**row, "relevant": "1"} for row in rows[1:]]),
        "unknown item_id": write_sheet([{**row, "relevant": "1", "item_id": row["item_id"] + "x"} for row in rows]),
        "duplicate item_id": write_sheet([{**row, "relevant": "1"} for row in rows + rows[:1]]),
        "labeller or stage changed": write_sheet(
            [{**row, "relevant": "1", "labeller": "someone" if row is rows[0] else row["labeller"]} for row in rows]
        ),
    }
    for message, text in cases.items():
        with pytest.raises(ValueError, match=message):
            read_labels(text, key)


def test_agreement_and_final_judgments_need_adjudication_for_conflicts():
    records = [query(i) for i in range(4)]
    key, _ = make_key_and_sheet(records, shared=2)
    shared_entries = [entry for entry in key if entry["shared"]]
    conflict = next(e for e in shared_entries if e["labeller"] == LABELLERS[0])
    labels = {entry["item_id"]: (0 if entry is conflict else 1) for entry in key}

    stats = agreement(key, labels)
    assert stats["shared_items"] == len(shared_entries) // 2
    assert len(stats["disagreements"]) == 1 and stats["shared_queries"] == 2
    assert stats["cohen_kappa"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="no agreed label"):
        final_judgments(key, labels)

    pair = (conflict["stage"], conflict["query_id"], conflict["doc_id"])
    judgments = final_judgments(key, labels, {pair: 1})
    assert len(judgments) == len({(e["query_id"], e["doc_id"]) for e in key})
    resolved = next(j for j in judgments if (j["stage"], j["query_id"], j["doc_id"]) == pair)
    assert resolved["relevant"] == 1 and resolved["source"] == "adjudicated"
    assert {j["source"] for j in judgments} >= {"both", "adjudicated"}
    assert agreement(key, dict.fromkeys(labels, 1))["cohen_kappa"] == 1.0


def write_fixture(root):
    """Frozen query files, matching ranking runs and a corpus run on disk."""
    records = {
        "A": [query(i, query_set="calibration") for i in range(6)],
        "B": [query(i, follow_up_type="off_topic" if i == 0 else "mechanism") for i in range(8)],
    }
    passages = [{"doc_id": f"d{i}", "text": f"passage {i}"} for i in range(9)]
    passage_body = "".join(json.dumps(p) + "\n" for p in passages)
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "passages.jsonl").write_text(passage_body, encoding="utf-8", newline="\n")
    passages_sha = sha256(passage_body.encode()).hexdigest()
    (corpus / "config.json").write_text(
        json.dumps({"run_id": "c1", "passages_file": "passages.jsonl", "passages_sha256": passages_sha}),
        encoding="utf-8",
    )
    query_files = {}
    for stage, items in records.items():
        path = root / f"queries_{stage}.jsonl"
        body = "".join(json.dumps(r) + "\n" for r in items)
        path.write_text(body, encoding="utf-8", newline="\n")
        query_files[stage] = path
        for method, by_query in fake_rankings(items).items():
            run_dir = root / "rankings" / method / f"run-{stage}"
            run_dir.mkdir(parents=True)
            (run_dir / "rankings.jsonl").write_text(
                "".join(json.dumps({"query_id": q, "results": r}) + "\n" for q, r in by_query.items()),
                encoding="utf-8",
            )
            config = {
                "queries_sha256": sha256(body.encode()).hexdigest(),
                "index": {"passages_sha256": passages_sha},
                "answer_source": "gold",
                "k": 5,
            }
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    explanations = {r["question_id"]: "ref" for items in records.values() for r in items}
    return query_files, explanations


def export(tmp_path, query_files, explanations, pool_id="pool-1"):
    return EXPORT.run(
        pool_id=pool_id,
        run_ids={"A": "run-A", "B": "run-B"},
        corpus_run=tmp_path / "corpus",
        explanations=explanations,
        query_files=query_files,
        rankings_root=tmp_path / "rankings",
        sheet_root=tmp_path / "sheets",
        key_root=tmp_path / "keys",
        shared={"A": 0, "B": 2},
    )


def test_export_then_import_round_trip_with_adjudication(tmp_path):
    query_files, explanations = write_fixture(tmp_path)
    summary = export(tmp_path, query_files, explanations)

    assert summary["unique_pairs_per_stage"]["A"] <= 6 * 3
    assert summary["shared_items"] > 0
    with pytest.raises(FileExistsError, match="not empty"):
        export(tmp_path, query_files, explanations)

    sheet = (tmp_path / "sheets" / "pool-1" / "labelling_sheet.csv").read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(sheet)))
    key = IMPORT.load_key(tmp_path / "keys" / "pool-1")
    first_shared = next(e["item_id"] for e in key if e["shared"] and e["labeller"] == LABELLERS[0])
    labels_dir = tmp_path / "labels" / "pool-1"
    labels_dir.mkdir(parents=True)
    (labels_dir / "labelling_sheet.csv").write_text(
        fill(rows, lambda row: "0" if row["item_id"] == first_shared else "1"), encoding="utf-8"
    )

    first = IMPORT.run(pool_id="pool-1", key_root=tmp_path / "keys", labels_root=tmp_path / "labels")
    assert first["status"] == "needs_adjudication" and first["unresolved"] == 1
    assert not (tmp_path / "keys" / "pool-1" / "judgments.jsonl").exists()
    template = list(csv.DictReader(io.StringIO((labels_dir / "disagreements.csv").read_text(encoding="utf-8-sig"))))
    assert len(template) == 1 and template[0]["agreed_label"] == ""

    template[0]["agreed_label"] = "1"
    with (labels_dir / "adjudication.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=template[0].keys())
        writer.writeheader()
        writer.writerows(template)
    final = IMPORT.run(pool_id="pool-1", key_root=tmp_path / "keys", labels_root=tmp_path / "labels")
    assert final["status"] == "complete" and final["adjudicated"] == 1
    judgments = (tmp_path / "keys" / "pool-1" / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(judgments) == sum(summary["unique_pairs_per_stage"].values())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        IMPORT.run(pool_id="pool-1", key_root=tmp_path / "keys", labels_root=tmp_path / "labels")


def test_export_rejects_rankings_from_other_query_files(tmp_path):
    query_files, explanations = write_fixture(tmp_path)
    query_files["B"].write_text(query_files["B"].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="different query file"):
        export(tmp_path, query_files, explanations)


def test_import_rejects_tampered_pool_key(tmp_path):
    query_files, explanations = write_fixture(tmp_path)
    export(tmp_path, query_files, explanations)
    key_path = tmp_path / "keys" / "pool-1" / "pool_key.jsonl"
    key_path.write_text(key_path.read_text(encoding="utf-8").replace('"relevant"', '"x"') + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        IMPORT.load_key(tmp_path / "keys" / "pool-1")


def test_each_follow_up_type_is_split_evenly_between_labellers():
    types = ["off_topic", "why_correct", "key_clue", "mechanism", "why_not_option"]
    records = [query(i, follow_up_type=types[i % 5]) for i in range(40)]
    entries = build_pool(stage="B", queries=records, rankings=fake_rankings(records), depth=1, shared_count=4,
                         rng=Random(7))
    solo = {(e["labeller"], e["query_id"], e["follow_up_type"]) for e in entries if not e["shared"]}
    counts = Counter((labeller, follow_up_type) for labeller, _, follow_up_type in solo)
    for follow_up_type in types:
        assert abs(counts[(LABELLERS[0], follow_up_type)] - counts[(LABELLERS[1], follow_up_type)]) <= 1
