import json
import sys
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from datasets import Dataset

from clearmed.data.splitting import membership_fingerprint
from clearmed.retrieval.queries import (
    OFF_TOPIC_FOLLOW_UPS,
    build_query_set,
    build_query_text,
    is_usable_query_mcq,
    mark_rq4_subset,
    query_text_for,
    select_demo_questions,
)

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "make_followup_queries.py"
SCRIPT_SPEC = spec_from_file_location("make_followup_queries_for_test", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
SCRIPT_MODULE = module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = SCRIPT_MODULE
SCRIPT_SPEC.loader.exec_module(SCRIPT_MODULE)


def mcq(index, *, prefix="q", cop=None, subject=None, question=None):
    return {
        "id": f"{prefix}-{index}",
        "question": question or f"{prefix} clinical question number {index}?",
        "opa": f"alpha {index}",
        "opb": f"beta {index}",
        "opc": f"gamma {index}",
        "opd": f"delta {index}",
        "cop": index % 4 if cop is None else cop,
        "exp": f"SECRET EXPLANATION {index}",
        "subject_name": subject or ("Pharmacology", "Surgery", "Anatomy")[index % 3],
    }


def dataset(count, **kwargs):
    return Dataset.from_list([mcq(index, **kwargs) for index in range(count)])


def record(**overrides):
    base = build_query_set(dataset(4), query_set="evaluation", source_split="validation", size=1)[0]
    return {**base, **overrides}


def test_query_text_puts_follow_up_first_and_keeps_all_options():
    text = build_query_text(record(follow_up="Why is this dangerous?"), "C")
    lines = text.splitlines()

    assert lines[0] == "Follow-up: Why is this dangerous?"
    assert lines[1].startswith("Answer: C. gamma")
    assert lines[2].startswith("Question: ")
    assert [line[:2] for line in lines[3:]] == ["A.", "B.", "C.", "D."]
    assert "SECRET EXPLANATION" not in text


def test_answer_source_rules_follow_agents_md():
    item = record(gold_label="B")

    assert "Answer: B." in query_text_for(item, stage="rq3", answer_source="gold")
    assert "Answer: B." in query_text_for(item, stage="calibration", answer_source="gold")
    assert "Answer: D." in query_text_for(item, stage="rq4", answer_source="predicted", predicted_label="D")
    for stage in ("rq4", "demo"):
        with pytest.raises(ValueError, match="may not use the 'gold' answer"):
            query_text_for(item, stage=stage, answer_source="gold")
    with pytest.raises(ValueError, match="Unknown stage"):
        query_text_for(item, stage="training", answer_source="gold")
    with pytest.raises(ValueError, match="predicted answer label is required"):
        query_text_for(item, stage="rq4", answer_source="predicted")
    with pytest.raises(ValueError, match="must be one of"):
        query_text_for(item, stage="rq4", answer_source="predicted", predicted_label="MALFORMED")


def test_query_set_is_deterministic_sized_and_typed():
    rows = dataset(200)
    first = build_query_set(rows, query_set="evaluation", source_split="validation", size=60)
    second = build_query_set(rows, query_set="evaluation", source_split="validation", size=60)

    assert first == second
    assert len({item["question_id"] for item in first}) == 60
    assert [item["query_id"] for item in first][:2] == ["evaluation-001", "evaluation-002"]
    counts = Counter(item["follow_up_type"] for item in first)
    assert counts["off_topic"] == 9
    assert sorted(counts[name] for name in ("why_correct", "why_not_option", "key_clue", "mechanism")) == [
        12,
        13,
        13,
        13,
    ]
    assert all(item["follow_up"] in OFF_TOPIC_FOLLOW_UPS for item in first if item["follow_up_type"] == "off_topic")
    assert all(item["option_asked"] in "ABCD" for item in first if item["follow_up_type"] == "why_not_option")


def test_follow_ups_never_depend_on_the_gold_answer():
    as_given = build_query_set(dataset(50), query_set="evaluation", source_split="validation", size=40)
    all_a = build_query_set(dataset(50, cop=0), query_set="evaluation", source_split="validation", size=40)

    def visible(items):
        return [(item["question_id"], item["follow_up"], item["option_asked"]) for item in items]

    assert visible(as_given) == visible(all_a)


def test_query_set_skips_invalid_rows_and_rejects_oversized_requests():
    rows = dataset(5).to_list()
    rows[0]["opa"] = ""
    built = build_query_set(Dataset.from_list(rows), query_set="calibration", source_split="internal_dev", size=4)
    assert "q-0" not in {item["question_id"] for item in built}
    with pytest.raises(ValueError, match="only 4 usable"):
        build_query_set(Dataset.from_list(rows), query_set="calibration", source_split="internal_dev", size=5)


def test_ambiguous_and_image_questions_are_never_selected():
    rows = dataset(8).to_list()
    rows[0].update(opc="Indapamide", opd="indapamide.")
    rows[1].update(**dict.fromkeys(("opa", "opb", "opc", "opd"), "| TIBC, | Ferritin"))
    rows[2]["question"] = "Process shown in color plate is:"
    rows[3]["question"] = "Identify the structure marked in the image"
    rows[4]["question"] = "Which drug has been shown to reduce mortality?"
    rows[5].update(opa="Gram +ve", opb="Gram -ve", opc="p<0.05", opd="p>0.05")
    bad = {"q-0", "q-1", "q-2", "q-3"}

    assert not is_usable_query_mcq(rows[0]) and not is_usable_query_mcq(rows[1])
    assert is_usable_query_mcq(rows[4]) and is_usable_query_mcq(rows[5])
    built = build_query_set(Dataset.from_list(rows), query_set="evaluation", source_split="validation", size=4)
    assert not bad & {item["question_id"] for item in built}
    assert not bad & set(select_demo_questions(Dataset.from_list(rows), exclude_ids=set(), size=4))


def test_rq4_subset_flags_and_second_turns_are_on_topic():
    records = build_query_set(dataset(100), query_set="evaluation", source_split="validation", size=60)
    marked = mark_rq4_subset(records, size=30, two_turn=10)

    assert sum(item["rq4"] for item in marked) == 30
    second = [item for item in marked if item["second_turn"]]
    assert len(second) == 10
    assert all(item["rq4"] and item["follow_up_type"] != "off_topic" for item in second)
    assert marked == mark_rq4_subset(records, size=30, two_turn=10)
    assert not any(item["rq4"] for item in records)


def test_demo_questions_exclude_ids_and_rotate_subjects():
    rows = dataset(30)
    chosen = select_demo_questions(rows, exclude_ids={"q-0", "q-1"}, size=6)

    assert len(chosen) == 6 and len(set(chosen)) == 6
    assert not {"q-0", "q-1"} & set(chosen)
    subjects = Counter(rows[int(item.split("-")[1])]["subject_name"] for item in chosen)
    assert set(subjects.values()) == {2}
    with pytest.raises(ValueError, match="eligible"):
        select_demo_questions(rows, exclude_ids=set(), size=31)


def write_splits(splits_dir):
    partitions = {
        "internal_dev": dataset(40, prefix="dev"),
        "part1_train": dataset(10, prefix="train", question="Shared training question?"),
        "retrieval_kb": dataset(10, prefix="kb"),
    }
    manifest = {"partitions": {}}
    for name, rows in partitions.items():
        rows.save_to_disk(str(splits_dir / name))
        manifest["partitions"][name] = {"rows": len(rows), "membership_fingerprint": membership_fingerprint(rows)}
    (splits_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_script_writes_frozen_sets_and_excludes_duplicates(tmp_path):
    write_splits(tmp_path / "splits")
    validation_rows = [mcq(index, prefix="val") for index in range(40)]
    validation_rows[0]["question"] = "Shared training question?"
    sizes = {"calibration": 8, "evaluation": 20, "rq4": 10, "two_turn": 4, "demo": 10}

    summary = SCRIPT_MODULE.run(
        splits_dir=tmp_path / "splits",
        validation=Dataset.from_list(validation_rows),
        query_dir=tmp_path / "queries",
        demo_path=tmp_path / "demo.json",
        sizes=sizes,
    )

    evaluation = [
        json.loads(line) for line in (tmp_path / "queries" / "queries_eval.jsonl").read_text("utf-8").splitlines()
    ]
    demo = json.loads((tmp_path / "demo.json").read_text(encoding="utf-8"))["question_ids"]
    assert summary["validation_duplicates_excluded"] == 1
    assert "val-0" not in {item["question_id"] for item in evaluation} | set(demo)
    assert not {item["question_id"] for item in evaluation} & set(demo)
    assert summary["rq4_queries"] == 10 and summary["rq4_second_turns"] == 4
    assert len(demo) == 10
    with pytest.raises(FileExistsError, match="frozen"):
        SCRIPT_MODULE.run(
            splits_dir=tmp_path / "splits",
            validation=Dataset.from_list(validation_rows),
            query_dir=tmp_path / "queries",
            demo_path=tmp_path / "demo.json",
            sizes=sizes,
        )
