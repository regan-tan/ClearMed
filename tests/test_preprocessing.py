import pytest

from clearmed.data.preprocessing import (
    format_generation_input,
    format_generation_target,
    format_question,
    get_answer_label,
    get_answer_text,
    is_valid_generation_example,
    is_valid_qa_example,
    normalize_question_for_comparison,
)


@pytest.fixture
def row():
    return {"id": "example", "question": " What is asthma? ", "opa": " Option A ", "opb": "Option B", "opc": "Option C", "opd": "Option D", "cop": 2, "choice_type": "single", "exp": " Expert explanation. ", "subject_name": "Medicine", "topic_name": "Respiratory"}


def test_format_question_is_canonical_and_hides_gold(row):
    assert format_question(row) == "Question:\nWhat is asthma?\n\nOptions:\nA. Option A\nB. Option B\nC. Option C\nD. Option D"
    assert "Answer:" not in format_question(row)


def test_generation_formats(row):
    assert format_generation_input(row).startswith("Answer the following medical exam question and explain your answer.\n\nQuestion:")
    assert format_generation_target(row) == "Answer: C\nExplanation: Expert explanation."


@pytest.mark.parametrize("index,label", tuple(enumerate("ABCD")))
def test_answer_decoding(row, index, label):
    row["cop"] = index
    assert get_answer_label(row) == label
    assert get_answer_text(row) == f"Option {label}"


@pytest.mark.parametrize("value", [None, -1, 4, True, "2"])
def test_invalid_answers_are_rejected(row, value):
    row["cop"] = value
    with pytest.raises(ValueError):
        get_answer_label(row)
    assert not is_valid_qa_example(row)


def test_qa_validity_requires_question_and_options_but_not_explanation(row):
    assert is_valid_qa_example(row)
    row["exp"] = None
    assert is_valid_qa_example(row)
    for field, value in (("question", None), ("question", "  "), ("opa", None), ("opb", ""), ("opc", "  "), ("opd", None)):
        broken = row.copy()
        broken[field] = value
        assert not is_valid_qa_example(broken)


@pytest.mark.parametrize("explanation", [None, "", "   "])
def test_generation_validity_requires_explanation(row, explanation):
    row["exp"] = explanation
    assert not is_valid_generation_example(row)
    row["exp"] = "Present"
    row["question"] = ""
    assert not is_valid_generation_example(row)


def test_normalization_is_conservative():
    assert normalize_question_for_comparison("  What   Is ASTHMA?  ") == "what is asthma?"
    assert normalize_question_for_comparison("what is asthma") != normalize_question_for_comparison("what is asthma?")
