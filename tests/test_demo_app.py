import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_pipeline import FakeRetriever

from clearmed.app.demo import (
    GATE_OFF_NOTE,
    UI_MODELS,
    ChatTurn,
    TutorService,
    create_app,
    previous_turn,
    question_from_row,
)
from clearmed.conversation.pipeline import (
    FALLBACK_MESSAGE,
    StubGenerator,
    TutorPipeline,
)
from clearmed.retrieval.gate import Gate

SPEC = spec_from_file_location("serve_demo_for_test", Path(__file__).parents[1] / "scripts" / "serve_demo.py")
assert SPEC is not None and SPEC.loader is not None
SERVE = module_from_spec(SPEC)
sys.modules[SPEC.name] = SERVE
SPEC.loader.exec_module(SERVE)

ROW = {"id": "v1", "question": "Which drug should be avoided in asthma?", "opa": "vitamin", "opb": "losartan",
       "opc": "propranolol", "opd": "thiazide", "cop": 2, "exp": "SECRET-REFERENCE-EXPLANATION",
       "subject_name": "Pharmacology", "topic_name": "Autonomic"}
PASSAGES = {f"d{rank}": {"text": f"Question: Q{rank}? Answer: X Explanation: Ans. (c) passage {rank} text",
                         "subject": "Pharmacology"} for rank in range(1, 6)}


def shown(rank):
    return f"Question: Q{rank}? Answer: X Explanation: passage {rank} text"


def service(*, scores=(0.9, 0.8, 0.7, 0.6, 0.5), threshold=0.75, gate_off=False, predicted="C"):
    calls = []

    def predict(row):
        calls.append(row)
        return {"predicted_answer": predicted, "generated_explanation": "LoRA explanation."}

    generator = StubGenerator()
    tutors = {}
    for method_id in UI_MODELS.values():
        retriever = FakeRetriever(list(scores))
        if gate_off:
            tutors[method_id] = TutorPipeline(generator, stage="demo", retriever=retriever, gate_off=True,
                                              method_id=f"{method_id}_gate_off", clean_evidence=True)
        else:
            tutors[method_id] = TutorPipeline(generator, stage="demo", retriever=retriever,
                                              gate=Gate(method_id, threshold), method_id=method_id, clean_evidence=True)
    svc = TutorService([question_from_row(ROW)], predict=predict, tutors=tutors, passages=PASSAGES, gate_off=gate_off)
    return svc, calls, generator


def client(**kwargs):
    svc, calls, generator = service(**kwargs)
    return TestClient(create_app(svc)), calls, generator


def follow_up(**overrides):
    return {"question_id": "v1", "history": [], "message": "Why is that the correct answer?",
            "embedding_model": "minilm", "top_k": 3, **overrides}


@pytest.mark.parametrize("gate_off", [False, True])
def test_health_reports_loaded_service_without_running_inference(gate_off):
    api, calls, generator = client(gate_off=gate_off)
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "question_count": 1,
                               "embedding_models": ["minilm", "e5", "biolord"],
                               "gate_enabled": not gate_off}
    assert calls == [] and generator.prompts == []


def test_build_service_checks_artifacts_before_loading_models(monkeypatch):
    def missing(args):
        raise ValueError("Missing local artifacts")

    monkeypatch.setattr(SERVE, "check_artifacts", missing)
    with pytest.raises(ValueError, match="Missing local artifacts"):
        SERVE.build_service(None)


@pytest.mark.parametrize("fails", [False, True])
def test_check_artifacts_cli_exits_without_building_service(monkeypatch, capsys, fails):
    def check(args):
        if fails:
            raise ValueError("Missing local artifacts")

    def unexpected(args):
        pytest.fail("Check-only mode must not load models")

    monkeypatch.setattr(SERVE, "check_artifacts", check)
    monkeypatch.setattr(SERVE, "build_service", unexpected)
    monkeypatch.setattr(sys, "argv", ["serve_demo.py", "--adapter-dir", "unused", "--check-artifacts"])
    if fails:
        with pytest.raises(SystemExit) as error:
            SERVE.main()
        assert error.value.code == 2
        assert "Missing local artifacts" in capsys.readouterr().err
    else:
        SERVE.main()
        assert "have not been checked" in capsys.readouterr().out


def test_questions_match_the_frontend_contract_without_the_reference_explanation():
    api, _, _ = client()
    [question] = api.get("/api/questions").json()
    assert question == {"id": "v1", "question": ROW["question"],
                        "options": {"A": "vitamin", "B": "losartan", "C": "propranolol", "D": "thiazide"},
                        "correct": "C", "subject": "Pharmacology", "topic": "Autonomic"}
    assert "SECRET" not in json.dumps(question)


def test_predict_sends_only_question_and_options_and_is_cached():
    api, calls, _ = client()
    first = api.post("/api/questions/v1/predict").json()
    second = api.post("/api/questions/v1/predict").json()
    assert first == second == {"question_id": "v1", "predicted": "C", "explanation": "LoRA explanation."}
    assert len(calls) == 1 and set(calls[0]) == {"id", "question", "opa", "opb", "opc", "opd"}


def test_malformed_prediction_and_unknown_question_are_errors_not_guesses():
    api, _, _ = client(predicted=None)
    assert api.post("/api/questions/v1/predict").status_code == 502
    assert api.post("/api/follow-up", json=follow_up()).status_code == 502
    assert api.post("/api/questions/nope/predict").status_code == 404
    assert api.post("/api/follow-up", json=follow_up(question_id="nope")).status_code == 404


def test_follow_up_uses_the_predicted_answer_and_returns_top_k_evidence():
    api, _, generator = client(predicted="B")
    body = api.post("/api/follow-up", json=follow_up(top_k=5)).json()
    assert body["fallback"] is False and body["reply"].startswith("Answer: B")
    assert [e["id"] for e in body["evidence"]] == ["d1", "d2", "d3", "d4", "d5"]
    assert body["evidence"][0] == {"id": "d1", "text": shown(1), "subject": "Pharmacology", "score": 0.9}
    [prompt] = generator.prompts
    assert "propranolol" in prompt and "SECRET" not in prompt and prompt.count("\n[") == 3  # Frozen: 3 passages.
    one = api.post("/api/follow-up", json=follow_up(top_k=1)).json()
    assert len(one["evidence"]) == 1


def test_low_top1_score_gives_the_fixed_fallback_with_the_weak_evidence():
    api, _, generator = client(scores=(0.5, 0.4), threshold=0.75)
    body = api.post("/api/follow-up", json=follow_up()).json()
    assert body == {"reply": FALLBACK_MESSAGE, "fallback": True, "evidence": [
        {"id": "d1", "text": shown(1), "subject": "Pharmacology", "score": 0.5},
        {"id": "d2", "text": shown(2), "subject": "Pharmacology", "score": 0.4}]}
    assert generator.prompts == []


def test_gate_off_answers_and_labels_the_reply():
    api, _, _ = client(scores=(0.5, 0.4), gate_off=True)
    body = api.post("/api/follow-up", json=follow_up()).json()
    assert body["fallback"] is False and body["reply"].endswith(GATE_OFF_NOTE)


def test_each_embedding_model_routes_to_its_own_tutor():
    svc, _, _ = service()
    api = TestClient(create_app(svc))
    api.post("/api/follow-up", json=follow_up(embedding_model="biolord"))
    assert len(svc.tutors["biolord_2023"].retriever.queries) == 1
    assert svc.tutors["minilm"].retriever.queries == [] and svc.tutors["e5_base_v2"].retriever.queries == []


def test_history_supplies_the_previous_turn():
    history = [ChatTurn(role="user", content="Why?"), ChatTurn(role="assistant", content="Because.")]
    assert previous_turn(history) == ("Why?", "Because.")
    assert previous_turn([ChatTurn(role="user", content="Why?")]) is None
    api, _, generator = client()
    api.post("/api/follow-up", json=follow_up(history=[{"role": "user", "content": "Why?"},
                                                      {"role": "assistant", "content": "Because."}]))
    assert "Because." in generator.prompts[0]


@pytest.mark.parametrize("bad", [{"top_k": 0}, {"top_k": 6}, {"embedding_model": "gpt"}, {"message": ""}])
def test_invalid_requests_are_rejected(bad):
    api, _, _ = client()
    assert api.post("/api/follow-up", json=follow_up(**bad)).status_code == 422


def test_service_rejects_missing_tutors_or_mixed_gate_settings():
    svc, _, _ = service()
    with pytest.raises(ValueError, match="Missing tutors"):
        TutorService([], predict=lambda row: {}, tutors={"minilm": svc.tutors["minilm"]}, passages={})
    with pytest.raises(ValueError, match="gate setting"):
        TutorService([], predict=lambda row: {}, tutors=svc.tutors, passages={}, gate_off=True)


def test_demo_questions_load_in_config_order_and_require_every_id(tmp_path):
    config = tmp_path / "demo.json"
    config.write_text(json.dumps({"question_ids": ["v2", "v1"]}), "utf-8")
    rows = [ROW, {**ROW, "id": "v2", "cop": 0}]
    assert [q["id"] for q in SERVE.load_demo_questions(rows, config)] == ["v2", "v1"]
    with pytest.raises(ValueError, match="not in the validation split"):
        SERVE.load_demo_questions([ROW], config)
