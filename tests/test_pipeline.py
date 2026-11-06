import pytest
import torch
from test_embedders import FakeEncoder

from clearmed.conversation.pipeline import (
    FALLBACK_MESSAGE,
    DenseRetriever,
    FlanT5Generator,
    StubGenerator,
    TutorPipeline,
    build_prompt,
    strip_answer_prefixes,
)
from clearmed.retrieval.embedders import Embedder, load_embedding_config
from clearmed.retrieval.gate import Gate
from clearmed.retrieval.index import DenseIndex

RECORD = {
    "query_id": "calibration-001",
    "question_id": "q1",
    "follow_up": "Why is that the correct answer?",
    "question": "Which drug should be avoided in asthma?",
    "options": {"A": "vitamin", "B": "losartan", "C": "propranolol", "D": "thiazide"},
    "gold_label": "C",
    "gold_answer_text": "propranolol",
    "exp": "SECRET-REFERENCE-EXPLANATION",
    "second_turn": None,
}


class FakeRetriever:
    def __init__(self, scores):
        self.scores = scores
        self.queries = []

    def retrieve(self, query_text, k):
        self.queries.append(query_text)
        return [
            {"rank": rank, "doc_id": f"d{rank}", "score": score, "text": f"passage {rank} " + "word " * 20}
            for rank, score in enumerate(self.scores[:k], start=1)
        ]


def pipeline(scores=(0.9, 0.8, 0.7, 0.6, 0.5), threshold=0.75, generator=None):
    generator = generator or StubGenerator()
    return TutorPipeline(
        generator, stage="calibration", retriever=FakeRetriever(list(scores)),
        gate=Gate("minilm", threshold), method_id="minilm",
    ), generator


def test_fallback_never_calls_the_generator():
    tutor, generator = pipeline(scores=(0.70, 0.60), threshold=0.75)
    result = tutor.respond(RECORD, predicted_label="B")
    assert result["fallback"] and result["reply"] == FALLBACK_MESSAGE
    assert result["gate_accepted"] is False and result["top1_score"] == 0.70
    assert generator.prompts == [] and result["prompt"] is None


def test_accepted_reply_uses_at_most_three_passages_and_logs_everything():
    tutor, _ = pipeline()
    result = tutor.respond(RECORD, predicted_label="B")
    assert not result["fallback"] and result["passage_ids"] == ["d1", "d2", "d3"]
    assert len(result["retrieved"]) == 5 and result["threshold"] == 0.75
    assert result["prompt"].count("\n[") == 3 and "[4]" not in result["prompt"]
    assert result["reply"].startswith("Answer: B\nExplanation:")
    assert result["prompt"].endswith("Student follow-up: Why is that the correct answer?")


def test_predicted_mode_never_sees_gold_answer_or_reference_explanation():
    tutor, _ = pipeline()
    result = tutor.respond(RECORD, predicted_label="B")
    query = tutor.retriever.queries[0]
    assert "Answer: B. losartan" in query and "Answer: C" not in query
    for text in (query, result["prompt"]):
        assert "SECRET" not in text
    # The prompt carries the four options but no answer line for the MCQ.
    assert "Answer:" not in result["prompt"].split("Evidence:")[0]


def test_rq4_stage_refuses_gold_answers_but_runs_predicted():
    tutor = TutorPipeline(StubGenerator(), stage="rq4", retriever=FakeRetriever([0.9]),
                          gate=Gate("minilm", 0.5), method_id="minilm")
    assert not tutor.respond(RECORD, predicted_label="A")["fallback"]
    with pytest.raises(ValueError, match="Predicted label"):
        tutor.respond(RECORD, predicted_label="MALFORMED")


def test_follow_up_and_mcq_survive_trimming():
    generator = StubGenerator(max_input_tokens=50)
    tutor, _ = pipeline(generator=generator)
    result = tutor.respond(RECORD, predicted_label="B")
    prompt = result["prompt"]
    assert generator.count_tokens(prompt) <= 50
    assert prompt.endswith("Student follow-up: Why is that the correct answer?")
    assert all(f"{label}. {text}" in prompt for label, text in RECORD["options"].items())
    assert result["passages_trimmed"] == 1 and result["passage_ids"] == ["d1"]
    assert "[1] passage 1 word" in prompt and "[2]" not in prompt


def test_prompt_too_small_for_mcq_is_rejected():
    with pytest.raises(ValueError, match="exceed"):
        build_prompt(RECORD, "Why?", [], count_tokens=lambda t: len(t.split()), max_tokens=5)


def test_no_retrieval_mode_always_generates_without_evidence():
    generator = StubGenerator()
    tutor = TutorPipeline(generator, stage="rq4")
    result = tutor.respond(RECORD, predicted_label="D")
    assert not result["fallback"] and result["passage_ids"] == [] and result["gate_accepted"] is None
    assert "Evidence:" not in result["prompt"] and result["query_text"] is None
    with pytest.raises(ValueError, match="needs a gate"):
        TutorPipeline(generator, stage="rq4", retriever=FakeRetriever([0.9]))
    with pytest.raises(ValueError, match="Gate is for"):
        TutorPipeline(generator, stage="rq4", retriever=FakeRetriever([0.9]), gate=Gate("e5_base_v2", 0.5),
                      method_id="minilm")


def test_second_turn_carries_the_previous_turn_and_is_deterministic():
    record = {**RECORD, "second_turn": "Can you summarise the key point in one sentence?"}
    first_run = pipeline()[0].converse(record, predicted_label="B")
    second_run = pipeline()[0].converse(record, predicted_label="B")
    assert first_run == second_run and [t["turn"] for t in first_run] == [1, 2]
    second = first_run[1]
    assert "Previous turn:\nStudent: Why is that the correct answer?\nTutor: Answer: B" in second["prompt"]
    assert second["follow_up"] == record["second_turn"]


def test_dense_retriever_returns_passage_texts():
    config = load_embedding_config("minilm")
    embedder = Embedder(config, FakeEncoder())
    passages = [{"doc_id": "beta", "text": "propranolol beta wheezing asthma"}, {"doc_id": "vit", "text": "vitamin"}]
    vectors = embedder.encode_documents([p["text"] for p in passages])
    description = embedder.describe(dimension=vectors.shape[1])
    index = DenseIndex(vectors, ["beta", "vit"],
                       {k: description[k] for k in ("method_id", "model_name", "revision", "dimension")})
    hits = DenseRetriever(embedder, index, passages).retrieve("propranolol asthma", k=2)
    assert [h["rank"] for h in hits] == [1, 2] and {h["text"] for h in hits} == {p["text"] for p in passages}
    with pytest.raises(ValueError, match="no text"):
        DenseRetriever(embedder, index, passages[:1])


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=True, truncation=False, return_tensors=None):
        ids = [len(word) for word in text.split()] + ([1] if add_special_tokens else [])
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{i}" for i in ids)


class FakeModel:
    class config:
        decoder_start_token_id = 0

    def eval(self):
        return self

    def generate(self, *, decoder_input_ids, **kwargs):
        self.decoder_input_ids, self.kwargs = decoder_input_ids, kwargs
        assert kwargs["do_sample"] is False
        return torch.cat([decoder_input_ids, torch.tensor([[7, 8]])], dim=1)


def test_flan_t5_generator_prefills_the_predicted_answer():
    model = FakeModel()
    generator = FlanT5Generator(model, FakeTokenizer(), max_input_tokens=50, device="cpu")
    reply = generator.generate("some prompt text", answer_label="C")
    # Pre-fill = decoder start + tokens of "Answer: C\nExplanation:" (word lengths 7, 1, 12).
    assert model.decoder_input_ids.tolist() == [[0, 7, 1, 12]]
    assert reply == "Answer: C\nExplanation: t7 t8"
    with pytest.raises(ValueError, match="input limit"):
        FlanT5Generator(model, FakeTokenizer(), max_input_tokens=2, device="cpu").generate("a b c", answer_label="C")


def test_generator_passes_loop_and_banned_phrase_controls_only_when_set():
    model = FakeModel()
    FlanT5Generator(model, FakeTokenizer(), max_input_tokens=50, device="cpu").generate("p", answer_label="A")
    assert "no_repeat_ngram_size" not in model.kwargs and "bad_words_ids" not in model.kwargs
    generator = FlanT5Generator(model, FakeTokenizer(), max_input_tokens=50, device="cpu",
                                no_repeat_ngram_size=3, banned_phrases=["Ans"])
    generator.generate("p", answer_label="A")
    assert model.kwargs["no_repeat_ngram_size"] == 3 and model.kwargs["bad_words_ids"] == [[3]]
    assert generator.settings["banned_phrases"] == ["Ans"]


@pytest.mark.parametrize("explanation, expected", [
    ("Ans. is 'a' i.e., Clozapine o Antipsychotics", "Clozapine o Antipsychotics"),
    ("Ans. (c) ASOM (Ref. Scott Brown)", "ASOM (Ref. Scott Brown)"),
    ("(c) ASOM(Ref. Scott Brown)", "ASOM(Ref. Scott Brown)"),
    ("A i.e. Fluoxetine", "Fluoxetine"),
    ("C. Rotor syndrome(Ref: Nelson)", "Rotor syndrome(Ref: Nelson)"),
    ("Ans. Dhatura", "Dhatura"),
    ("A 40-year-old woman has", "A 40-year-old woman has"),
    ("Answer to this is cardiac output", "Answer to this is cardiac output"),
    ("Gilbert syndrome causes", "Gilbert syndrome causes"),
])
def test_strip_answer_prefixes(explanation, expected):
    passage = f"Question: Q? Answer: X Explanation: {explanation}"
    assert strip_answer_prefixes(passage) == f"Question: Q? Answer: X Explanation: {expected}"


def test_clean_evidence_changes_only_the_prompt_passages():
    passages = [{"doc_id": "d1", "text": "Question: Q? Answer: Clozapine Explanation: Ans. is 'c' i.e., Clozapine"}]
    plain = build_prompt(RECORD, "Why?", passages, count_tokens=lambda t: len(t.split()), max_tokens=500)
    clean = build_prompt(RECORD, "Why?", passages, count_tokens=lambda t: len(t.split()), max_tokens=500,
                         clean_evidence=True)
    assert "Ans. is 'c'" in plain.text and "Ans." not in clean.text
    assert "Explanation: Clozapine" in clean.text and clean.passage_ids == plain.passage_ids


class FakePeftModel(FakeModel):
    def __init__(self):
        self.adapter_on, self.adapter_during_generate = True, []

    def disable_adapter(self):
        model = self

        class Off:
            def __enter__(self):
                model.adapter_on = False

            def __exit__(self, *exc):
                model.adapter_on = True

        return Off()

    def generate(self, **kwargs):
        self.adapter_during_generate.append(self.adapter_on)
        return super().generate(**kwargs)


def test_reply_without_adapter_switches_lora_off_only_while_generating():
    model = FakePeftModel()
    FlanT5Generator(model, FakeTokenizer(), max_input_tokens=50, device="cpu", use_adapter=False).generate(
        "p", answer_label="A")
    FlanT5Generator(model, FakeTokenizer(), max_input_tokens=50, device="cpu").generate("p", answer_label="A")
    assert model.adapter_during_generate == [False, True] and model.adapter_on
    with pytest.raises(ValueError, match="PEFT model"):
        FlanT5Generator(FakeModel(), FakeTokenizer(), max_input_tokens=50, device="cpu", use_adapter=False)


def test_stated_answer_line_uses_the_predicted_answer_only():
    tutor = TutorPipeline(StubGenerator(), stage="rq4", retriever=FakeRetriever([0.9]), gate=Gate("minilm", 0.5),
                          method_id="minilm", state_answer=True)
    result = tutor.respond(RECORD, predicted_label="B")
    assert "\n\nTutor's answer: B. losartan\n\nEvidence:" in result["prompt"]
    assert "propranolol\n\nTutor's answer" not in result["prompt"] and result["state_answer"] is True
    assert "Tutor's answer" not in pipeline()[0].respond(RECORD, predicted_label="B")["prompt"]
    with pytest.raises(ValueError, match="Stated answer"):
        build_prompt(RECORD, "Why?", [], count_tokens=len, max_tokens=999, stated_answer="E")
