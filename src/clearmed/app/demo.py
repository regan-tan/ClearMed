"""Live demo backend for the Practice MCQ mode of the frontend (Plan 8).

Implements the contract in ``frontend/src/api/types.ts``:

* ``GET /api/questions``: the frozen demo questions (``configs/demo/demo_questions.json``);
* ``GET /api/models``: the answer/reply models the server can use (the first is the default);
* ``POST /api/questions/{id}/predict``: the MCQ answer and explanation (Part I FLAN-T5 + LoRA by default);
* ``POST /api/follow-up``: the Part II tutor, with the student's chosen embedding model.

Extra models are added with ``TutorService.add_model``; requests pick one with ``llm`` and
fall back to the default when it is omitted.

The tutor is the frozen RQ4 system: the LoRA model's *predicted* answer goes into
the retrieval query, the top-1 score passes the frozen gate (unless the server
runs with the gate off), and plain FLAN-T5 writes the reply from at most three
passages. The correct answer is sent to the UI so it can reveal it after the
student checks their answer; it never reaches either model.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clearmed.conversation.pipeline import TutorPipeline, mcq_row, strip_answer_prefixes
from clearmed.data.preprocessing import OPTION_LABELS

# Frontend model names -> standard method identifiers.
UI_MODELS: dict[str, str] = {"minilm": "minilm", "e5": "e5_base_v2", "biolord": "biolord_2023"}
GATE_OFF_NOTE = "\n\n_Gate off: this reply was written without the evidence confidence check._"

OptionKey = Literal["A", "B", "C", "D"]
DEFAULT_MODEL = "flan_t5_lora"
DEFAULT_MODEL_LABEL = "FLAN-T5 + LoRA (Part I)"


class UnknownModelError(LookupError):
    """The requested answer/reply model is not registered on this server."""


class PassageLookupError(RuntimeError):
    """A retrieved doc_id is missing from the loaded corpus (index/corpus mismatch)."""


class Question(BaseModel):
    id: str
    question: str
    options: dict[OptionKey, str]
    correct: OptionKey
    subject: str
    topic: str | None = None


class McqPrediction(BaseModel):
    question_id: str
    predicted: OptionKey
    explanation: str


class ModelInfo(BaseModel):
    id: str
    label: str


class Evidence(BaseModel):
    id: str
    text: str
    subject: str
    score: float


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class FollowUpRequest(BaseModel):
    question_id: str
    history: list[ChatTurn] = Field(default_factory=list)
    message: str = Field(min_length=1, max_length=2000)
    embedding_model: Literal["minilm", "e5", "biolord"]
    top_k: int = Field(ge=1, le=5)
    llm: str | None = None


class FollowUpResponse(BaseModel):
    reply: str
    evidence: list[Evidence]
    fallback: bool


def question_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """A MedMCQA row in the frontend's ``Question`` shape."""
    return {
        "id": str(row["id"]),
        "question": str(row["question"]),
        "options": {label: str(row[f"op{label.lower()}"]) for label in OPTION_LABELS},
        "correct": OPTION_LABELS[int(row["cop"])],
        "subject": str(row.get("subject_name") or "Unknown"),
        "topic": str(row["topic_name"]) if row.get("topic_name") else None,
    }


def previous_turn(history: Sequence[ChatTurn]) -> tuple[str, str] | None:
    """The last complete (student, tutor) exchange, which the prompt includes."""
    for index in range(len(history) - 1, 0, -1):
        if history[index].role == "assistant" and history[index - 1].role == "user":
            return history[index - 1].content, history[index].content
    return None


class TutorService:
    """Demo logic, independent of HTTP so it can be tested with fake models.

    ``predict`` maps model-input rows (question and options only) to
    ``{"predicted_answer", "generated_explanation"}``. ``tutors`` holds one
    retrieval tutor per method identifier. ``passages`` maps doc_id to the
    passage fields shown as evidence.
    """

    def __init__(
        self,
        questions: Sequence[Mapping[str, Any]],
        *,
        predict: Callable[[dict[str, Any]], Mapping[str, Any]],
        tutors: Mapping[str, TutorPipeline],
        passages: Mapping[str, Mapping[str, Any]],
        gate_off: bool = False,
    ) -> None:
        missing = sorted(set(UI_MODELS.values()) - set(tutors))
        if missing:
            raise ValueError(f"Missing tutors for {missing}.")
        if any(tutor.gate_off != gate_off for tutor in tutors.values()):
            raise ValueError("Every tutor must match the service's gate setting.")
        self.questions = {q["id"]: dict(q) for q in questions}
        self.tutors, self.passages, self.gate_off = tutors, passages, gate_off
        self._models: dict[str, tuple[str, Callable[[dict[str, Any]], Mapping[str, Any]], Mapping[str, TutorPipeline]]] = {
            DEFAULT_MODEL: (DEFAULT_MODEL_LABEL, predict, tutors)
        }
        self._predictions: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = threading.Lock()  # One model call at a time: torch on CPU is not shared safely.

    def add_model(
        self,
        model_id: str,
        label: str,
        *,
        predict: Callable[[dict[str, Any]], Mapping[str, Any]],
        tutors: Mapping[str, TutorPipeline],
    ) -> None:
        """Register another answer/reply model, with one retrieval tutor per method identifier."""
        if model_id in self._models:
            raise ValueError(f"Model {model_id!r} is already registered.")
        missing = sorted(set(UI_MODELS.values()) - set(tutors))
        if missing:
            raise ValueError(f"Missing tutors for {missing}.")
        if any(tutor.gate_off != self.gate_off for tutor in tutors.values()):
            raise ValueError("Every tutor must match the service's gate setting.")
        self._models[model_id] = (label, predict, tutors)

    def list_models(self) -> list[dict[str, str]]:
        return [{"id": model_id, "label": label} for model_id, (label, _, _) in self._models.items()]

    def _model(self, llm: str | None) -> tuple[str, Callable[[dict[str, Any]], Mapping[str, Any]], Mapping[str, TutorPipeline]]:
        model_id = llm or DEFAULT_MODEL
        if model_id not in self._models:
            raise UnknownModelError(model_id)
        _, predict, tutors = self._models[model_id]
        return model_id, predict, tutors

    def _question(self, question_id: str) -> dict[str, Any]:
        if question_id not in self.questions:
            raise KeyError(question_id)
        return self.questions[question_id]

    def list_questions(self) -> list[dict[str, Any]]:
        return list(self.questions.values())

    def predict(self, question_id: str, llm: str | None = None) -> dict[str, Any]:
        question = self._question(question_id)
        model_id, predict, _ = self._model(llm)
        with self._lock:
            if (model_id, question_id) not in self._predictions:
                # Only the question and options go to the model.
                output = predict({"id": question_id, **mcq_row(question)})
                label = output.get("predicted_answer")
                if label not in OPTION_LABELS:
                    raise ValueError("The MCQ model gave no valid A-D answer (MALFORMED).")
                self._predictions[(model_id, question_id)] = {
                    "question_id": question_id,
                    "predicted": label,
                    "explanation": str(output.get("generated_explanation") or ""),
                }
        return self._predictions[(model_id, question_id)]

    def follow_up(self, request: FollowUpRequest) -> dict[str, Any]:
        question = self._question(request.question_id)
        predicted = self.predict(request.question_id, request.llm)["predicted"]
        _, _, tutors = self._model(request.llm)
        tutor = tutors[UI_MODELS[request.embedding_model]]
        record = {"question_id": question["id"], "question": question["question"], "options": question["options"]}
        with self._lock:
            result = tutor.respond(record, predicted_label=predicted, follow_up=request.message.strip(),
                                   previous_turn=previous_turn(request.history))
        # Show what the reply was written from: the passages in the prompt, in prompt order.
        # On a gate fallback no passage was used, so the weak top-k hits are shown instead.
        hits = result["retrieved"]
        if not result["fallback"]:
            by_id = {hit["doc_id"]: hit for hit in hits}
            hits = [by_id[doc_id] for doc_id in result["passage_ids"]]
        evidence = []
        for hit in hits[: request.top_k]:
            passage = self.passages.get(hit["doc_id"])
            if passage is None:
                raise PassageLookupError(f"Retrieved passage {hit['doc_id']!r} is not in the loaded corpus.")
            text = str(passage["text"])
            evidence.append({
                "id": hit["doc_id"],
                "text": strip_answer_prefixes(text) if tutor.clean_evidence else text,
                "subject": str(passage.get("subject") or "Unknown"),
                "score": float(hit["score"]),
            })
        reply = result["reply"]
        if self.gate_off and not result["fallback"]:
            reply += GATE_OFF_NOTE
        return {"reply": reply, "evidence": evidence, "fallback": bool(result["fallback"])}


def create_app(service: TutorService) -> FastAPI:
    app = FastAPI(title="ClearMed demo", description=__doc__)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        # The CLI constructs the app only after all models and demo questions load.
        return {"status": "ready", "question_count": len(service.questions),
                "embedding_models": list(UI_MODELS), "gate_enabled": not service.gate_off}

    def _lookup(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except UnknownModelError as error:
            raise HTTPException(status_code=404, detail=f"Unknown model {error.args[0]!r}.") from error
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown question {error.args[0]!r}.") from error
        except ValueError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except PassageLookupError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/api/questions", response_model=list[Question])
    def list_questions() -> list[dict[str, Any]]:
        return service.list_questions()

    @app.get("/api/models", response_model=list[ModelInfo])
    def list_models() -> list[dict[str, str]]:
        return service.list_models()

    @app.post("/api/questions/{question_id}/predict", response_model=McqPrediction)
    def predict(question_id: str, llm: str | None = None) -> dict[str, Any]:
        return _lookup(lambda: service.predict(question_id, llm))

    @app.post("/api/follow-up", response_model=FollowUpResponse)
    def follow_up(request: FollowUpRequest) -> dict[str, Any]:
        return _lookup(lambda: service.follow_up(request))

    return app
