"""Live demo backend for the Practice MCQ mode of the frontend (Plan 8).

Implements the contract in ``frontend/src/api/types.ts``:

* ``GET /api/questions``: the frozen demo questions (``configs/demo/demo_questions.json``);
* ``POST /api/questions/{id}/predict``: the Part I FLAN-T5 + LoRA answer and explanation;
* ``POST /api/follow-up``: the Part II tutor, with the student's chosen embedding model.

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
        self._predict, self.tutors, self.passages, self.gate_off = predict, tutors, passages, gate_off
        self._predictions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()  # One model call at a time: torch on CPU is not shared safely.

    def _question(self, question_id: str) -> dict[str, Any]:
        if question_id not in self.questions:
            raise KeyError(question_id)
        return self.questions[question_id]

    def list_questions(self) -> list[dict[str, Any]]:
        return list(self.questions.values())

    def predict(self, question_id: str) -> dict[str, Any]:
        question = self._question(question_id)
        with self._lock:
            if question_id not in self._predictions:
                # Only the question and options go to the model.
                output = self._predict({"id": question_id, **mcq_row(question)})
                label = output.get("predicted_answer")
                if label not in OPTION_LABELS:
                    raise ValueError("The MCQ model gave no valid A-D answer (MALFORMED).")
                self._predictions[question_id] = {
                    "question_id": question_id,
                    "predicted": label,
                    "explanation": str(output.get("generated_explanation") or ""),
                }
        return self._predictions[question_id]

    def follow_up(self, request: FollowUpRequest) -> dict[str, Any]:
        question = self._question(request.question_id)
        predicted = self.predict(request.question_id)["predicted"]
        tutor = self.tutors[UI_MODELS[request.embedding_model]]
        record = {"question_id": question["id"], "question": question["question"], "options": question["options"]}
        with self._lock:
            result = tutor.respond(record, predicted_label=predicted, follow_up=request.message.strip(),
                                   previous_turn=previous_turn(request.history))
        evidence = []
        for hit in result["retrieved"][: request.top_k]:
            passage = self.passages[hit["doc_id"]]
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

    def _lookup(call: Callable[[], Any]) -> Any:
        try:
            return call()
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown question {error.args[0]!r}.") from error
        except ValueError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error

    @app.get("/api/questions", response_model=list[Question])
    def list_questions() -> list[dict[str, Any]]:
        return service.list_questions()

    @app.post("/api/questions/{question_id}/predict", response_model=McqPrediction)
    def predict(question_id: str) -> dict[str, Any]:
        return _lookup(lambda: service.predict(question_id))

    @app.post("/api/follow-up", response_model=FollowUpResponse)
    def follow_up(request: FollowUpRequest) -> dict[str, Any]:
        return _lookup(lambda: service.follow_up(request))

    return app
