"""Grounded tutoring pipeline (Plan 6).

One student follow-up becomes either an evidence-based reply or the fixed
fallback::

    follow-up -> query (predicted answer) -> retrieve top-k -> gate on top-1 score
       |- below threshold -> FALLBACK_MESSAGE (the generator is not called)
       '- at/above threshold -> prompt with <= 3 passages -> generator -> reply

A no-retrieval mode (``retriever=None``) skips retrieval and the gate and
always generates; it is the RQ4 control. The answer line in the query and the
reply's ``Answer:`` pre-fill are always the model's predicted label, never the
gold answer, and the reference explanation never enters a query or prompt.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from clearmed.data.preprocessing import OPTION_LABELS, format_generation_input
from clearmed.retrieval.embedders import Embedder
from clearmed.retrieval.gate import Gate
from clearmed.retrieval.index import DenseIndex
from clearmed.retrieval.queries import query_text_for

FALLBACK_MESSAGE = (
    "I could not find reliable evidence for this follow-up, so I will not guess. "
    "Please check your textbook or ask your tutor."
)
MAX_PASSAGES = 3
RETRIEVE_K = 5
NO_RETRIEVAL = "no_retrieval"
# A retrieval system run without its confidence gate (always answers from its top passages).
GATE_OFF_SUFFIX = "_gate_off"


class Generator(Protocol):
    """Writes the reply. ``answer_label`` is pre-filled as ``Answer: X``."""

    def generate(self, prompt: str, *, answer_label: str) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    @property
    def max_input_tokens(self) -> int: ...


class Retriever(Protocol):
    def retrieve(self, query_text: str, k: int) -> list[dict[str, Any]]: ...


class DenseRetriever:
    """Embeds a query and looks up passage texts for the top-k hits."""

    def __init__(self, embedder: Embedder, index: DenseIndex, passages: Sequence[Mapping[str, Any]]) -> None:
        self.embedder, self.index = embedder, index
        self.texts = {str(p["doc_id"]): str(p["text"]) for p in passages}
        missing = [doc_id for doc_id in index.doc_ids if doc_id not in self.texts]
        if missing:
            raise ValueError(f"{len(missing)} indexed passages have no text, e.g. {missing[:3]}.")
        self._checked = False

    def retrieve(self, query_text: str, k: int) -> list[dict[str, Any]]:
        vectors = self.embedder.encode_queries([query_text])
        if not self._checked:
            self.index.ensure_compatible(self.embedder.describe(dimension=int(vectors.shape[1])))
            self._checked = True
        hits = self.index.search(np.asarray(vectors), k=k)[0]
        return [{**hit, "text": self.texts[hit["doc_id"]]} for hit in hits]


def mcq_row(record: Mapping[str, Any]) -> dict[str, str]:
    """The Part I input fields of a query record: question and options only."""
    options = record["options"]
    return {"question": record["question"], **{f"op{label.lower()}": options[label] for label in OPTION_LABELS}}


@dataclass(frozen=True)
class Prompt:
    text: str
    passage_ids: list[str]
    passages_trimmed: int


def _trim_to_tokens(text: str, budget: int, count_tokens: Callable[[str], int]) -> str:
    """Longest word prefix of ``text`` whose token count fits ``budget``."""
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(" ".join(words[:middle])) <= budget:
            low = middle
        else:
            high = middle - 1
    return " ".join(words[:low])


# MedMCQA explanations often open with the letter of an option the student never
# sees: "Ans. is 'a' i.e., X", "Ans. (c) X", "A i.e. X", "C. X" or "Ans. X".
_LETTER_PREFIX = re.compile(
    r"(Explanation:\s*)(?:(?i:ans(?:wer)?)\b[\s.:\-]*(?:(?i:is)\b[\s.:\-]*)?)?"
    r"(?:\(\s*['\"‘’*]?[a-dA-D]['\"‘’]?\s*\)|['\"‘’*][a-dA-D]['\"‘’]"
    r"|[a-dA-D](?=\s*(?:i\.?\s*e\b|[.:)\-])))"
    r"\s*(?:i\.?\s*e\.?\s*,?|[.:)\-])?\s*"
)
_ANS_PREFIX = re.compile(r"(Explanation:\s*)(?i:ans)\b[\s.:\-]*")


def strip_answer_prefixes(text: str) -> str:
    """Remove a leading answer-letter prefix from a passage's explanation."""
    return _ANS_PREFIX.sub(r"\1", _LETTER_PREFIX.sub(r"\1", text, count=1), count=1)


def build_prompt(
    record: Mapping[str, Any],
    follow_up: str,
    passages: Sequence[Mapping[str, Any]],
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    previous_turn: tuple[str, str] | None = None,
    clean_evidence: bool = False,
    stated_answer: str | None = None,
) -> Prompt:
    """Part I input, then numbered evidence, the previous turn and the follow-up.

    Passages fill the token budget left after the fixed parts, best first; the
    one that overflows is cut at a word boundary and later ones are dropped.
    The MCQ and the follow-up are never cut. ``clean_evidence`` strips
    answer-letter prefixes from the passages shown to the generator (the
    retrieval index is unchanged). ``stated_answer`` (a predicted label, never
    the gold one) adds a ``Tutor's answer: X. <option>`` line after the MCQ.
    """
    head = format_generation_input(mcq_row(record))
    if stated_answer is not None:
        if stated_answer not in OPTION_LABELS:
            raise ValueError(f"Stated answer must be one of {OPTION_LABELS}, got {stated_answer!r}.")
        head += f"\n\nTutor's answer: {stated_answer}. {record['options'][stated_answer]}"
    tail = ""
    if previous_turn is not None:
        tail += f"Previous turn:\nStudent: {previous_turn[0]}\nTutor: {previous_turn[1]}\n\n"
    tail += f"Student follow-up: {follow_up}"

    def render(evidence: list[str]) -> str:
        if not evidence:
            return f"{head}\n\n{tail}"
        lines = "\n".join(f"[{number}] {text}" for number, text in enumerate(evidence, start=1))
        return f"{head}\n\nEvidence:\n{lines}\n\n{tail}"

    if count_tokens(render([])) > max_tokens:
        raise ValueError("The MCQ and follow-up alone exceed the generator's input limit.")
    kept: list[str] = []
    ids: list[str] = []
    trimmed = 0
    for passage in list(passages)[:MAX_PASSAGES]:
        full = " ".join(str(passage["text"]).split())
        if clean_evidence:
            full = strip_answer_prefixes(full)
        if count_tokens(render([*kept, full])) <= max_tokens:
            kept.append(full)
            ids.append(str(passage["doc_id"]))
            continue
        spare = max_tokens - count_tokens(render([*kept, ""]))
        cut = _trim_to_tokens(full, spare, count_tokens) if spare > 0 else ""
        while cut and count_tokens(render([*kept, cut])) > max_tokens:
            cut = " ".join(cut.split()[:-1])
        if cut:
            kept.append(cut)
            ids.append(str(passage["doc_id"]))
            trimmed += 1
        break
    text = render(kept)
    if not kept and passages:
        raise ValueError("No room for any evidence; raise the generator's input limit.")
    return Prompt(text=text, passage_ids=ids, passages_trimmed=trimmed)


def evidence_block(prompt: str | None) -> str:
    """The numbered passages exactly as a prompt gave them to the generator."""
    if not prompt or "\n\nEvidence:\n" not in prompt:
        return ""
    body = prompt.split("\n\nEvidence:\n", 1)[1]
    return re.split(r"\n\n(?:Previous turn:|Student follow-up:)", body, maxsplit=1)[0]


class TutorPipeline:
    """Retrieve, gate and generate for one follow-up at a time."""

    def __init__(
        self,
        generator: Generator,
        *,
        stage: str,
        retriever: Retriever | None = None,
        gate: Gate | None = None,
        method_id: str = NO_RETRIEVAL,
        clean_evidence: bool = False,
        state_answer: bool = False,
        gate_off: bool = False,
    ) -> None:
        if gate_off:
            if retriever is None or gate is not None:
                raise ValueError("gate_off needs a retriever and no gate.")
            if not method_id.endswith(GATE_OFF_SUFFIX):
                raise ValueError(f"A gate-off pipeline's method_id must end with {GATE_OFF_SUFFIX!r}.")
        elif (retriever is None) != (gate is None):
            raise ValueError("A retriever needs a gate and a gate needs a retriever.")
        if retriever is None and method_id != NO_RETRIEVAL:
            raise ValueError(f"No-retrieval mode must use method_id {NO_RETRIEVAL!r}.")
        if gate is not None and gate.method_id != method_id:
            raise ValueError(f"Gate is for {gate.method_id!r}, pipeline is {method_id!r}.")
        self.generator, self.stage, self.retriever, self.gate, self.method_id = (
            generator, stage, retriever, gate, method_id,
        )
        self.gate_off = gate_off
        self.clean_evidence, self.state_answer = clean_evidence, state_answer

    def respond(
        self,
        record: Mapping[str, Any],
        *,
        predicted_label: str,
        follow_up: str | None = None,
        previous_turn: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        """Reply to ``follow_up`` (default: the record's) and log every step."""
        if predicted_label not in OPTION_LABELS:
            raise ValueError(f"Predicted label must be one of {OPTION_LABELS}, got {predicted_label!r}.")
        follow_up = record["follow_up"] if follow_up is None else follow_up
        log: dict[str, Any] = {
            "query_id": record.get("query_id"),
            "question_id": record.get("question_id"),
            "method_id": self.method_id,
            "follow_up": follow_up,
            "previous_turn": list(previous_turn) if previous_turn else None,
            "predicted_label": predicted_label,
            "query_text": None,
            "retrieved": [],
            "top1_score": None,
            "threshold": None,
            "gate_accepted": None,
            "clean_evidence": self.clean_evidence,
            "state_answer": self.state_answer,
        }
        passages: list[dict[str, Any]] = []
        if self.retriever is not None:
            query = query_text_for(
                {**record, "follow_up": follow_up},
                stage=self.stage, answer_source="predicted", predicted_label=predicted_label,
            )
            hits = self.retriever.retrieve(query, RETRIEVE_K)
            top1 = hits[0]["score"] if hits else float("-inf")
            accepted = bool(hits) and (self.gate_off or (self.gate is not None and self.gate.accept(top1)))
            log.update(
                query_text=query,
                retrieved=[{"doc_id": h["doc_id"], "rank": h["rank"], "score": h["score"]} for h in hits],
                top1_score=top1 if hits else None,
                threshold=self.gate.threshold if self.gate is not None else None,
                gate_accepted=accepted,
                gate_off=self.gate_off,
            )
            if not accepted:
                return {**log, "fallback": True, "passage_ids": [], "passages_trimmed": 0,
                        "prompt": None, "reply": FALLBACK_MESSAGE}
            passages = hits[:MAX_PASSAGES]
        prompt = build_prompt(
            record, follow_up, passages, previous_turn=previous_turn, clean_evidence=self.clean_evidence,
            stated_answer=predicted_label if self.state_answer else None,
            count_tokens=self.generator.count_tokens, max_tokens=self.generator.max_input_tokens,
        )
        reply = self.generator.generate(prompt.text, answer_label=predicted_label)
        return {**log, "fallback": False, "passage_ids": prompt.passage_ids,
                "passages_trimmed": prompt.passages_trimmed, "prompt": prompt.text, "reply": reply}

    def converse(self, record: Mapping[str, Any], *, predicted_label: str) -> list[dict[str, Any]]:
        """The first follow-up, then the record's second turn if it has one."""
        first = self.respond(record, predicted_label=predicted_label)
        turns = [{**first, "turn": 1}]
        if record.get("second_turn"):
            second = self.respond(
                record, predicted_label=predicted_label, follow_up=record["second_turn"],
                previous_turn=(first["follow_up"], first["reply"]),
            )
            turns.append({**second, "turn": 2})
        return turns


class StubGenerator:
    """Deterministic stand-in for FLAN-T5, for tests. Records every prompt."""

    def __init__(self, max_input_tokens: int = 512) -> None:
        self._max = max_input_tokens
        self.prompts: list[str] = []

    @property
    def max_input_tokens(self) -> int:
        return self._max

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, prompt: str, *, answer_label: str) -> str:
        self.prompts.append(prompt)
        evidence = prompt.count("\n[")
        return f"Answer: {answer_label}\nExplanation: stub reply using {evidence} evidence passage(s)."


class FlanT5Generator:
    """Greedy FLAN-T5 decoding with the output pre-filled as ``Answer: X Explanation:``.

    T5's tokenizer turns the target newline into a space, so the pre-fill is
    the same token sequence Part I trained on (``Answer: X\\nExplanation: ...``).
    ``no_repeat_ngram_size`` stops decoding loops; ``banned_phrases`` are token
    sequences the reply may not contain (e.g. ``"Ans"``, the MedMCQA
    answer-letter opener). Both default off, i.e. plain greedy decoding.
    ``use_adapter=False`` writes replies with the LoRA adapter switched off,
    i.e. the plain pretrained FLAN-T5 the adapter was trained on.
    """

    def __init__(self, model: Any, tokenizer: Any, *, max_input_tokens: int, device: Any,
                 max_new_tokens: int = 128, no_repeat_ngram_size: int | None = None,
                 banned_phrases: Sequence[str] = (), use_adapter: bool = True) -> None:
        if not use_adapter and not hasattr(model, "disable_adapter"):
            raise ValueError("use_adapter=False needs a PEFT model with disable_adapter().")
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self._max, self.max_new_tokens = max_input_tokens, max_new_tokens
        self.use_adapter = use_adapter
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.banned_phrases = tuple(banned_phrases)
        self._bad_words_ids = [
            tokenizer(phrase, add_special_tokens=False)["input_ids"] for phrase in self.banned_phrases
        ]

    @property
    def settings(self) -> dict[str, Any]:
        return {"decoding": "greedy", "use_adapter": self.use_adapter, "max_new_tokens": self.max_new_tokens,
                "no_repeat_ngram_size": self.no_repeat_ngram_size, "banned_phrases": list(self.banned_phrases)}

    @property
    def max_input_tokens(self) -> int:
        return self._max

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def generate(self, prompt: str, *, answer_label: str) -> str:
        import torch

        if answer_label not in OPTION_LABELS:
            raise ValueError(f"Answer label must be one of {OPTION_LABELS}, got {answer_label!r}.")
        inputs = self.tokenizer(prompt, truncation=False, return_tensors="pt")
        if inputs["input_ids"].shape[1] > self._max:
            raise ValueError("Prompt exceeds the generator's input limit; build it with build_prompt.")
        prefill_ids = self.tokenizer(f"Answer: {answer_label}\nExplanation:", add_special_tokens=False)["input_ids"]
        start = self.model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[start, *prefill_ids]])
        self.model.eval()
        adapter = nullcontext() if self.use_adapter else self.model.disable_adapter()
        with torch.inference_mode(), adapter:
            output = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                do_sample=False, max_new_tokens=self.max_new_tokens,
                **({"no_repeat_ngram_size": self.no_repeat_ngram_size} if self.no_repeat_ngram_size else {}),
                **({"bad_words_ids": self._bad_words_ids} if self._bad_words_ids else {}),
            )[0]
        continuation = output[decoder_input_ids.shape[1]:].detach().cpu().tolist()
        explanation = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        return f"Answer: {answer_label}\nExplanation: {explanation}"
