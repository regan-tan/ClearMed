"""Tokenisation, vocabulary, and batching for the scratch Seq2Seq model.

Design decisions (see ``docs/seq2seq.md``):

* **Source** = question + all four options, built at token level with marker
  tokens (``<q> ... <A> ... <B> ... <C> ... <D> ...``).  Truncation is applied
  per segment, so an over-long question can never push an option out of the
  input.  The source is built from ``get_question_and_options`` only, so
  ``cop`` / ``exp`` can never enter model inputs.
* **Target** = one answer-label token first (``<ans_C>``), then the expert
  explanation tokens when one exists, then ``<eos>``.  Rows without an
  explanation get an answer-only target; no explanation text is fabricated.
* **Vocabulary** is fit on training rows only and shared by encoder and decoder.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset

from clearmed.data.preprocessing import (
    OPTION_LABELS,
    get_answer_label,
    get_question_and_options,
    is_nonblank_text,
    is_valid_qa_example,
)

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
QUESTION_MARKER = "<q>"
OPTION_MARKERS = tuple(f"<{label}>" for label in OPTION_LABELS)
ANSWER_TOKENS = tuple(f"<ans_{label}>" for label in OPTION_LABELS)
ANSWER_TOKEN_TO_LABEL = dict(zip(ANSWER_TOKENS, OPTION_LABELS, strict=True))
SPECIAL_TOKENS = (PAD, UNK, SOS, EOS, QUESTION_MARKER, *OPTION_MARKERS, *ANSWER_TOKENS)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]")
_ANSWER_AND_EXPLANATION_PATTERN = re.compile(
    r"\AAnswer: ([ABCD])(?:\r?\n|[ \t]+)Explanation:(?:[ \t]*(.*))?\Z", re.DOTALL
)
_ANSWER_ONLY_PATTERN = re.compile(r"\AAnswer: ([ABCD])\Z")
_STANDALONE_ANSWER_PATTERN = re.compile(r"\A([ABCD])\.?\Z")


@dataclass(frozen=True)
class ParsedGeneration:
    """Strictly interpreted generation output for answer and format reporting."""

    predicted_answer: str | None
    generated_explanation: str
    answer_parse_mode: str
    answer_explanation_format_compliant: bool


def tokenize(text: str) -> list[str]:
    """Lower-case word/punctuation tokeniser (no learned parameters)."""
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    return _TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class Seq2SeqDataConfig:
    """Explicit, recorded limits for vocabulary and sequence construction."""

    min_freq: int = 2
    max_vocab_size: int | None = 30_000
    max_question_tokens: int = 96
    max_option_tokens: int = 24
    max_explanation_tokens: int = 128
    include_explanations: bool = True

    def __post_init__(self) -> None:
        for name in ("min_freq", "max_question_tokens", "max_option_tokens", "max_explanation_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.max_vocab_size is not None and self.max_vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError("max_vocab_size must leave room for non-special tokens.")

    @property
    def max_source_len(self) -> int:
        return 1 + self.max_question_tokens + len(OPTION_LABELS) * (1 + self.max_option_tokens)

    @property
    def max_target_len(self) -> int:
        """Answer token + explanation + <eos> (excludes the <sos> decoder input)."""
        return 1 + self.max_explanation_tokens + 1


@dataclass
class Vocabulary:
    """Shared token <-> id mapping. Special tokens always occupy fixed ids."""

    itos: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))

    def __post_init__(self) -> None:
        if tuple(self.itos[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("Vocabulary must start with the fixed special tokens.")
        if len(set(self.itos)) != len(self.itos):
            raise ValueError("Vocabulary tokens must be unique.")
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def sos_id(self) -> int:
        return self.stoi[SOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def answer_ids(self) -> tuple[int, ...]:
        return tuple(self.stoi[token] for token in ANSWER_TOKENS)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[index] if 0 <= index < len(self.itos) else UNK for index in ids]

    def to_dict(self) -> dict[str, Any]:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Vocabulary:
        return cls(itos=list(payload["itos"]))


def _row_text_tokens(row: Mapping[str, Any], config: Seq2SeqDataConfig) -> list[str]:
    question, options = get_question_and_options(row)
    tokens = tokenize(question)
    for option in options:
        tokens.extend(tokenize(option))
    if config.include_explanations and is_nonblank_text(row.get("exp")):
        tokens.extend(tokenize(row["exp"]))
    return tokens


def build_vocabulary(
    train_rows: Iterable[Mapping[str, Any]], config: Seq2SeqDataConfig
) -> Vocabulary:
    """Fit a vocabulary on QA-valid **training** rows only.

    Callers are responsible for passing only ``part1_train`` rows; the training
    script enforces this with ``assert_train_dev_disjoint`` before calling.
    Ties in frequency are broken alphabetically, so the result is deterministic.
    """
    counts: Counter[str] = Counter()
    for row in train_rows:
        if is_valid_qa_example(row):
            counts.update(_row_text_tokens(row, config))
    for special in SPECIAL_TOKENS:
        counts.pop(special, None)
    eligible = sorted(
        (token for token, count in counts.items() if count >= config.min_freq),
        key=lambda token: (-counts[token], token),
    )
    if config.max_vocab_size is not None:
        eligible = eligible[: config.max_vocab_size - len(SPECIAL_TOKENS)]
    return Vocabulary(itos=[*SPECIAL_TOKENS, *eligible])


@dataclass(frozen=True)
class EncodedExample:
    example_id: str
    source_ids: list[int]
    target_ids: list[int] | None
    question_truncated: bool
    options_truncated: int
    explanation_truncated: bool
    has_explanation: bool


def encode_source(
    row: Mapping[str, Any], vocab: Vocabulary, config: Seq2SeqDataConfig
) -> tuple[list[int], bool, int]:
    """Encode question + all four options; reads no gold fields.

    Returns ``(ids, question_truncated, number_of_truncated_options)``.
    """
    question, options = get_question_and_options(row)
    question_tokens = tokenize(question)
    tokens = [QUESTION_MARKER, *question_tokens[: config.max_question_tokens]]
    options_truncated = 0
    for marker, option in zip(OPTION_MARKERS, options, strict=True):
        option_tokens = tokenize(option)
        options_truncated += len(option_tokens) > config.max_option_tokens
        tokens.extend([marker, *option_tokens[: config.max_option_tokens]])
    return (
        vocab.encode(tokens),
        len(question_tokens) > config.max_question_tokens,
        options_truncated,
    )


def encode_target(
    row: Mapping[str, Any], vocab: Vocabulary, config: Seq2SeqDataConfig
) -> tuple[list[int], bool, bool]:
    """Encode ``<ans_X> [explanation tokens] <eos>`` for a QA-valid row.

    Returns ``(ids, has_explanation, explanation_truncated)``.
    """
    label = get_answer_label(row)
    tokens = [f"<ans_{label}>"]
    has_explanation = config.include_explanations and is_nonblank_text(row.get("exp"))
    truncated = False
    if has_explanation:
        explanation_tokens = tokenize(row["exp"])
        truncated = len(explanation_tokens) > config.max_explanation_tokens
        tokens.extend(explanation_tokens[: config.max_explanation_tokens])
    tokens.append(EOS)
    return vocab.encode(tokens), has_explanation, truncated


def encode_example(
    row: Mapping[str, Any],
    vocab: Vocabulary,
    config: Seq2SeqDataConfig,
    *,
    with_target: bool = True,
) -> EncodedExample:
    source_ids, question_truncated, options_truncated = encode_source(row, vocab, config)
    target_ids, has_explanation, explanation_truncated = (None, False, False)
    if with_target:
        target_ids, has_explanation, explanation_truncated = encode_target(row, vocab, config)
    return EncodedExample(
        example_id=str(row.get("id")),
        source_ids=source_ids,
        target_ids=target_ids,
        question_truncated=question_truncated,
        options_truncated=options_truncated,
        explanation_truncated=explanation_truncated,
        has_explanation=has_explanation,
    )


class Seq2SeqMCQDataset(TorchDataset):
    """Encoded QA-valid rows; invalid rows are skipped and counted."""

    def __init__(
        self,
        rows: Iterable[Mapping[str, Any]],
        vocab: Vocabulary,
        config: Seq2SeqDataConfig,
    ) -> None:
        self.examples: list[EncodedExample] = []
        self.skipped_invalid = 0
        for row in rows:
            if not is_valid_qa_example(row):
                self.skipped_invalid += 1
                continue
            self.examples.append(encode_example(row, vocab, config))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedExample:
        return self.examples[index]

    def statistics(self, vocab: Vocabulary) -> dict[str, Any]:
        """Coverage and truncation stats to record with every run."""
        total = len(self.examples)
        source_tokens = sum(len(example.source_ids) for example in self.examples)
        source_unk = sum(example.source_ids.count(vocab.unk_id) for example in self.examples)

        def rate(numerator: int) -> float:
            return numerator / total if total else 0.0

        with_explanation = sum(example.has_explanation for example in self.examples)
        return {
            "examples": total,
            "skipped_invalid_rows": self.skipped_invalid,
            "with_explanation": with_explanation,
            "answer_only_targets": total - with_explanation,
            "explanation_coverage": rate(with_explanation),
            "question_truncated_rate": rate(sum(e.question_truncated for e in self.examples)),
            "rows_with_truncated_option_rate": rate(sum(e.options_truncated > 0 for e in self.examples)),
            "explanation_truncated_rate": rate(sum(e.explanation_truncated for e in self.examples)),
            "source_oov_rate": source_unk / source_tokens if source_tokens else 0.0,
        }


@dataclass
class Seq2SeqBatch:
    example_ids: list[str]
    source: torch.Tensor  # (B, S)
    source_lengths: torch.Tensor  # (B,) on CPU, for packing
    decoder_input: torch.Tensor | None  # (B, T): <sos> + target[:-1]
    decoder_target: torch.Tensor | None  # (B, T): target

    def to(self, device: torch.device | str) -> Seq2SeqBatch:
        return Seq2SeqBatch(
            example_ids=self.example_ids,
            source=self.source.to(device),
            source_lengths=self.source_lengths,
            decoder_input=None if self.decoder_input is None else self.decoder_input.to(device),
            decoder_target=None if self.decoder_target is None else self.decoder_target.to(device),
        )


def _pad(sequences: Sequence[Sequence[int]], pad_id: int) -> torch.Tensor:
    width = max(len(sequence) for sequence in sequences)
    padded = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return padded


def collate_examples(examples: Sequence[EncodedExample], vocab: Vocabulary) -> Seq2SeqBatch:
    """Pad a batch; builds teacher-forcing inputs when targets exist."""
    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    source = _pad([example.source_ids for example in examples], vocab.pad_id)
    lengths = torch.tensor([len(example.source_ids) for example in examples], dtype=torch.long)
    decoder_input = decoder_target = None
    if all(example.target_ids is not None for example in examples):
        targets = [example.target_ids for example in examples]
        decoder_target = _pad(targets, vocab.pad_id)  # type: ignore[arg-type]
        decoder_input = _pad([[vocab.sos_id, *target[:-1]] for target in targets], vocab.pad_id)  # type: ignore[index]
    return Seq2SeqBatch(
        example_ids=[example.example_id for example in examples],
        source=source,
        source_lengths=lengths,
        decoder_input=decoder_input,
        decoder_target=decoder_target,
    )


def render_output(token_ids: Iterable[int], vocab: Vocabulary) -> str:
    """Render generated ids into the shared ``Answer: X\\nExplanation: ...`` contract.

    A first token that is not an answer token is rendered verbatim, so strict
    parsing later reports the output as malformed instead of guessing.
    """
    tokens: list[str] = []
    for token in vocab.decode(token_ids):
        if token == EOS:
            break
        tokens.append(token)
    if not tokens or tokens[0] not in ANSWER_TOKEN_TO_LABEL:
        return " ".join(tokens)
    answer = ANSWER_TOKEN_TO_LABEL[tokens[0]]
    explanation = " ".join(tokens[1:])
    return f"Answer: {answer}\nExplanation: {explanation}".rstrip()


def parse_generation_output(raw_output: str) -> ParsedGeneration:
    """Parse only unambiguous MCQ answers without repairing model output.

    Accepted answers are the requested ``Answer: X`` form (with optional
    ``Explanation:`` field), or a standalone ``A``--``D`` optionally followed
    by a period. T5 decoding normalizes target newlines into spaces, so the
    structural ``Answer: X Explanation: ...`` rendering is equivalent to the
    newline form. Anything containing additional answer-like text is malformed
    rather than guessed, e.g. ``A or B``, ``B12.``, and ``ASD``.
    """
    output = (raw_output or "").strip()
    match = _ANSWER_AND_EXPLANATION_PATTERN.fullmatch(output)
    if match:
        explanation = (match.group(2) or "").strip()
        return ParsedGeneration(match.group(1), explanation, "answer_and_explanation", True)
    match = _ANSWER_ONLY_PATTERN.fullmatch(output)
    if match:
        return ParsedGeneration(match.group(1), "", "answer_only", False)
    match = _STANDALONE_ANSWER_PATTERN.fullmatch(output)
    if match:
        return ParsedGeneration(match.group(1), "", "standalone_letter", False)
    return ParsedGeneration(None, "", "malformed", False)


def parse_answer_label(raw_output: str) -> str | None:
    """Return only a strictly unambiguous A--D prediction, if present."""
    return parse_generation_output(raw_output).predicted_answer


def parse_explanation(raw_output: str) -> str:
    """Return an explicitly generated explanation; never infer one."""
    return parse_generation_output(raw_output).generated_explanation


def config_to_dict(config: Seq2SeqDataConfig) -> dict[str, Any]:
    return asdict(config)
