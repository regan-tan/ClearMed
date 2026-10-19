"""Scratch GRU encoder-decoder with additive (Bahdanau) attention.

All weights are randomly initialised; nothing pretrained is loaded.

Shapes use B = batch, S = source length, T = target length, E = embedding
size, H = hidden size.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class Seq2SeqModelConfig:
    vocab_size: int
    pad_id: int
    embedding_dim: int = 256
    hidden_dim: int = 256
    attention_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.embedding_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("vocab_size, embedding_dim and hidden_dim must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Encoder(nn.Module):
    """Bidirectional GRU over the question + options."""

    def __init__(self, config: Seq2SeqModelConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(config.dropout)
        self.rnn = nn.GRU(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        # Maps [forward_last; backward_last] (2H) -> decoder initial state (H), per layer.
        self.bridge = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.num_layers = config.num_layers

    def forward(
        self, source: torch.Tensor, source_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(source))  # (B, S, E)
        packed = pack_padded_sequence(
            embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs, batch_first=True, total_length=source.size(1)
        )  # (B, S, 2H)
        # hidden: (layers * 2, B, H) -> (layers, 2, B, H)
        hidden = hidden.view(self.num_layers, 2, source.size(0), -1)
        decoder_hidden = torch.tanh(self.bridge(torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)))
        return outputs, decoder_hidden  # (B, S, 2H), (layers, B, H)


class AdditiveAttention(nn.Module):
    """score(s, h_j) = v^T tanh(W_s s + W_h h_j); padded positions are masked."""

    def __init__(self, config: Seq2SeqModelConfig) -> None:
        super().__init__()
        self.query_proj = nn.Linear(config.hidden_dim, config.attention_dim, bias=False)
        self.key_proj = nn.Linear(2 * config.hidden_dim, config.attention_dim, bias=False)
        self.v = nn.Linear(config.attention_dim, 1, bias=False)

    def forward(
        self,
        query: torch.Tensor,  # (B, H) decoder state
        keys: torch.Tensor,  # (B, S, 2H) encoder outputs
        projected_keys: torch.Tensor,  # (B, S, A) cached key_proj(keys)
        source_mask: torch.Tensor,  # (B, S) True for real tokens
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(projected_keys + self.query_proj(query).unsqueeze(1))).squeeze(-1)
        scores = scores.masked_fill(~source_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, S)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, 2H)
        return context, weights


class Decoder(nn.Module):
    """One-token-at-a-time GRU decoder fed [embedding; attention context]."""

    def __init__(self, config: Seq2SeqModelConfig, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.dropout = nn.Dropout(config.dropout)
        self.attention = AdditiveAttention(config)
        self.rnn = nn.GRU(
            config.embedding_dim + 2 * config.hidden_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(
            config.hidden_dim + 2 * config.hidden_dim + config.embedding_dim, config.vocab_size
        )

    def step(
        self,
        token: torch.Tensor,  # (B,)
        hidden: torch.Tensor,  # (layers, B, H)
        keys: torch.Tensor,
        projected_keys: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(token))  # (B, E)
        context, weights = self.attention(hidden[-1], keys, projected_keys, source_mask)
        rnn_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, hidden)
        logits = self.output(torch.cat([output.squeeze(1), context, embedded], dim=-1))
        return logits, hidden, weights  # (B, V), (layers, B, H), (B, S)


class Seq2SeqModel(nn.Module):
    def __init__(self, config: Seq2SeqModelConfig) -> None:
        super().__init__()
        self.config = config
        # One embedding table shared by encoder and decoder (shared vocabulary).
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_id)
        self.encoder = Encoder(config, self.embedding)
        self.decoder = Decoder(config, self.embedding)

    def _encode(self, source: torch.Tensor, source_lengths: torch.Tensor):
        keys, hidden = self.encoder(source, source_lengths)
        projected_keys = self.decoder.attention.key_proj(keys)
        # Mask from true lengths (not token values) so padding content is irrelevant.
        positions = torch.arange(source.size(1), device=source.device)
        source_mask = positions.unsqueeze(0) < source_lengths.to(source.device).unsqueeze(1)
        return keys, projected_keys, source_mask, hidden

    def forward(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        decoder_input: torch.Tensor,
        *,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Return logits (B, T, V). With ratio < 1, some steps feed back predictions."""
        keys, projected_keys, source_mask, hidden = self._encode(source, source_lengths)
        token = decoder_input[:, 0]
        all_logits = []
        for step in range(decoder_input.size(1)):
            logits, hidden, _ = self.decoder.step(token, hidden, keys, projected_keys, source_mask)
            all_logits.append(logits)
            if step + 1 < decoder_input.size(1):
                use_gold = teacher_forcing_ratio >= 1.0 or torch.rand(()) < teacher_forcing_ratio
                token = decoder_input[:, step + 1] if use_gold else logits.argmax(-1)
        return torch.stack(all_logits, dim=1)

    @torch.no_grad()
    def generate(
        self,
        source: torch.Tensor,
        source_lengths: torch.Tensor,
        *,
        sos_id: int,
        eos_id: int,
        max_len: int,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding from the model's own predictions only (no targets)."""
        was_training = self.training
        self.eval()
        keys, projected_keys, source_mask, hidden = self._encode(source, source_lengths)
        batch = source.size(0)
        token = torch.full((batch,), sos_id, dtype=torch.long, device=source.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=source.device)
        generated, attention = [], []
        for _ in range(max_len):
            logits, hidden, weights = self.decoder.step(token, hidden, keys, projected_keys, source_mask)
            token = logits.argmax(-1).masked_fill(finished, eos_id)
            generated.append(token)
            attention.append(weights)
            finished |= token == eos_id
            if finished.all():
                break
        self.train(was_training)
        output = torch.stack(generated, dim=1)
        if return_attention:
            return output, torch.stack(attention, dim=1)
        return output


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
