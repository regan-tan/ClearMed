import torch

from clearmed.data.flan_t5 import (
    FlanT5DataConfig,
    FlanT5MCQDataset,
    collate_flan_t5_examples,
    format_flan_t5_input,
    format_flan_t5_target,
)


class ToyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.tokens = {"<pad>": 0}
        self.reverse = {0: "<pad>"}

    def _encode(self, text):
        ids = []
        for token in text.replace("\n", " ").split():
            if token not in self.tokens:
                token_id = len(self.tokens)
                self.tokens[token] = token_id
                self.reverse[token_id] = token
            ids.append(self.tokens[token])
        return ids

    def __call__(self, text, *, add_special_tokens=True, max_length=None, padding=False, truncation=False, return_tensors=None):
        texts = [text] if isinstance(text, str) else text
        rows = [self._encode(value) for value in texts]
        if truncation and max_length:
            rows = [row[:max_length] for row in rows]
        if return_tensors == "pt":
            width = max(len(row) for row in rows)
            if padding:
                rows = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
            return {"input_ids": torch.tensor(rows), "attention_mask": torch.tensor([[int(value != 0) for value in row] for row in rows])}
        return {"input_ids": rows[0]}

    def decode(self, ids, *, skip_special_tokens=True):
        return " ".join(self.reverse[int(value)] for value in ids if not (skip_special_tokens and int(value) == 0))


def row(*, exp="Atenolol is a beta blocker.", cop=1, question="Which drug is a beta blocker?"):
    return {"id": "example-1", "question": question, "opa": "Metformin", "opb": "Atenolol", "opc": "Amoxicillin", "opd": "Omeprazole", "cop": cop, "exp": exp}


def test_sources_do_not_leak_gold_fields_and_targets_handle_missing_explanations():
    tokenizer, config = ToyTokenizer(), FlanT5DataConfig(80, 20)
    source, _ = format_flan_t5_input(row(), tokenizer, config)
    changed, _ = format_flan_t5_input(row(cop=3, exp="secret reference"), tokenizer, config)
    assert source == changed
    assert "Atenolol" not in source or "Answer: B" not in source
    assert format_flan_t5_target(row()).startswith("Answer: B\nExplanation:")
    assert format_flan_t5_target(row(exp="   ")) == "Answer: B"


def test_option_preserving_source_truncation_and_impossible_limit():
    tokenizer = ToyTokenizer()
    long = row(question=" ".join(["question"] * 100))
    source, truncated = format_flan_t5_input(long, tokenizer, FlanT5DataConfig(25, 20))
    assert truncated
    for label in ("A.", "B.", "C.", "D."):
        assert label in source
    try:
        format_flan_t5_input(long, tokenizer, FlanT5DataConfig(3, 20))
    except ValueError as error:
        assert "all four options" in str(error)
    else:
        raise AssertionError("An impossible source length must be rejected.")


def test_collation_masks_only_target_padding_and_skips_invalid_rows():
    tokenizer, config = ToyTokenizer(), FlanT5DataConfig(80, 20)
    dataset = FlanT5MCQDataset([row(), row(exp=""), {**row(), "id": "bad", "opd": ""}], tokenizer, config)
    assert len(dataset) == 2 and dataset.statistics()["skipped_invalid_rows"] == 1
    batch = collate_flan_t5_examples(dataset.examples, tokenizer, config)
    assert batch["labels"].dtype == torch.long
    assert (batch["labels"] == -100).any()
