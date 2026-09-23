"""Small, split-preserving loader for the MedMCQA dataset."""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

DATASET_NAME = "openlifescienceai/medmcqa"
SUPPORTED_SPLITS = ("train", "validation", "test")


def load_medmcqa() -> DatasetDict:
    """Load the complete MedMCQA DatasetDict without combining its splits."""
    return load_dataset(DATASET_NAME)


def load_medmcqa_split(
    split: str, max_samples: int | None = None, seed: int = 42
) -> Dataset:
    """Load one official split, optionally returning a deterministic development subset."""
    if split not in SUPPORTED_SPLITS:
        supported = ", ".join(SUPPORTED_SPLITS)
        raise ValueError(f"Unsupported split {split!r}. Expected one of: {supported}.")
    if max_samples is not None and (
        isinstance(max_samples, bool)
        or not isinstance(max_samples, int)
        or max_samples <= 0
    ):
        raise ValueError("max_samples must be a positive integer or None.")

    dataset = load_medmcqa()[split]
    if max_samples is None:
        return dataset

    sample_count = min(max_samples, len(dataset))
    return dataset.shuffle(seed=seed).select(range(sample_count))
