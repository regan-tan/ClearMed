import pytest

from clearmed.data import loader


class FakeSplit:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def shuffle(self, seed):
        return FakeSplit(list(reversed(self.values)) if seed == 42 else self.values)

    def select(self, indices):
        return FakeSplit([self.values[index] for index in indices])


def test_loader_validates_split_and_sample_size():
    with pytest.raises(ValueError):
        loader.load_medmcqa_split("other")
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            loader.load_medmcqa_split("train", value)


def test_loader_sampling_is_deterministic(monkeypatch):
    fake = FakeSplit([1, 2, 3, 4])
    monkeypatch.setattr(loader, "load_medmcqa", lambda: {"train": fake})
    assert loader.load_medmcqa_split("train", max_samples=2, seed=42).values == [4, 3]
    assert loader.load_medmcqa_split("train", max_samples=2, seed=42).values == [4, 3]
