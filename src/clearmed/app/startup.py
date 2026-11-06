"""Lightweight local-file checks before loading the demo's models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def check_demo_artifacts(
    *, adapter_dir: Path, corpus_run: Path, index_dirs: Mapping[str, Path],
    config_paths: Sequence[Path],
) -> None:
    """Report missing/invalid local files together; never download or write artifacts.

    This checks presence and JSON syntax, not weights, checksums, index compatibility,
    or Hugging Face caches. Existing loaders perform their deeper checks at startup.
    """
    problems: list[str] = []

    def require(path: Path) -> bool:
        if not path.is_file():
            problems.append(f"Missing file: {path}")
            return False
        if path.stat().st_size == 0:
            problems.append(f"Empty file: {path}")
            return False
        return True

    def read_object(path: Path) -> dict | None:
        if not require(path):
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("expected a JSON object")
            return value
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            problems.append(f"Invalid JSON in {path}: {error}")
            return None

    for name in ("adapter_metadata.json", "adapter_config.json", "tokenizer_config.json"):
        read_object(adapter_dir / name)
    for alternatives in (("adapter_model.safetensors", "adapter_model.bin"),
                         ("tokenizer.json", "spiece.model")):
        candidates = [adapter_dir / name for name in alternatives]
        if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
            problems.append("Missing non-empty file (one required): " + " or ".join(map(str, candidates)))

    corpus = read_object(corpus_run / "config.json")
    if corpus is not None:
        passages_file = corpus.get("passages_file")
        if not isinstance(passages_file, str) or not passages_file.strip():
            problems.append(f"Missing passages_file in {corpus_run / 'config.json'}")
        else:
            require(corpus_run / passages_file)

    for directory in index_dirs.values():
        read_object(directory / "index.json")
        require(directory / "doc_ids.json")
        require(directory / "embeddings.npy")
    for path in config_paths:
        read_object(path)

    if problems:
        raise ValueError(
            "Demo artifact checks failed:\n- " + "\n- ".join(problems)
            + "\nRun from the repository root. Restore the existing trained artifacts or select "
            "their locations with --adapter-dir, --corpus-run and --index-run. See docs/demo.md."
        )
