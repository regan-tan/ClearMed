"""Shared checks that a loaded dataset is the agreed data and is leak-free.

These helpers reuse the fingerprint and duplicate rules from
``clearmed.data.splitting`` so every model applies identical criteria: a
partition matches the manifest only when its row count and membership
fingerprint both match, and two rows are duplicates when they share a row ID
or a conservatively normalised question.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import Dataset

from clearmed.data.preprocessing import normalize_question_for_comparison
from clearmed.data.splitting import _question_index, _row_id, membership_fingerprint

EVIDENCE_PARTITION = "retrieval_kb_evidence"
"""On-disk name of the explanation-bearing retrieval subset.

``build_manifest`` records it under the top-level ``retrieval_evidence`` key
rather than under ``partitions``.
"""


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a split manifest, failing with an actionable message."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found at {manifest_path}. Generate the splits first with "
            "`uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits`."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Manifest {manifest_path} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest {manifest_path} must contain a JSON object.")
    return manifest


def manifest_entry(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return the recorded row count and fingerprint for one saved partition."""
    if name == EVIDENCE_PARTITION:
        entry = manifest.get("retrieval_evidence")
    else:
        partitions = manifest.get("partitions")
        entry = partitions.get(name) if isinstance(partitions, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError(f"Manifest has no entry for partition {name!r}.")  # noqa: TRY004
    rows = entry.get("rows")
    fingerprint = entry.get("membership_fingerprint")
    if isinstance(rows, bool) or not isinstance(rows, int) or not isinstance(fingerprint, str):
        raise ValueError(f"Manifest entry for {name!r} lacks a row count or fingerprint.")  # noqa: TRY004
    return entry


def verify_against_manifest(
    dataset: Dataset, name: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm *dataset* is exactly the partition the manifest recorded.

    Returns the comparison for experiment metadata and raises ``ValueError``
    on any row-count or fingerprint mismatch, so altered or incomplete data is
    never used silently.
    """
    entry = manifest_entry(manifest, name)
    actual_rows = len(dataset)
    actual_fingerprint = membership_fingerprint(dataset)
    mismatches: list[str] = []
    if actual_rows != entry["rows"]:
        mismatches.append(f"row count is {actual_rows}, expected {entry['rows']}")
    if actual_fingerprint != entry["membership_fingerprint"]:
        mismatches.append("membership fingerprint differs from the manifest")
    if mismatches:
        raise ValueError(
            f"Partition {name!r} does not match the manifest: "
            + "; ".join(mismatches)
            + ". Regenerate the splits rather than using altered data."
        )
    return {
        "partition": name,
        "rows": actual_rows,
        "membership_fingerprint": actual_fingerprint,
        "status": "verified",
    }


def find_duplicates(dataset: Dataset, reference: Dataset) -> list[dict[str, Any]]:
    """Return rows of *dataset* that duplicate any row of *reference*.

    A row matches by shared row ID, by identical normalised question text, or
    both. Reworded questions are deliberately not matched. Results keep the
    order of *dataset*.
    """
    reference_ids = {_row_id(row) for row in reference}
    reference_questions = set(_question_index(reference))
    duplicates: list[dict[str, Any]] = []
    for index, row in enumerate(dataset):
        row_id = _row_id(row)
        question = row.get("question")
        matched_by: list[str] = []
        if row_id in reference_ids:
            matched_by.append("row_id")
        if isinstance(question, str) and (
            normalize_question_for_comparison(question) in reference_questions
        ):
            matched_by.append("normalized_question")
        if matched_by:
            duplicates.append(
                {"index": index, "id": row_id, "question": question, "matched_by": matched_by}
            )
    return duplicates
