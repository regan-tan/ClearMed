# Data

ClearMed uses [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa)
(`openlifescienceai/medmcqa`). It is downloaded with Hugging Face Datasets and is not
committed. Everything in `data/` is git-ignored except this file and `data/retrieval_eval/`.

## MedMCQA fields

`id`, `question`, `opa`, `opb`, `opc`, `opd`, `cop`, `choice_type`, `exp`, `subject_name`,
`topic_name`. `cop` is the correct option (0–3, i.e. A–D). A QA-valid example needs a question,
four nonblank options and a valid answer. Explanation generation also needs a nonblank `exp`.

## Partitions (shared by Part I and Part II)

`scripts/generate_splits.py` splits only the **official training split** into `part1_train`
(75%), `retrieval_kb` (20%) and `internal_dev` (5%) with seed 42. Identical normalised
questions are grouped first, so a duplicate cannot cross partitions. The script also writes
`retrieval_kb_evidence`, the `retrieval_kb` rows that have a nonblank explanation, which is the
source of the Part II passage bank.

```bash
uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits
```

`data/splits/manifest.json` records sizes, membership fingerprints, validity counts, duplicate
statistics and overlap checks against official validation and test. Official validation is
reserved for final scoring. The official test split is not used.

**Leakage policy.** Only official-training rows are indexed for retrieval. Validation and test
explanations never enter the passage bank, queries or model inputs.

## `data/retrieval_eval/` (tracked, Part II)

The small, frozen files that the RQ3 and RQ4 results depend on:

| Path | Contents |
|---|---|
| `queries_calibration.jsonl` | 80 tuning follow-ups on `internal_dev` MCQs |
| `queries_eval.jsonl` | 60 test follow-ups on official-validation MCQs (30 used for RQ4, with 10 second turns) |
| `query_sets.json` | Checksums and counts for the query sets |
| `judging/pool-20261020b/` | Labelling pool key, final relevance judgments, labeller agreement |
| `labels/pool-20261020b/` | Completed labelling sheet and its provenance |
| `rq4/rq4-20261104/` | Graded RQ4 replies (`blind_sheet.csv`, `grounding_sheet.csv`), provenance and hand-checked samples (`spotcheck/`) |

The raw relevance-label file `final_labels.jsonl` sits at the repository root. Its provenance
and checksum are recorded in `labels/pool-20261020b/provenance.json`.
