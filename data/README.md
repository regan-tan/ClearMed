# Data

ClearMed uses the MedMCQA dataset as its primary medical examination
question-answering dataset.

The dataset is downloaded programmatically using Hugging Face Datasets
and is therefore not committed to this repository.

## MedMCQA

Each example may contain:

- a medical examination question
- four answer options
- the correct answer
- an expert explanation
- subject metadata
- topic metadata

The verified dataset fields are `id`, `question`, `opa`, `opb`, `opc`,
`opd`, `cop`, `choice_type`, `exp`, `subject_name`, and `topic_name`.
`cop` is a class label with `a`, `b`, `c`, and `d`, corresponding to the
canonical answer labels A through D. QA-valid examples require a question,
four nonblank options, and a valid answer; explanation generation additionally
requires a nonblank `exp` field. Official splits remain separate.

## Data Leakage Policy

Only training-set examples and explanations may be used to construct
retrieval indices.

Validation and test explanations must not be included in the retrieval
corpus because doing so would leak gold-answer information into the
evaluation process.

## Generated partition artifacts

`scripts/generate_splits.py` partitions only the official training split into
`part1_train` (75%), `retrieval_kb` (20%), and `internal_dev` (5%). Identical
normalized questions are grouped before partitioning, preventing duplicate
question leakage. The generated `retrieval_kb_evidence` subset is filtered
from `retrieval_kb` and requires valid QA fields plus a nonblank explanation;
rows lacking explanations remain available in their primary QA partition.

Run `uv run python scripts/generate_splits.py --seed 42`. The resulting
`data/splits/manifest.json` records membership fingerprints, validity counts,
duplicate statistics, and validation/test overlap warnings. Generated dataset
directories and manifests are local artifacts and must not be committed.
