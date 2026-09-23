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
