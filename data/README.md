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

## Data Leakage Policy

Only training-set examples and explanations may be used to construct
retrieval indices.

Validation and test explanations must not be included in the retrieval
corpus because doing so would leak gold-answer information into the
evaluation process.