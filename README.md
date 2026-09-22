# ClearMed Study

ClearMed Study is a retrieval-grounded conversational tutor for medical students preparing for written medical examinations. It answers medical MCQs, generates explanations, supports follow-up questions, and retrieves supporting medical evidence. When the available evidence is insufficient, it uses a deterministic fallback rather than generating an answer.

This is an NLP research project, not simply a chatbot application.

## Research Questions

**Part I**

- RQ1: Does LoRA fine-tuning of FLAN-T5 improve medical MCQ performance compared with classical and scratch-trained NLP baselines?
- RQ2: Can retrieval-grounded generation improve follow-up reliability and reduce unsupported medical claims?

**Part II**

- RQ3: How does embedding-model choice affect retrieval effectiveness for medical follow-up questions?
- RQ4: Does stronger retrieval lead to more accurate and better-grounded conversational responses?

## Dataset

MedMCQA is the primary dataset. Official train, validation, and test splits are preserved where appropriate. Only training evidence may be indexed for retrieval; validation and test explanations must not be included in the retrieval corpus.

## Part I Models

1. TF-IDF baseline
2. Scratch Seq2Seq using LSTM/GRU with attention
3. FLAN-T5 + LoRA

FLAN-T5-small is intended for local development and smoke testing. FLAN-T5-base is the intended main GPU experiment if feasible.

## Part II Embedding Models

1. `sentence-transformers/all-MiniLM-L6-v2` — general-purpose semantic baseline
2. `intfloat/e5-base-v2` — retrieval-specialised query/passage embedding model
3. `FremyCompany/BioLORD-2023` — biomedical-specialised embedding model

Part II changes only the embedding model while keeping the rest of the retrieval and generation pipeline fixed.

## Overall Pipeline

```text
Medical MCQ
  → preprocessing
  → FLAN-T5 + LoRA
  → answer + explanation
  → student follow-up
  → evidence retrieval
  → sufficient relevant evidence?

YES → top-k evidence → FLAN-T5 + LoRA → grounded answer
NO  → deterministic fallback
```

The fallback does not pass through FLAN-T5.

## Evaluation

Part I: MCQ answer accuracy, accuracy by medical subject, malformed-output rate, ROUGE-L, BERTScore, and qualitative error analysis.

Part II: Recall@1, Recall@3, Recall@5, MRR, downstream groundedness, unsupported claims, and fallback behaviour.

## Tech Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- scikit-learn
- SentenceTransformers
- Gradio
- uv
- Git / GitHub

## Setup

```bash
uv sync
```

## Project Status

Current phase: dataset exploration and preprocessing.

## Disclaimer

ClearMed is an educational research prototype for medical examination preparation. It is not intended for diagnosis, treatment recommendations, or clinical decision-making.
