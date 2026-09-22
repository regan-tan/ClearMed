# ClearMed Study

ClearMed Study is a retrieval-grounded conversational tutor for
medical examination preparation.

The system takes a medical multiple-choice question, predicts the
correct answer, generates an explanation, and supports follow-up
questions using retrieved medical evidence.

## Research Question

Does LoRA fine-tuning of a pretrained language model on medical
examination questions improve answer accuracy and explanation quality
over classical NLP and scratch-trained sequence-to-sequence baselines,
and can retrieval-grounded generation support reliable multi-turn
tutoring?

## Models

1. TF-IDF/classical baseline
2. Seq2seq model trained from scratch
3. FLAN-T5 fine-tuned using LoRA

## Dataset

Primary dataset: MedMCQA

## System Pipeline

Medical MCQ
→ preprocessing
→ answer + explanation generation
→ student follow-up
→ retrieval
→ grounded answer or fallback

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT / LoRA
- scikit-learn
- SentenceTransformers
- Gradio

## Setup

```bash
uv sync