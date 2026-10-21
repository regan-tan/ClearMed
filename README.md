# ClearMed Study

ClearMed Study is a retrieval-grounded conversational tutor for medical students preparing for written medical examinations. It answers medical MCQs, generates explanations, supports follow-up questions, and retrieves supporting medical evidence. When the available evidence is insufficient, it uses a deterministic fallback rather than generating an answer.

This is an NLP research project, not simply a chatbot application.

## Research Questions

- RQ1: MCQ accuracy — TF-IDF vs Scratch Seq2Seq vs FLAN-T5 + LoRA.
- RQ2: Explanation quality — Scratch Seq2Seq vs FLAN-T5 + LoRA.
- RQ3: Retrieval effectiveness — MiniLM vs E5-base-v2 vs BioLORD-2023.
- RQ4: Effect of retrieval quality on grounded conversational responses.

## Dataset

MedMCQA is the primary dataset. Official train, validation, and test splits are preserved where appropriate. Only training evidence may be indexed for retrieval; validation and test explanations must not be included in the retrieval corpus.

### Reproducible training partitions

The official MedMCQA training split is divided into 75% `part1_train` for
Part I model training, 20% `retrieval_kb` for later retrieval development, and
5% `internal_dev` for internal development. The official validation split
remains reserved for final evaluation; the official test split remains
untouched and is not used for scored answer accuracy.

The splitter groups identical normalized question text before assigning rows,
so duplicate questions cannot leak between the three generated subsets. Rows
without explanations remain in their primary partition for MCQ QA; only
QA-valid rows with a nonblank expert explanation are emitted as
`retrieval_kb_evidence`.

Generate artifacts (which are intentionally ignored by Git) with:

```bash
uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits
```

Review `data/splits/manifest.json` after generation for exact sizes,
fingerprints, duplicate statistics, and any reported overlaps with the
official validation or test data. Choose a new output directory to regenerate;
the script refuses to overwrite a non-empty one.

## Part I Models

1. TF-IDF baseline
2. Scratch Seq2Seq using bidirectional GRU with attention
3. FLAN-T5 + LoRA

FLAN-T5-small is intended for local development and smoke testing. FLAN-T5-base is the intended main GPU experiment if feasible.

### TF-IDF MCQ baseline

The Part I classical baseline treats each QA-valid MCQ as four binary
question--candidate examples: one positive correct option and three negative
options. A TF-IDF vocabulary is fit only on `part1_train` question and option
text. The logistic classifier receives candidate TF-IDF features, sparse
element-wise question--candidate interactions, and a cosine-similarity scalar.
It does not receive answer positions, other distractors, gold labels, or expert
explanations as input.

At inference, the model independently scores all four candidates and chooses
the highest-scoring option (ties select the earliest option). Those values are
binary correctness scores, not a normalized four-class distribution or medical
confidence. Training uses binary candidate labels, but the reported primary
metric is original-MCQ answer accuracy on `internal_dev`.

Run a small local smoke experiment after generating partitions:

```bash
uv run python scripts/train_tfidf.py --seed 42 --max-train-questions 200 --max-dev-questions 50 --model-dir checkpoints/tfidf-smoke --output-dir outputs/tfidf-smoke
```

Run the full CPU experiment with:

```bash
uv run python scripts/train_tfidf.py --seed 42
```

Artifacts are saved under `checkpoints/tfidf/` and `outputs/tfidf/` by default:
the fitted vectorizer/classifier, configuration, metrics, subject accuracy, and
a deterministic capped error sample. `internal_dev` is only for development;
the official validation split remains reserved for later comparative
evaluation. Lexical TF-IDF is intentionally lightweight and interpretable, but
has limited capacity for clinical reasoning, paraphrase, and long-range medical
context.

The completed seed-42 baseline scored **39.84% (3,642/9,141)** on
`internal_dev`, after training on 137,117 MCQs (548,468 candidate pairs).
This is an internal-development result, not final validation accuracy; the
official validation split remains reserved for later model comparison.

### Scratch Seq2Seq baseline

The Part I neural baseline is a PyTorch encoder-decoder trained from random
initialisation, with no pretrained weights: a bidirectional GRU encoder,
additive (Bahdanau) attention with padding masks, and a GRU decoder. The source
is the question plus all four labelled options; each segment is truncated
separately, so no option can be dropped. The target is the answer label
followed by the expert explanation, rendered as `Answer: X` /
`Explanation: ...`. Rows without an explanation get an answer-only target, so
no explanation text is fabricated. The vocabulary is fit only on `part1_train`,
and gold labels and explanations never enter model inputs.

Training uses teacher forcing, AdamW, and gradient clipping. Each epoch is
evaluated with greedy generation on `internal_dev`; the checkpoint with the
best accuracy is kept, and training stops early once accuracy stops improving.
Predictions are parsed strictly; outputs without a valid A–D label count as
`MALFORMED` and are scored by the shared `clearmed.evaluation.qa` evaluator.

Run a small local smoke experiment after generating partitions:

```bash
uv run python scripts/train_seq2seq.py --max-train-questions 2000 --max-dev-questions 300 --max-epochs 3 --run-id smoke-2k
```

Run the full experiment on a GPU, then evaluate the selected checkpoint:

```bash
uv run python scripts/train_seq2seq.py --device cuda --batch-size 128 --run-id full-seed42
uv run python scripts/evaluate_seq2seq.py --checkpoint-dir checkpoints/seq2seq/full-seed42/best --output-dir outputs/seq2seq/full-seed42/eval-internal-dev
```

Artifacts are saved under `checkpoints/seq2seq/<run_id>/` and
`outputs/seq2seq/<run_id>/`. The full-data GPU run has not been performed yet.
See [docs/seq2seq.md](docs/seq2seq.md) for explanation-coverage statistics,
preprocessing decisions, smoke-test results, and known limitations.

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
- React 19, TypeScript, Vite, Tailwind CSS, and localStorage-backed preview adapters
- uv
- Git / GitHub

## Setup

```bash
uv sync
```

## Project Status

Current phase: Step 1 partitions and the TF-IDF MCQ baseline are complete. The
Scratch Seq2Seq pipeline is implemented and smoke-tested; its full GPU training
run is pending. The FLAN-T5 + LoRA pipeline is implemented and prepared for a
resumable SMU GPU experiment; its full-dataset run remains pending. Retrieval
experiments and backend integration remain planned research work.

## Disclaimer

ClearMed is an educational research prototype for medical examination preparation. It is not intended for diagnosis, treatment recommendations, or clinical decision-making.
