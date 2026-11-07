# ClearMed Study

ClearMed Study is a conversational tutor for medical examination practice, built and
evaluated on [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa). It is a
CS425 (Generative AI for Natural Language Communication) group research project in two parts:

| Part | Question | Research questions | Methods |
|---|---|---|---|
| **I** | Answer a medical MCQ and explain it | RQ1 MCQ accuracy, RQ2 explanation quality | `tfidf`, `seq2seq`, `flan_t5_lora` |
| **II** | Answer the student's follow-up from retrieved evidence | RQ3 retrieval quality, RQ4 effect of retrieval on replies | `minilm`, `e5_base_v2`, `biolord_2023` |

Part I picks and freezes the MCQ answerer (FLAN-T5-base + LoRA). Part II builds the follow-up
tutor on top of that frozen checkpoint. The final report describes the full method and results;
this README covers the code layout, the results in brief and how to run everything.

> ClearMed is an exam-practice research prototype. It is not for diagnosis, treatment or
> clinical decision support, and its replies are not verified medical statements.

## Repository layout

```text
src/clearmed/
  data/          shared  MedMCQA loading, MCQ formatting, seed-42 partitioning, leakage checks
                 Part I  seq2seq.py, flan_t5.py (model-specific datasets)
  baselines/     Part I  TF-IDF + logistic regression
  models/        Part I  scratch GRU Seq2Seq, FLAN-T5 + LoRA
  training/      Part I  training loops for Seq2Seq and FLAN-T5 + LoRA
  evaluation/    Part I  qa.py (shared MCQ evaluator), flan_t5.py
                 Part II retrieval.py (RQ3 metrics), conversation.py (RQ4 scoring)
  retrieval/     Part II passage bank, embedding adapters, index, queries, confidence check, labelling pool
  conversation/  Part II tutor pipeline and RQ4 rubric
  app/           Part II FastAPI backend for the live demo
scripts/         thin CLI entry points (listed per part below)
configs/         Part II frozen settings: retrieval/, conversation/, demo/
data/            dataset artifacts (git-ignored) + data/retrieval_eval/ (tracked Part II labels and sheets)
docs/            per-model and per-step documentation (see docs/README.md)
frontend/        React + Vite study app; Practice MCQ mode talks to the demo backend
tests/           synthetic-data tests; no GPU, network, dataset or checkpoint needed
```

`data/`, `checkpoints/` and `outputs/` hold downloaded datasets, generated splits, model
checkpoints, indexes and run outputs. They are git-ignored except for `data/README.md` and the
small Part II evaluation files in `data/retrieval_eval/`.

## Setup

Python 3.11, managed with [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
uv run pytest        # full test suite, synthetic data only
uv run ruff check .
```

## Shared data: the seed-42 partitions

Both parts use the same partitions of the **official MedMCQA training split** (182,822 rows),
made once with seed 42. Duplicate normalised questions are grouped first, so no question can
appear in two partitions.

| Partition | Rows | Purpose |
|---|---:|---|
| `part1_train` | 137,117 | Part I: fit all three models |
| `internal_dev` | 9,141 | Part I: checkpoint choice. Part II: the 80 tuning questions |
| `retrieval_kb` | 36,564 | Part II: source of the passage bank; never used to train a model |
| Official validation | 4,183 | Final held-out scoring for both parts |

The official test split is not used.

```bash
uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits
```

This writes the partitions plus `data/splits/manifest.json` (sizes, membership fingerprints,
overlap audits). The script refuses a non-empty output directory. Every training script checks
the partitions against the manifest before use. See [data/README.md](data/README.md).

---

## Part I: MCQ answering and explanation generation

### Models

| Model | ID | Summary | Docs |
|---|---|---|---|
| TF-IDF + LR | `tfidf` | Scores each of the four question–option pairs with a binary logistic classifier; picks the highest | [docs/tfidf_baseline.md](docs/tfidf_baseline.md) |
| Scratch Seq2Seq | `seq2seq` | Bidirectional GRU encoder + attention + GRU decoder, no pretrained weights | [docs/seq2seq.md](docs/seq2seq.md) |
| FLAN-T5 + LoRA | `flan_t5_lora` | `google/flan-t5-base`, LoRA on q/v (rank 8, α 16) | [docs/flan_t5_lora.md](docs/flan_t5_lora.md), [GPU guide](docs/flan_t5_gpu_training.md) |

The models see only the question and its four options, never the gold answer or the
explanation. The two generators output `Answer: X` / `Explanation: ...`. The answer letter is
parsed strictly, and anything outside A–D counts as `MALFORMED`.

### Results (official validation, 4,183 questions)

| Model | Accuracy | 95% Wilson CI |
|---|---:|---|
| FLAN-T5 + LoRA | 33.37% | 31.96–34.82% |
| TF-IDF + LR | 32.78% | 31.37–34.21% |
| Scratch Seq2Seq | 31.03% | 29.65–32.45% |

- **RQ1.** FLAN-T5 is significantly better than Seq2Seq (Holm-adjusted McNemar p = 0.0261). It
  is not significantly different from TF-IDF (p = 0.5571).
- **RQ2.** This was measured on the 2,206 questions that have a reference explanation. FLAN-T5
  scored higher than Seq2Seq on ROUGE-L F1 (0.09415 vs 0.06622) and BERTScore F1 (0.71543 vs
  0.58144). These measure similarity to the reference, not medical correctness.

### How to run

```bash
uv run python scripts/train_tfidf.py --seed 42
uv run python scripts/train_seq2seq.py --device cuda --batch-size 128 \
  --max-dev-questions 2000 --run-id full-seed42
uv run python scripts/train_flan_t5_lora.py --device cuda --require-cuda --batch-size 2 \
  --gradient-accumulation-steps 4 --run-id flan-base-full-seed42
```

The Seq2Seq and FLAN-T5 runs need a GPU. TF-IDF runs on CPU, and a quick smoke run is:

```bash
uv run python scripts/train_tfidf.py --seed 42 --max-train-questions 200 --max-dev-questions 50 \
  --model-dir checkpoints/tfidf-smoke --output-dir outputs/tfidf-smoke
```

For development evaluation on `internal_dev`, use `scripts/evaluate_seq2seq.py` and
`scripts/evaluate_flan_t5_lora.py` with `--checkpoint-dir` and a fresh `--output-dir`. These
scripts check `internal_dev` and must not be pointed at official validation. The final held-out
scoring used a separately archived heldout-v2 evaluator and Slurm wrappers that are not part of
this repository (see the report, Section 6 and Appendix A).

Part I scripts: `generate_splits.py`, `train_tfidf.py`, `train_seq2seq.py`,
`evaluate_seq2seq.py`, `train_flan_t5_lora.py`, `evaluate_flan_t5_lora.py`,
`slurm/flan_t5_lora.sbatch`. Data inspection helpers: `analyze_dataset.py`,
`inspect_dataset.py`, `inspect_seq2seq_data.py`.

---

## Part II: follow-up questions answered from retrieved evidence

### How the tutor works

```text
student's MCQ + follow-up question
  1. Part I FLAN-T5 + LoRA picks A–D (the predicted answer; never the correct one)
  2. query = follow-up + predicted answer + question and options
  3. MiniLM / E5-base-v2 / BioLORD-2023 search the 32,218-passage bank and keep the top 5 (cosine)
  4. top-1 score >= that model's threshold?
       no  -> fixed fallback message ("I could not find reliable evidence..."); nothing is generated
       yes -> 5. plain FLAN-T5-base (LoRA adapter off) reads the top 3 passages and writes the reply
```

The passage bank comes from the 36,564 `retrieval_kb` questions, minus the 4,346 that have an
empty explanation. Each passage is a question, the text of its correct option and its
explanation.

### Settings (frozen before testing)

| | MiniLM | E5-base-v2 | BioLORD-2023 |
|---|---|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` | `intfloat/e5-base-v2` | `FremyCompany/BioLORD-2023` |
| Max input length (tokens) | 256 | 512 | 128 |
| Input prefixes | none | `query: ` / `passage: ` | none |
| Confidence threshold | 0.673 | 0.867 | 0.648 |

Thresholds were set on 80 `internal_dev` tuning follow-ups (best F1) and are frozen in
[configs/retrieval/gate_thresholds.json](configs/retrieval/gate_thresholds.json). The reply
settings are frozen in [configs/conversation/rq4_frozen.json](configs/conversation/rq4_frozen.json).

### Results (60 official-validation MCQs, one follow-up each)

**RQ3: retrieval**, scored on the 24 follow-ups that have at least one relevant passage
(95% bootstrap CIs):

| Model | Hit@5 | Recall@5 | MRR@5 |
|---|---|---|---|
| MiniLM | 0.92 [0.79, 1.00] | 0.80 [0.67, 0.91] | 0.66 [0.51, 0.80] |
| E5-base-v2 | 0.75 [0.58, 0.92] | 0.58 [0.42, 0.74] | 0.51 [0.35, 0.68] |
| BioLORD-2023 | 0.46 [0.25, 0.67] | 0.32 [0.17, 0.49] | 0.38 [0.20, 0.56] |

MiniLM beats BioLORD (Hit@5 p = 0.004). MiniLM and E5 do not differ significantly.

**RQ4: replies.** The tutor answered 40 follow-ups on a random 30 of the MCQs: 30 first turns
and 10 second turns. Each reply was scored 0–2 per criterion. With the confidence check on, the
tutor refused 30–34 of the 40. With it off:

| System | Correct | No unsupported claims | Grounded |
|---|---|---|---|
| No retrieval (control) | 0.05 | 0.62 | – |
| MiniLM | 0.23 | 0.88 | 0.97 |
| E5-base-v2 | 0.23 | 0.75 | 0.65 |
| BioLORD-2023 | 0.20 | 0.88 | 0.70 |

Replies written with passages scored higher on correctness. The gap narrowly missed
significance (p = 0.062–0.107).

### How to run

Part II runs on CPU. It needs the Part I splits (`data/splits`) and the frozen checkpoint
`checkpoints/flan_t5_lora/flan-base-full-seed42/best`. Use a new `--run-id` for any rerun;
scripts never overwrite an existing run.

```bash
# 1. Passage bank and the three search indexes
uv run python scripts/build_retrieval_corpus.py
uv run python scripts/build_retrieval_index.py --method minilm
uv run python scripts/build_retrieval_index.py --method e5_base_v2
uv run python scripts/build_retrieval_index.py --method biolord_2023

# 2. Follow-up questions (80 tuning, 60 test) and retrieval runs
uv run python scripts/make_followup_queries.py
uv run python scripts/run_retrieval.py --queries calibration --index-run full-20261016
uv run python scripts/run_retrieval.py --queries evaluation --index-run full-20261016

# 3. Relevance labels, thresholds and RQ3
uv run python scripts/import_labels.py --pool-id pool-20261020b
uv run python scripts/calibrate_thresholds.py --pool-id pool-20261020b
uv run python scripts/evaluate_retrieval.py --pool-id pool-20261020b

# 4. RQ4: generate replies, export grading sheets, score them
uv run python scripts/run_conversation_eval.py --run-id rq4-new \
  --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best \
  --clean-evidence --no-repeat-ngram-size 3 --reply-without-adapter --gate-off
uv run python scripts/export_rubric_sheets.py --run-id rq4-new
uv run python scripts/evaluate_conversation.py --run-id rq4-new
uv run python scripts/analyze_gate_tradeoff.py --run-id rq4-new
```

Notes on reproducing:

- **Frozen run IDs.** The reported runs are corpus `20261016-142815` and indexes
  `full-20261016`. To rebuild indexes, pass `--run-id <new-id>` to `build_retrieval_index.py`,
  then use that ID as `--index-run` in later steps. The defaults of `run_conversation_eval.py`
  and `serve_demo.py` point at the frozen runs.
- **Frozen inputs are committed.** The query sets (`data/retrieval_eval/queries_*.jsonl`), the
  relevance judgments (`data/retrieval_eval/judging/pool-20261020b/`) and the thresholds
  (`configs/retrieval/gate_thresholds.json`) are already in the repository.
  `make_followup_queries.py`, `import_labels.py` and `calibrate_thresholds.py` refuse to
  overwrite them, so in a fresh checkout you can skip those three steps.
  `export_judging_pool.py` built the labelling pool and is only needed to make a new pool.
- **Reported results.** RQ3 is `outputs/retrieval/eval/rq3-20261101/` and RQ4 is
  `outputs/conversation/eval/rq4-20261104/`. All graded RQ4 replies, with their scores, are in
  `data/retrieval_eval/rq4/rq4-20261104/` (`blind_sheet.csv`, `grounding_sheet.csv`). The
  hand-checked samples are in `spotcheck/` in the same folder.
- **Reply-model trial.** `run_conversation_pilot.py` is the `internal_dev` trial that chose
  plain FLAN-T5-base as the reply model (report Section 3.6).

Labelling and grading guides: [docs/retrieval_labelling_guideline.md](docs/retrieval_labelling_guideline.md),
[docs/rq4_grading_guide.md](docs/rq4_grading_guide.md).

### Live demo

```bash
uv run python scripts/serve_demo.py --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best
cd frontend && npm run dev                        # PowerShell: npm.cmd run dev
```

Open the Vite address and switch to **Practice MCQ**. See [docs/demo.md](docs/demo.md) and
[frontend/README.md](frontend/README.md).

## License

See [LICENSE](LICENSE).
