# ClearMed Study

**Retrieval-grounded conversational tutoring for medical examination preparation**  

ClearMed Study investigates a medical exam-preparation tutor that answers multiple-choice questions (MCQs), explains answers, and supports follow-up questions with retrieved study evidence. The research has two connected parts: comparing medical QA models and explanations, then testing whether retrieval improves a conversational tutor's responses.

> **Research and educational use only.** ClearMed is not a diagnostic tool, a source of verified medical advice, or a clinical decision-support system. Generated answers and explanations can be wrong.

## Research questions and key findings

| Research question | Main finding |
|---|---|
| **RQ1 — MCQ answering** | FLAN-T5-base + LoRA reached **33.37%** accuracy, versus **32.78%** for TF-IDF + logistic regression and **31.03%** for scratch Seq2Seq, on 4,183 held-out MCQs. |
| **RQ2 — Explanation generation** | FLAN-T5 + LoRA had higher ROUGE-L and BERTScore similarity to reference explanations than scratch Seq2Seq; this does **not** establish medical factual correctness. |
| **RQ3 — Evidence retrieval** | MiniLM ranked highest among three retrievers, with **Hit@5 = 0.92** on 24 scorable follow-ups; it significantly outperformed BioLORD, but not E5. |
| **RQ4 — Conversational grounding** | On 40 follow-up turns, adding retrieved passages **did not significantly improve** MedGemma-27B's response correctness over a no-retrieval control. Grounding and confidence-gate coverage remained limitations. |

**Important architectural change:** The original proposal used a frozen FLAN-T5 + LoRA model as the MCQ answerer for Part II. After poor held-out accuracy and four follow-up diagnostic experiments, we investigated MedGemma-4B and MedGemma-27B. The **final RQ4 research evaluation uses MedGemma-27B** for MCQ answering and reply generation. The `main` branch still includes the **earlier FLAN-based conversational evaluator and demo**; the MedGemma research and demo workflows were maintained in separate GPU packages. These are distinct systems and should not be confused.

## 1. Repository structure

```text
ClearMed/
├── configs/
│   ├── conversation/       # Frozen earlier FLAN-tutor settings
│   ├── demo/               # Demo question IDs
│   └── retrieval/          # Retriever configurations and frozen gate thresholds
├── data/
│   ├── README.md
│   └── retrieval_eval/     # Tracked follow-up queries, relevance labels and earlier RQ4 grading sheets
├── docs/                   # Model guides, labelling rubric, architecture and demo notes
├── frontend/               # React, TypeScript and Vite application
├── scripts/                # Preprocessing, training, retrieval, evaluation and demo CLIs
├── src/clearmed/
│   ├── app/                 # FastAPI demo backend
│   ├── baselines/           # TF-IDF + logistic regression
│   ├── conversation/        # Follow-up tutor and grading logic
│   ├── data/                # MedMCQA loading, splits, preprocessing and audits
│   ├── evaluation/          # QA, retrieval and conversation metrics
│   ├── models/              # GRU Seq2Seq and FLAN-T5 + LoRA
│   ├── retrieval/           # Corpus, embeddings, indexes, queries and confidence gate
│   └── training/            # Neural training loops
├── tests/                  # Synthetic-data tests
├── pyproject.toml
└── uv.lock
```

Generated files under `checkpoints/`, `outputs/` and most of `data/` are excluded from Git. **Cloning this repository does not automatically provide the trained checkpoints, retrieval indexes or archived final predictions.** See [Artifacts and reproducibility](#8-artifacts-and-reproducibility).

## 2. Requirements and installation

- Python **3.11**, as specified in `.python-version`.
- [`uv`](https://docs.astral.sh/uv/) for the locked Python environment.
- Git and internet access for the initial MedMCQA and pretrained-model downloads.
- CPU for TF-IDF, dataset preparation, retrieval and automated tests. A CUDA-capable GPU is recommended for Seq2Seq and FLAN-T5 training; MedGemma-27B 4-bit inference requires a sufficiently provisioned GPU environment.
- Node.js and npm for the optional frontend.

```bash
git clone https://github.com/regan-tan/ClearMed.git
cd ClearMed
uv sync --frozen

# Quick verification; these tests use synthetic data rather than trained checkpoints.
uv run pytest
uv run ruff check .
```

Commands below use **Bash** (Linux, macOS, or WSL). Run them from the repository root unless stated otherwise. Use fresh output IDs rather than overwriting frozen experiment artifacts.

## 3. Dataset and leakage controls

The project uses [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), containing medical questions, four answer options, a gold answer and, for some records, an expert explanation. The official training split contains 182,822 questions. A seed-42 partition groups duplicate normalised questions so that copies do not cross partitions.

| Partition | Questions | Use |
|---|---:|---|
| `part1_train` | 137,117 | Train all three Part I models |
| `internal_dev` | 9,141 | Checkpoint selection, diagnostics and threshold calibration |
| `retrieval_kb` | 36,564 | Source material for Part II retrieval; excluded from Part I fitting |
| Official validation | 4,183 | Held-out evaluation |
| Official test | 6,150 | Not evaluated because public gold labels were unavailable |

The retrieval bank has **32,218 passages** after excluding 4,346 KB rows without usable explanations. Each passage contains its source question, correct option text and reference explanation. The dataset audit found 74 duplicate-question groups (148 rows) and no overlapping normalised questions across canonical partitions or with held-out data.

Generate the canonical split only when it is absent:

```bash
uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits
```

The generated `data/splits/manifest.json` records membership fingerprints and verification results. The split script refuses to overwrite a nonempty output directory. In MCQ inference, models receive **the question and four options, never the gold answer or reference explanation**.

For RQ2, only **2,206** official-validation questions have a usable reference explanation. MedMCQA does not supply real student dialogues, so Part II uses 80 development follow-ups and 60 official-validation follow-ups built from question templates, including deliberately off-topic prompts. RQ4 uses 30 of these MCQs with 10 additional second turns, producing **40 follow-up turns** per configuration.

See [data/README.md](data/README.md) for the file formats and split rules.

## 4. Part I — MCQ answering and explanation generation

### Models and training

| Model | Method | Reported configuration |
|---|---|---|
| **TF-IDF + LR** | Binary classifier scores each question–option pair and selects the highest score | Unigrams/bigrams, up to 50k features, `min_df=2`, liblinear, balanced classes |
| **Scratch Seq2Seq** | Bidirectional GRU encoder, additive attention, autoregressive GRU decoder | 30k vocabulary, 256-dimensional embeddings/hidden states, dropout 0.2, AdamW LR 1e-3, batch 128 |
| **FLAN-T5 + LoRA** | `google/flan-t5-base` with frozen base weights and LoRA on query/value attention projections | LoRA rank 8, alpha 16, dropout 0.05, AdamW LR 1e-4, batch 2, accumulation 4, bf16 |

The selected Seq2Seq checkpoint was **epoch 9**, based on development MCQ accuracy over a 2,000-question selection subset. The selected FLAN-T5 adapter was **step 51,000**, based on teacher-forced loss on 512 development examples. Generative answers are parsed as A–D; invalid labels count as wrong. The final FLAN held-out evaluation used greedy decoding with up to 128 new tokens for answer-plus-explanation output.

Run the training entry points after generating the canonical splits:

```bash
# CPU baseline
uv run python scripts/train_tfidf.py --seed 42

# Scratch GRU Seq2Seq (GPU)
uv run python scripts/train_seq2seq.py --device cuda --batch-size 128 \
  --max-dev-questions 2000 --run-id seq2seq-reproduction

# FLAN-T5-base + LoRA (GPU)
uv run python scripts/train_flan_t5_lora.py --device cuda --require-cuda \
  --batch-size 2 --gradient-accumulation-steps 4 \
  --run-id flan-reproduction
```

A small CPU-friendly TF-IDF smoke run is available:

```bash
uv run python scripts/train_tfidf.py --seed 42 \
  --max-train-questions 200 --max-dev-questions 50 \
  --model-dir checkpoints/tfidf-smoke --output-dir outputs/tfidf-smoke
```

For **development-only** evaluation of saved neural checkpoints:

```bash
uv run python scripts/evaluate_seq2seq.py \
  --checkpoint-dir checkpoints/seq2seq/seq2seq-reproduction/best \
  --output-dir outputs/seq2seq/eval-readme

uv run python scripts/evaluate_flan_t5_lora.py \
  --checkpoint-dir checkpoints/flan_t5_lora/flan-reproduction/best \
  --output-dir outputs/flan_t5_lora/eval-readme
```

These two evaluators operate on `internal_dev`; **they are not the archived official-validation evaluator**. The published held-out Part I scores were produced with separately archived `heldout-v2` evaluation scripts and Slurm wrappers. Reproducing the exact scores requires those scripts, frozen checkpoints, manifests and model settings, not merely the training commands above.

### Part I results

| Model | Correct / 4,183 | Answer accuracy |
|---|---:|---:|
| FLAN-T5 + LoRA | 1,396 | **33.37%** |
| TF-IDF + LR | 1,371 | 32.78% |
| Scratch Seq2Seq | 1,298 | 31.03% |

FLAN-T5 significantly exceeded scratch Seq2Seq in paired comparisons (Holm-adjusted McNemar **p = 0.0261**), but did not significantly exceed TF-IDF (**p = 0.5571**).

On the 2,206 explanation-eligible validation questions:

| Metric | Scratch Seq2Seq | FLAN-T5 + LoRA |
|---|---:|---:|
| ROUGE-L F1 | 0.06622 | **0.09415** |
| BERTScore F1 | 0.58144 | **0.71543** |
| Nonempty explanations | 75.7% | **87.2%** |

ROUGE-L and BERTScore measure similarity to reference explanations and do not independently verify medical correctness.

### Why we explored MedGemma

Because 33.37% MCQ accuracy was unsuitable for a dependable tutor, we ran four development-stage investigations:

1. **Base versus fine-tuned FLAN-T5.** Accuracy on a 300-MCQ pilot changed from 31.67% to 37.00%.
2. **Rotated answer options.** Semantic consistency was 46.33% for the base model versus 8.00% for the fine-tuned model, indicating strong answer-position sensitivity.
3. **MCQ-only, shuffled-option LoRA.** Accuracy dropped from 34.33% to 28.00%, while 90.67% of predictions selected option C.
4. **Single-letter, shuffled-option LoRA.** Consistency increased from 7.67% to 30.00%, but accuracy declined from 36.67% to 33.67%.

We then explored **MedGemma-4B** (40.33% in an exploratory pilot) and **MedGemma-27B** (227/300, or 75.67%, on a development pilot). On the 30 RQ4 MCQs, MedGemma-27B correctly answered **22**, compared with **9** for FLAN-T5. These are smaller exploratory comparisons and **do not replace** the original 4,183-question Part I benchmark or isolate parameter count as the sole cause of improvement.

Model-specific documentation: [TF-IDF](docs/tfidf_baseline.md) · [Seq2Seq](docs/seq2seq.md) · [FLAN-T5 + LoRA](docs/flan_t5_lora.md) · [GPU training](docs/flan_t5_gpu_training.md).

## 5. Part II — Retrieval and conversational tutoring

### Final research architecture

```text
Medical MCQ + options
       |
       v
MedGemma-27B scores A / B / C / D
       |
       v
Predicted answer + student's follow-up + original MCQ
       |
       v
Construct retrieval query
       |
       v
MiniLM / E5-base-v2 / BioLORD-2023
       |
       v
Rank the shared 32,218-passage bank by cosine similarity
       |
       v
Retrieve top 5 passages
       |
       v
Optional confidence gate on top-1 similarity
       |                              |
   Accepted                       Rejected
       |                              |
       v                              v
Top 3 passages + context        Fixed fallback reply
       |
       v
MedGemma-27B generates a follow-up response
```

**Evaluation controls:** RQ3 constructs retrieval queries using the **gold MCQ answer** to isolate retriever performance. RQ4 uses **MedGemma's predicted answer** to test the complete tutor. The seven RQ4 conditions are no retrieval; three retrievers with confidence gating; and the same three with gating disabled. The frozen confidence thresholds were calibrated on development data rather than the test set.

### Retrievers and settings

| Setting | MiniLM | E5-base-v2 | BioLORD-2023 |
|---|---|---|---|
| Hugging Face model | `sentence-transformers/all-MiniLM-L6-v2` | `intfloat/e5-base-v2` | `FremyCompany/BioLORD-2023` |
| Input limit | 256 tokens | 512 tokens | 128 tokens |
| Query/document prefixes | None | `query: ` / `passage: ` | None |
| Frozen top-1 threshold | 0.673 | 0.867 | 0.648 |
| Passage truncation rate | 15.8% | 3.3% | 46.2% |

Thresholds were selected by maximising top-1 relevance F1 on **80** `internal_dev` follow-ups. They are stored in [configs/retrieval/gate_thresholds.json](configs/retrieval/gate_thresholds.json). The models embed the same passage bank separately; similarity scores and thresholds are **not interchangeable across models**.

### Building a new corpus and indexes

The following is a **new exploratory run**, not a claim to recreate the exact archived 2026 indexes. It uses distinct run identifiers so frozen artifacts are not overwritten.

```bash
# Requires data/splits from Section 3.
uv run python scripts/build_retrieval_corpus.py --run-id corpus-readme

# --corpus-run is REQUIRED by the index-builder CLI.
for m in minilm e5_base_v2 biolord_2023; do
  uv run python scripts/build_retrieval_index.py --method "$m" \
    --corpus-run outputs/retrieval/corpus/corpus-readme \
    --run-id index-readme --device cpu
done

# Run the already tracked evaluation queries through the new indexes.
uv run python scripts/run_retrieval.py --queries evaluation \
  --index-run index-readme --run-id retrieval-readme
```

`data/retrieval_eval/queries_eval.jsonl` and `queries_calibration.jsonl` are **already committed**. Do not rerun `make_followup_queries.py` on a normal checkout: it deliberately refuses to overwrite those frozen files. The same applies to `import_labels.py` when the final relevance judgments already exist.

The published RQ3 metric-generation command is:

```bash
uv run python scripts/evaluate_retrieval.py \
  --pool-id pool-20261020b --run-id rq3-reproduction
```

**Prerequisite:** This evaluator reads the exact ranking run referenced in the frozen judging-pool metadata. The relevant historical rankings and indexes must first be restored to their recorded paths. It will **not automatically score** the new `retrieval-readme` rankings against the old pooled judgments. For a genuinely new experiment, build and label a new judging pool as described in [the labelling guide](docs/retrieval_labelling_guideline.md).

### RQ3 results

Of 60 evaluation follow-ups, **24** had at least one known relevant passage in the pooled candidate set. The other 36 included nine intentionally off-topic queries and 27 for which none of the pooled passages was labelled relevant. The figures below are conditional on those **24 scorable follow-ups**, not all 60 queries.

| Retriever | Hit@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| MiniLM | **0.92** | **0.80** | **0.66** |
| E5-base-v2 | 0.75 | 0.58 | 0.51 |
| BioLORD-2023 | 0.46 | 0.32 | 0.38 |

MiniLM outperformed BioLORD on Hit@5 (**p = 0.004**), Recall@5 (**p < 0.001**) and MRR@5 (**p = 0.011**) in the reported paired tests. MiniLM did not significantly outperform E5. BioLORD's short 128-token limit, which truncated 46.2% of passages, is a plausible contributor to its weaker performance but was not isolated as the sole cause.

### RQ4 evaluation and results

RQ4 compares the same **40 follow-up turns** across **seven conditions**. Each generated response is rated **0, 1 or 2** for medical correctness, absence of unsupported claims, evidence grounding (retrieval conditions only) and second-turn consistency. Fixed refusals are counted separately, not scored as generated responses. LLM assistance was used for relevance labels and reply scoring, with manual checks; this is distinct from locally hosted research-model inference.

**When confidence gating is disabled** (40 replies per condition):

| System | Mean correctness (0–2) | Mean grounding (0–2) |
|---|---:|---:|
| No retrieval | **1.00** | Not applicable |
| MiniLM | 0.95 | 0.70 |
| E5-base-v2 | 0.95 | 0.68 |
| BioLORD-2023 | 0.93 | 0.43 |

The differences in correctness relative to no retrieval were **not statistically significant** (paired p = 0.78, 0.73 and 0.25, respectively). These figures are **rubric means, not percentage accuracies**.

**When confidence gating is enabled** (40 opportunities per retriever):

| Retriever | Replies issued | Fixed refusals | Correctness on accepted replies | Grounding on accepted replies |
|---|---:|---:|---:|---:|
| MiniLM | 12 | 28 | 1.17 | 1.00 |
| E5-base-v2 | 7 | 33 | 1.29 | 1.43 |
| BioLORD-2023 | 8 | 32 | 1.63 | 1.13 |

Higher scores on the few accepted replies must not be compared directly with the all-answer conditions. The system was much more selective, and most follow-ups received a fixed refusal.

The MedGemma tutor was substantially more correct than the earlier FLAN-based tutor on the same 40 follow-ups, but that comparison changed the MCQ answerer, retrieval query and reply generator together. When the initial MedGemma MCQ answer was correct, subsequent response correctness averaged **1.39**; when it was wrong, the mean was **0.08**. The final finding is therefore that **better evidence retrieval did not automatically produce evidence-faithful follow-up explanations**.

### Re-running the earlier FLAN-based RQ4 pipeline

The following commands reproduce the **workflow** in the public repository for the earlier FLAN-based tutor; they do **not** reproduce the final MedGemma RQ4 results. They require the historical FLAN adapter and original retrieval corpus/index runs to be restored to the indicated paths:

```bash
uv run python scripts/run_conversation_eval.py \
  --run-id flan-rq4-reproduction \
  --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best \
  --corpus-run outputs/retrieval/corpus/20261016-142815 \
  --index-run full-20261016 \
  --reply-without-adapter --clean-evidence \
  --no-repeat-ngram-size 3 --gate-off

uv run python scripts/export_rubric_sheets.py --run-id flan-rq4-reproduction
```

The exported rubric sheets are **not automatically graded**. Complete the blind correctness and grounding sheets using [docs/rq4_grading_guide.md](docs/rq4_grading_guide.md) and put the completed files in `data/retrieval_eval/rq4/flan-rq4-reproduction/`, retaining the generated rubric key. Then compute the metrics:

```bash
uv run python scripts/evaluate_conversation.py --run-id flan-rq4-reproduction
```

This writes scored summaries for the earlier FLAN pipeline. For the published **MedGemma** comparison, use the separately archived controlled RQ4 package, frozen predictions and matching grading artifacts; its exact launcher is not present on `main`.

## 6. Running the included FLAN-based tutor and frontend

The **included** FastAPI demo uses the earlier FLAN-T5 + LoRA MCQ answerer and plain FLAN-T5-base reply model. It is useful for demonstrating the repository's integration, but it **is not the final MedGemma RQ4 research system**.

First restore the corresponding checkpoint, retrieval corpus/index runs and Hugging Face caches. Check the required artifact paths:

```bash
uv run python scripts/serve_demo.py \
  --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best \
  --check-artifacts
```

Then launch the backend:

```bash
uv run python scripts/serve_demo.py \
  --adapter-dir checkpoints/flan_t5_lora/flan-base-full-seed42/best
```

In a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the local Vite address shown in the terminal. The frontend supports **Practice MCQ** with follow-up chat and evidence display, plus a browser-local **Flashcards** study mode. The default development proxy forwards `/api` to `http://127.0.0.1:8000`. When changing the backend origin, copy `frontend/.env.example` to `.env.local`, set `CLEARMED_BACKEND_URL`, and restart Vite.

After the backend is ready, check `http://127.0.0.1:8000/api/health`. See [docs/demo.md](docs/demo.md) and [frontend/README.md](frontend/README.md) for integration checks, health routes, gate settings, API contracts and troubleshooting.

**Separate MedGemma demo:** The group also prepared `ClearMed_MedGemma27B_MiniLM_Demo.zip` with a 4-bit MedGemma server, MiniLM index and Slurm launcher. It is **not included in this `main` branch**. Its demo index and gate-off settings are not the complete three-retriever RQ4 experiment. The standalone server has no authentication and must not be exposed publicly; follow the separate GPU deployment instructions in [docs/demo.md](docs/demo.md).

## 7. Testing

From the repository root:

```bash
uv sync --frozen
uv run pytest -v
uv run ruff check .
```

From `frontend/`:

```bash
npm ci
npm run test
npm run lint
npm run build
npm run format:check
```

Unit/integration tests use synthetic fixtures and do not load the historical trained weights. Passing them checks source-code behaviour, **not** held-out accuracy, retrieval quality or medical correctness. A complete end-to-end demo additionally requires the real checkpoints, corpus, indexes, pretrained-model caches and a working backend.

## 8. Artifacts and reproducibility

**The repository contains the code and selected small frozen evaluation inputs, not every artifact needed to reproduce every reported table from scratch.** In particular, `checkpoints/` and `outputs/` are git-ignored, and the historical Part I held-out evaluator and final MedGemma RQ4 GPU launcher are maintained separately. The scripts under `scripts/run_conversation_eval.py` and `scripts/serve_demo.py` are **FLAN-based**, so they must **not** be presented as reproducing the MedGemma RQ4 scores.

| Reproduction target | What is available in `main` | Additional requirements |
|---|---|---|
| Installation and tests | Python project, lockfile, tests and frontend | Python/uv; Node/npm for frontend |
| Dataset partitioning | Seed-42 split generator and audit logic | MedMCQA download/cache |
| Part I training | TF-IDF, Seq2Seq and FLAN-T5 + LoRA scripts | Compute, model downloads and historical configuration/artifacts for exact reruns |
| Part I official-validation metrics | Reported results and development evaluators | Frozen checkpoints, archived `heldout-v2` evaluator, manifests and prediction files |
| RQ3 | Retriever code, frozen queries, pool labels and thresholds | Original corpus, indexes and ranking-run files for exact historical scoring |
| Earlier FLAN RQ4 | Conversation pipeline, rubric scripts and committed grading sheets | Frozen FLAN adapter and retrieval artifacts; review source run IDs |
| **Final MedGemma RQ4** | Method and reported results in the final report | **Separate controlled GPU launcher, model access, frozen predictions, grading artifacts and run manifest** |
| Live FLAN demo | FastAPI + React code | Adapter, corpus, indexes and caches |
| MedGemma + MiniLM demo | Frontend compatibility and deployment guide | Separately distributed demo ZIP and GPU node |

For a **complete course-submission reproduction package**, provide the following alongside the report and repository: the exact Git commit (`git rev-parse HEAD`), dependency/driver information, split and dataset manifests, original frozen model checkpoints, retrieval corpus and all three indexes, held-out Part I evaluation scripts and predictions, RQ3 ranking files, MedGemma RQ4 launcher and prompt/configuration snapshot, all seven RQ4 condition outputs, human/LLM grading records and any execution logs needed to trace the reported results. Keep the frozen originals read-only and write new experiments under new run IDs.

The recorded reference run IDs are:

| Purpose | Identifier |
|---|---|
| Retrieval corpus | `20261016-142815` |
| Retrieval indexes | `full-20261016` |
| RQ3 evaluation | `rq3-20261101` |
| Earlier FLAN-based RQ4 | `rq4-20261104` |
| Final MedGemma RQ4 | `rq4-medgemma27b-end2end-20261108` |

**Verification status:** The repository paths and command-line arguments above have been checked against the current `main` branch source. They have **not** all been executed in a fresh environment during preparation of this README. The final MedGemma RQ4 command cannot be verified from this branch because its standalone GPU implementation is absent. Publish the companion execution instructions and artifacts before claiming complete one-command reproduction.

## 9. Research limitations

- **MCQ prediction quality:** The original Part I models reached only 31–33% held-out accuracy. An incorrect predicted option can contaminate both retrieval queries and the follow-up explanation.
- **Evaluation scope:** RQ3's main retrieval metrics condition on 24 scorable queries out of 60; RQ4 has 40 templated follow-ups and is not a real-student user study.
- **Grounding:** MedGemma can answer from pretrained knowledge, ignore passages, or misquote them; retrieval did not produce a detectable improvement in the tested correctness metric.
- **Selective refusals:** Frozen confidence thresholds rejected most RQ4 queries. Scores on the small accepted subset must be reported with coverage.
- **Annotation:** Relevance labelling and conversational scoring used LLM assistance with human checks, not comprehensive independent clinician adjudication.
- **Causal interpretation:** The FLAN-to-MedGemma system comparison changes multiple components simultaneously. The experiments do not identify model size alone as the cause of the gains.

Future work includes evidence-supported correction of wrong MCQ answers, stricter passage attribution and citation checking, longer and better-controlled follow-up generation, re-calibration using predicted-answer queries, and evaluation with authentic student questions and qualified medical reviewers.

## 10. Team, documentation and license

**G1 Group 6:** Reiwen Ing, Regan Tan, Shalyn Eng, Win Lei Thawdar and Joel Foo.

Additional technical documentation is in [docs/README.md](docs/README.md). The dataset is from [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa); pretrained models remain subject to their respective model licenses and access conditions.

Source code is provided under the [MIT License](LICENSE). The license does not imply that generated medical content has been clinically validated.
