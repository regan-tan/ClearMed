# ClearMed — Coding Agent Instructions

> **Scope:** These instructions apply to the entire ClearMed repository and all AI coding agents, including Claude Code and Codex.

## 1. Mandatory Agent Rules

### 1.1 Read AGENTS.md Before Every Task

**All coding agents MUST read this entire file before planning, implementing, refactoring, testing, or reviewing any code.**

- Read it at the beginning of every new task.
- Re-read it after context resets.
- Follow its conventions throughout implementation.
- If unavailable, stop and request guidance instead of assuming conventions.

### 1.2 AGENTS.md Must NEVER Be Modified by Coding Agents

**CRITICAL: `AGENTS.md` is strictly human-owned and immutable to all coding agents.**

Agents must NEVER:

- Edit, rewrite, append to, or overwrite `AGENTS.md`.
- Delete, rename, move, or reformat it.
- Modify it as part of automated formatting, refactoring, or documentation updates.
- Create or modify additional nested `AGENTS.md` files.
- Stage or commit changes to any `AGENTS.md`.

**Only human maintainers may update this file manually.**

If an instruction is outdated, unclear, or contradictory, report the issue to the user. Suggest changes in chat, but NEVER modify this file.

### 1.3 Git Restrictions

Coding agents must not perform Git state-changing operations.

**Forbidden:** `git add`, `commit`, `push`, `pull`, `switch`, `checkout`, `merge`, `rebase`, `reset`, `restore`, `clean`, branch creation/deletion, or PR creation/merging.

**Allowed:** Read-only inspection using `git status`, `git diff`, `git log`, etc.

Humans handle branching, staging, commits, pushes, and merges.

### 1.4 Preserve Existing Work

- Never overwrite or delete existing datasets, checkpoints, metrics, or experimental outputs.
- Do not retrain completed experiments unless explicitly instructed.
- Do not change research methodology without approval.
- Avoid unnecessary refactoring, folder migrations, dependency changes, and cosmetic formatting.
- Inspect relevant code and tests before making changes.
- Prefer minimal, maintainable implementations over unnecessary abstractions.

---

## 2. Project Overview

**ClearMed Study** is an NLP research project investigating medical multiple-choice question answering, explanation generation, evidence retrieval, and grounded conversational tutoring.

### Part I — Medical QA and Explanation Generation

| Research Question | Methods |
|---|---|
| RQ1: MCQ accuracy | TF-IDF + Logistic Regression, Scratch Seq2Seq, FLAN-T5 + LoRA |
| RQ2: Explanation quality | Scratch Seq2Seq vs FLAN-T5 + LoRA |

### Part II — Retrieval and Grounded Conversation

| Research Question | Methods |
|---|---|
| RQ3: Medical evidence retrieval | MiniLM, E5-base-v2, BioLORD-2023 |
| RQ4: Impact of retrieval on grounded conversation | RAG and appropriate non-retrieval comparison conditions |

Keep the two research parts logically separate while sharing common utilities where appropriate.

**Completed baseline:** TF-IDF + Logistic Regression achieved **39.84% accuracy on 9,141 internal-development MCQs**. This is not official held-out test performance.

---

## 3. Repository Structure

Follow the existing Python `src` layout and extend it incrementally.

```text
ClearMed/
├── AGENTS.md
├── src/clearmed/
│   ├── data/           # Shared loading, preprocessing, splitting
│   ├── baselines/      # Classical ML (TF-IDF)
│   ├── models/         # Neural architectures
│   ├── training/       # Model-specific training logic
│   ├── evaluation/     # Shared metrics and evaluation
│   ├── retrieval/      # Embedding and retrieval implementations
│   ├── conversation/   # Grounded conversation logic
│   └── app/            # Backend/API integration
├── scripts/            # Thin CLI entrypoints
├── configs/            # Tracked experiment configurations
├── tests/              # Automated tests
├── frontend/           # React + TypeScript + Vite
├── data/               # Raw/processed datasets and splits
├── checkpoints/        # Saved model artifacts
├── outputs/            # Predictions, metrics, analysis
├── docs/               # Research and architecture documentation
├── pyproject.toml
└── uv.lock
```

**Important conventions:**

- `data/` contains actual dataset files.
- `src/clearmed/data/` contains Python data-processing code.
- These two directories are NOT duplicates and must never be merged.
- `baselines/` contains classical ML baselines.
- `models/` contains neural model architectures.
- `training/` contains training loops, optimisation, and checkpoint logic.
- `evaluation/` contains reusable metrics across implementations.
- `retrieval/` contains shared retrieval logic and model-specific embedding adapters.
- `scripts/` contains thin CLI entrypoints rather than duplicated model logic.
- `configs/` contains reproducible, version-controlled configurations.

**Do not create empty placeholder directories or restructure existing modules unnecessarily.** Add components only when needed, preserving existing interfaces and imports.

### Standard Method Identifiers

Use consistent names across code, configuration, artifact paths, and reports:

- `tfidf`
- `seq2seq`
- `flan_t5_lora`
- `minilm`
- `e5_base_v2`
- `biolord_2023`

---

## 4. Dataset and Data Leakage Rules

Use the existing reproducible MedMCQA partitions.

| Partition | Size | Purpose |
|---|---:|---|
| `part1_train` | 137,117 | Part I model training |
| `internal_dev` | 9,141 | Development evaluation and model selection |
| `retrieval_kb` | 36,564 | Part II evidence retrieval |
| Official validation/test | Protected | Held-out evaluation under an agreed protocol |

### Mandatory Safeguards

1. Never create new random data splits independently for each model.
2. Never use `internal_dev` for model weight fitting or vocabulary construction.
3. Never use `retrieval_kb` for Part I supervised model training.
4. Prevent train/dev overlap using both row IDs and conservatively normalised questions.
5. Verify complete canonical partitions against `data/splits/manifest.json` before optional sampling/subsetting.
6. Identify canonical paths independently of user-supplied manifest overrides; reject conflicting overrides.
7. For custom datasets, retain overlap checks even when canonical manifest checks are inapplicable.
8. Preserve deterministic behaviour with default seed **42**, unless an experiment explicitly documents another seed.
9. Do not use official validation/test data for iterative hyperparameter optimisation.

**Inference leakage rule:** Gold answers (`cop`), reference explanations (`exp`), and target-derived information must never enter prediction-time features or model inputs.

Any preprocessing that learns parameters or a vocabulary from data must fit on the permitted training partition only.

---

## 5. Shared Model Input/Output Contracts

All Part I models must support comparable evaluation.

### MCQ Input

Provide the question and all four labelled options in stable A–D order.

```text
Question: ...
A. ...
B. ...
C. ...
D. ...
```

Preserve all four options. Do not silently truncate away options.

### Generative Output

Scratch Seq2Seq and FLAN-T5 + LoRA should use:

```text
Answer: C
Explanation: ...
```

Requirements:

- Generate an explicit A/B/C/D answer.
- Handle missing training explanations without fabricating targets.
- Never use gold/reference data during inference.
- Extract predictions using strict answer-label parsing.
- Count invalid/missing labels as `MALFORMED`.
- Never silently substitute the correct answer.

### Shared MCQ Evaluation

Reuse `src/clearmed/evaluation/qa.py`.

- Evaluate per original MCQ, not per binary candidate.
- Compare all Part I models on the same `internal_dev` partition.
- Report accuracy, sample count, malformed rate, and relevant breakdowns.
- Ensure confusion matrices and correct/incorrect totals reconcile.
- Preserve prediction-level error analysis.

### Explanation Evaluation

Treat explanation generation as a separate evaluation objective.

- Assess medical factual correctness, relevance, clarity, and unsupported claims.
- Reference-overlap metrics may supplement, but must not replace, factual-quality assessment.
- Use the same evaluation rubric and examples when comparing Seq2Seq and FLAN-T5 + LoRA.

### Part II Evaluation

- Use the same documented retrieval benchmark across MiniLM, E5-base-v2, and BioLORD-2023.
- Report Recall@K, MRR, nDCG, or similar metrics only when supported by appropriate relevance judgments.
- Prevent evaluation-answer leakage into retrieval indexes or queries.
- Record retrieved evidence, source identifiers, and grounding/citation assessment for conversational experiments.
- Keep retrieval quality and generated-answer quality as distinct measurements.

---

## 6. Model Implementation Standards

### TF-IDF Baseline

- Preserve the completed implementation, features, configuration, checkpoints, and experiment outputs.
- Do not modify its methodology simply to match neural model designs.

### Scratch Seq2Seq

- Implement using PyTorch without pretrained model weights.
- Recommended architecture: bidirectional GRU encoder, additive attention, autoregressive GRU decoder.
- Separate dataset processing, architecture, training, and inference.
- Verify padding masks, teacher forcing, generation independence, tensor dimensions, and checkpoint restoration.
- Test locally on small datasets before attempting full GPU training.

### FLAN-T5 + LoRA

- Use Hugging Face Transformers and PEFT.
- Record the exact pretrained checkpoint and tokenizer.
- Verify LoRA trainable/frozen parameters.
- Keep training, generation, and evaluation modular.
- Use compatible MCQ evaluation and explanation-output contracts.

### Embedding Models

- Implement model-specific adapters behind a common embedding/retrieval interface.
- Do not duplicate dataset handling, indexing, ranking, and metric calculation for each embedding model.
- Preserve each model's required query/document preprocessing.
- Record embedding model versions, dimensions, normalisation, similarity metric, and indexing configuration.

---

## 7. Experiment Tracking and Artifact Naming

Every significant experiment must be reproducible.

Use new run-specific paths such as:

```text
checkpoints/<model_id>/<run_id>/

outputs/<model_id>/<run_id>/
├── config.json
├── metrics.json
├── predictions.jsonl
└── training_history.json

outputs/retrieval/<method_id>/<run_id>/
```

Existing TF-IDF artifacts may retain their current paths. **Do not move or rewrite them just to satisfy these conventions.**

### Required Experiment Metadata

Record where applicable:

- Model identifier and exact model/tokeniser version
- Training hyperparameters
- Random seed
- Dataset partition identity and membership fingerprint
- Training environment/device
- Validation and checkpoint-selection criteria
- Evaluation metrics and sample counts
- Dependency/configuration versions needed for reproducibility

Use a new `run_id` rather than overwriting earlier runs.

### Prediction Records

Prefer JSONL, with one record per original MCQ and consistent field names:

```json
{
  "example_id": "...",
  "gold_label": "C",
  "predicted_label": "C",
  "raw_output": "Answer: C...",
  "generated_explanation": "..."
}
```

Gold/reference fields belong in training targets or evaluation exports only, never in inference inputs.

### Git Tracking

**Commit:**
- Source code
- Tests
- Documentation
- Lightweight configurations
- Dependency manifests and lockfiles when intentionally changed

**Do not commit:**
- Downloaded datasets or generated splits
- Model checkpoints
- Embedding indexes
- Training caches
- Large generated outputs
- Secrets or credentials

Check `.gitignore` before introducing new artifact types.

---

## 8. Collaboration and Branch Ownership

The team may implement models in parallel.

- Keep each model's implementation, training, and tests self-contained.
- Share existing data and evaluation infrastructure.
- Avoid modifying unrelated models.
- Avoid duplicate utilities serving identical purposes.
- Coordinate shared interface changes before implementation.
- Flag possible merge conflicts in `pyproject.toml`, `uv.lock`, data utilities, and evaluation modules.
- Do not change common output contracts independently in different branches.

In particular, Seq2Seq and FLAN-T5 + LoRA must remain independently trainable while producing compatible evaluation outputs.

Coding agents do not perform Git operations; humans manage branches and merges.

---

## 9. Testing and Verification

Every implementation must include meaningful tests, including failure cases.

### Python

Run when relevant:

```bash
uv run pytest -v
uv run ruff check .
git diff --check
```

### Frontend

Use existing npm scripts when relevant:

```bash
npm run test
npm run lint
npm run build
```

On Windows PowerShell, use `npm.cmd` if `npm.ps1` is blocked.

Known unrelated Prettier formatting debt should not trigger broad cosmetic rewrites.

### Model-Specific Tests

Cover relevant aspects such as:

- Data formatting and validation
- Dataset isolation and leakage prevention
- Tensor dimensions and attention masking
- Forward/backward propagation
- Inference independence from gold targets
- Deterministic predictions
- Tiny-batch overfitting
- Checkpoint save/load
- Prediction parsing and malformed outputs
- Evaluation integration

Use synthetic or small datasets for automated tests. Tests must not require the full MedMCQA dataset, a GPU, network access, or existing trained checkpoints unless explicitly designated as integration tests with those prerequisites.

Never launch full training merely to check correctness.

---

## 10. Frontend and Integration Rules

- Preserve the existing React + TypeScript + Vite frontend.
- Keep mock/preview APIs clearly separated from real backend integrations.
- Never represent mocked responses as real model predictions or retrieved evidence.
- Preserve user interactions, local persistence, and UI behaviour.
- Avoid mass component restructuring without adequate regression coverage.
- Integrate trained models through clearly defined backend interfaces when that phase begins.

---

## 11. Agent Workflow and Final Reporting

For every development task:

1. **Read `AGENTS.md` before doing anything else.**
2. Inspect the relevant implementation, dependencies, and tests.
3. Identify reusable utilities and shared contracts.
4. Implement the smallest correct solution.
5. Add or update appropriate tests.
6. Run relevant verification.
7. Inspect the final diff for unrelated changes.
8. Report precisely what changed and what remains unresolved.

### Required Final Report

Include:

- Summary of implementation
- Files created, modified, or removed
- Architectural/methodological decisions
- Exact test results and failed/skipped checks
- New artifact locations, if any
- Known limitations or follow-up work
- Current Git status

Explicitly confirm:

- No Git state-changing operations were performed.
- Existing research artifacts were not modified.
- **`AGENTS.md` was read and was NOT modified.**

### Conflicts and Exceptions

If instructions conflict with a requested implementation, existing experimental integrity, or another collaborator's changes, explain the conflict and request a human decision rather than silently deviating.

**This file must remain unchanged by all coding agents under all circumstances. Human maintainers alone are responsible for updating `AGENTS.md`.**
