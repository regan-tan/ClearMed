# TF-IDF + Logistic Regression Baseline (Part I)

ClearMed's `tfidf` baseline is the classical, non-generative reference model
for Part I medical multiple-choice question answering. It addresses RQ1 by
selecting one answer from four options and provides a lightweight comparison
point for the scratch Seq2Seq and FLAN-T5 + LoRA models. Unlike those
generative models, it does not generate explanations, use a language model, or
retrieve external evidence.

The model is intentionally simple and interpretable: it tests how far lexical
question--option associations can go before learned neural representations or
pretraining are introduced. Its reported metric is MCQ-level answer accuracy,
not explanation quality or pair-level binary accuracy.

## Dataset and data integrity

The implementation uses MedMCQA fields `id`, `question`, `opa`, `opb`, `opc`,
`opd`, and `cop`; `subject_name` and `topic_name` are retained for error
analysis. `cop` is the verified answer index `0`--`3`, corresponding to
`A`--`D`. A QA-valid example has a nonblank question, all four nonblank options,
and a valid `cop`. The optional expert explanation (`exp`) is neither required
for this baseline nor read during inference.

`scripts/generate_splits.py` partitions the official MedMCQA training split
with seed 42, grouping identical conservatively normalised question text before
assignment. The canonical partition sizes recorded in
`data/splits/manifest.json` are:

| Partition | Rows | TF-IDF use |
| --- | ---: | --- |
| `part1_train` | 137,117 | Fits the TF-IDF vocabulary and classifier. |
| `internal_dev` | 9,141 | Development evaluation and error analysis. |
| `retrieval_kb` | 36,564 | Reserved for Part II retrieval; not used here. |

The runner loads the already-saved partitions; it never downloads MedMCQA or
creates a fresh split. Before any optional deterministic subsetting, it rejects
training/development overlap by row ID and normalised question text. With the
canonical paths, it also verifies the complete partition row counts and
order-independent membership fingerprints against `data/splits/manifest.json`.
The canonical train and development paths must be supplied together, and a
conflicting manifest override is rejected. For two genuinely custom paths,
manifest verification is reported as inapplicable, but the overlap check still
runs.

The official MedMCQA validation and test splits are outside this experiment.
Consequently, `internal_dev` results are development results, not official
held-out validation or test performance.

## Model architecture and features

Each QA-valid MCQ is expanded into four question--candidate examples. During
training, the option selected by `cop` receives binary target `1`; the other
three receive `0`. The model therefore learns candidate correctness, while
evaluation remains at the original four-option MCQ level.

```text
question + option A  --\
question + option B  ---+--> shared TF-IDF vectorizer --> candidate features
question + option C  ---+--> candidate/question interaction --> logistic regression
question + option D  --/                                --> highest score wins
```

For each candidate, the sparse feature matrix concatenates:

- the candidate's TF-IDF vector;
- the element-wise product of the question and candidate TF-IDF vectors; and
- one cosine-similarity scalar derived from those vectors.

The vectorizer is fitted only on the training questions and option text. The
classifier receives no option position, other candidate options, gold answer,
expert explanation, or target-derived feature at prediction time. Inference
scores the four content-only candidates independently using the classifier's
probability for binary class `1`, then uses `numpy.argmax`; exact ties therefore
select the earliest option in A--D order.

These four values are independent binary correctness scores. They are not a
normalised four-class distribution and should not be interpreted as calibrated
medical confidence. For example, the selected option is simply the one with the
largest score:

| Option | Illustrative score |
| --- | ---: |
| A | 0.12 |
| B | 0.18 |
| C | 0.61 |
| D | 0.09 |

In this example, the prediction is `C`; the displayed scores need not sum to
one.

## Training procedure

The default `TfidfMCQConfig` and CLI settings used by the recorded full run are:

| Component | Setting |
| --- | --- |
| Text representation | Lowercased word TF-IDF |
| N-grams | Unigrams and bigrams (`(1, 2)`) |
| Maximum features | 50,000 |
| Minimum document frequency | 2 |
| Term-frequency scaling | Sublinear TF enabled |
| Classifier | `sklearn.linear_model.LogisticRegression` |
| Solver / class weighting | `liblinear` / `balanced` |
| Maximum iterations | 1,000 |
| Seed | 42 |
| Candidate-feature batch size | 2,000 MCQs |

The vocabulary is fitted before candidate-feature construction, exclusively on
QA-valid selected training rows. If a very small training subset leaves no
terms at the default minimum document frequency, the implementation retries
with `min_df=1` and records that effective value. It then fits a single binary
logistic-regression classifier on sparse candidate features. A full valid
training set of 137,117 MCQs yields 548,468 candidate pairs.

`--max-train-questions` and `--max-dev-questions` select deterministic
seeded MCQ subsets before four-way expansion. They are intended for diagnostics,
not comparable full-experiment results. Output and model directories must be
empty unless `--overwrite` is explicitly supplied.

## Evaluation methodology

`clearmed.evaluation.qa.evaluate_mcq_predictions` evaluates one predicted
label per QA-valid MCQ in `internal_dev`. It reports total, correct and
incorrect counts, MCQ accuracy, malformed-output count/rate, accuracy by gold
label, prediction distribution, and an A--D confusion matrix. It also writes
per-subject accuracy and a capped, deterministic sample of incorrect examples.

Malformed predictions are retained by the shared evaluator for comparability
with generative models. This baseline always chooses one of A--D when it
receives a valid four-option question, so malformed outputs are expected to be
zero rather than a generation-quality result. The primary metric is therefore
question-level MCQ accuracy; binary candidate-pair accuracy would be misleading
because three of every four training candidates are negative.

## Verified internal-development result

The available saved TF-IDF artifacts record the following seed-42 full-run
result. It used all 137,117 selected training MCQs and evaluated all 9,141
selected `internal_dev` MCQs.

| Metric | Recorded value |
| --- | ---: |
| MCQ accuracy | 39.84% (`0.3984246800`) |
| Correct / incorrect | 3,642 / 5,499 |
| Evaluated MCQs | 9,141 |
| TF-IDF vocabulary size | 50,000 |
| Training candidate pairs | 548,468 |
| Malformed predictions | 0 |

This is an internal-development measurement only. It exceeds the 25% expected
accuracy of uniform four-way guessing, but it is not a claim of clinical
reliability, official test performance, or generalisation beyond this setting.

## Reproduction

From the repository root, first create the canonical local partitions if they
are absent (see [the data guide](../data/README.md)):

```cmd
uv run python scripts\generate_splits.py --seed 42 --output-dir data\splits
```

Run a small CPU smoke experiment with separate artifact directories:

```cmd
uv run python scripts\train_tfidf.py --seed 42 --max-train-questions 200 --max-dev-questions 50 --model-dir checkpoints\tfidf-smoke --output-dir outputs\tfidf-smoke
```

Run the full CPU baseline using the canonical defaults:

```cmd
uv run python scripts\train_tfidf.py --seed 42
```

The script requires the Python dependencies locked for this repository,
including Hugging Face Datasets, NumPy, SciPy, scikit-learn, and joblib. Use
`uv sync` to install them. Inspect all current CLI options without training:

```cmd
uv run python scripts\train_tfidf.py --help
```

The CLI defaults are `data/splits/part1_train`, `data/splits/internal_dev`,
`data/splits/manifest.json`, `checkpoints/tfidf`, and `outputs/tfidf` for the
train partition, development partition, manifest, model, and outputs
respectively. It also exposes `--min-df`, `--max-features`, `--batch-size`,
`--error-limit`, and `--overwrite`.

## Saved artifacts

The default run writes the following local artifacts:

```text
checkpoints/tfidf/
  model.joblib          # fitted TfidfMCQBaseline (load only from trusted sources)
  model_config.json     # hyperparameters, effective min_df, vocabulary and training counts

outputs/tfidf/
  metrics.json          # metrics, environment, durations and provenance
  experiment_config.json # resolved CLI arguments
  subject_accuracy.csv  # subject-level accuracy breakdown
  errors.jsonl          # up to --error-limit incorrect MCQ records
```

The current runner reloads the saved model and verifies that it gives the same
prediction for the first development row before writing completion metadata.
`errors.jsonl` records the question ID, question, labelled option texts, gold
and predicted answers, candidate scores, subject, and topic for each retained
error. These artifacts are generated research outputs and are not recreated or
overwritten by documentation work.

## Limitations and interpretation

- **Lexical features, not clinical reasoning.** TF-IDF represents observed
  word and phrase patterns. The interaction and cosine features offer a simple
  question--candidate association signal but do not supply semantic or causal
  medical reasoning.
- **Training-vocabulary dependence.** Terms unseen during vectorizer fitting do
  not create new features, and paraphrases may share little sparse vocabulary.
- **Independent option scoring.** Candidates are ranked separately; the model
  does not jointly model relationships among distractors.
- **No explanations or dialogue.** The model only selects A--D. It cannot
  produce a rationale, answer a follow-up, or provide grounded citations.
- **Development-set scope.** Model selection or iteration informed by
  `internal_dev` can overfit development decisions. Future cross-model claims
  require the agreed untouched evaluation protocol.

The baseline is therefore best interpreted as a reproducible classical NLP
reference for RQ1, rather than as a medically reliable decision-support system.
