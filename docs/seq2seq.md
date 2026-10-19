# Scratch Seq2Seq Baseline (Part I)

A model that reads a medical MCQ and *writes* its answer and a
short explanation, learning everything from our training data alone (no
pretrained knowledge).


ClearMed compares three ways of answering medical MCQs, from simplest to most
powerful:

1. **TF-IDF** (done, 39.84%): matches words between the question and each
   option. It can pick an answer but cannot explain it.
2. **Scratch Seq2Seq** (this model): a neural network that *generates* text, so
   it can explain as well as answer. It starts from random weights.
3. **FLAN-T5 + LoRA**: a large pretrained model, fine-tuned on our data.

The Seq2Seq model sits in the middle. It tells us what a neural model can learn
from MedMCQA alone, before any pretraining is added:

- **RQ1 (accuracy):** does generating answers beat TF-IDF's word matching?
- **RQ2 (explanations):** how much better are FLAN-T5's explanations, i.e. how
  much is pretraining worth?

The method identifier used in code, configs and artifact paths is `seq2seq`.

## Architecture

All weights are randomly initialised; no pretrained embeddings or models are used.

```text
ENCODER (runs once)
  question + 4 options
        ↓  embedding
        ↓  bidirectional GRU
  encoder states  (one vector per input token)
        ↓  last forward + backward states → Linear + tanh
  decoder starting state

DECODER (runs once per output token)
  previous token  +  attention context
        ↓  GRU
        ↓  Linear → vocabulary scores
  next token  (first token = answer label, then the explanation)
```

**Attention at each decoder step:**

1. Score every input token against the current decoder state:
   `score_j = vᵀ · tanh(W_s · s + W_h · h_j)`
2. Set the scores of padding positions to −∞ so that they get zero weight.
3. `weights = softmax(scores)`
4. `context = Σ weights_j · h_j`, a weighted summary of the input.

| Component | Choice | Default |
|---|---|---|
| Embedding | one table shared by encoder/decoder (shared vocab) | 256 |
| Encoder | bidirectional GRU, packed sequences | hidden 256, 1 layer |
| Attention | additive (Bahdanau), mask from true lengths | attention dim 256 |
| Decoder | GRU on `[embedding; context]` | hidden 256 |
| Training | teacher forcing (ratio configurable), AdamW, grad-clip 1.0, dropout 0.2 | lr 1e-3, wd 0.01 |
| Inference | greedy decoding from the model's own predictions | max 130 tokens |

## Input / output contract

**Source** (built only from `question`, `opa`–`opd`; `cop`/`exp` are never read):

```text
<q> question tokens… <A> option tokens… <B> … <C> … <D> …
```

Each segment has its own budget (question 96, each option 24 tokens), so truncation
can shorten text but **can never remove an option marker**. Truncation rates are
recorded in `config.json` (`train_data_statistics`, `dev_data_statistics`).

**Target:** `<ans_X> [explanation tokens ≤128] <eos>`

- The answer is always the **first** generated token, as one dedicated vocabulary
  item (`<ans_A>`…`<ans_D>`).
- Rows without a usable `exp` get an answer-only target `<ans_X> <eos>`. No
  explanation text is fabricated. `--answer-only` trains every row this way (ablation).

**Rendered output** (shared contract, identical to FLAN-T5 + LoRA):

```text
Answer: C
Explanation: ...
```

Strict parsing (`^Answer: [ABCD]` at the start of the output). Anything else is
`MALFORMED` and is never replaced with the gold label. In this design, a malformed
output can only occur when the first generated token is not an answer token.


**Tokenisation / vocabulary:** lower-cased word + punctuation regex (no learned
parameters). The vocabulary is fit on the (possibly subsetted) `part1_train`
rows only: `min_freq=2`, at most 30k tokens, frequency ties broken alphabetically
(deterministic).

## Leakage and reproducibility safeguards

- Before any `--max-*` subsetting, the scripts reuse `scripts/train_tfidf.py`
  helpers for partition loading and **canonical manifest verification**, plus
  `assert_train_dev_disjoint` (row ID + normalised question).
- `internal_dev` is used only for model selection and evaluation, never for
  weights or vocabulary.
- Seed 42 by default, which controls initialisation and shuffling. Each epoch
  shuffles with `seed + epoch`, so a resumed run reproduces an uninterrupted one
  (tested).
- Run directories are never overwritten without `--overwrite`.

## Explanation data (`part1_train`, seed 42)

Produced by `uv run python scripts/inspect_seq2seq_data.py` (prints to the
terminal; add `--output outputs/seq2seq/data_inspection.json` to also save a copy).

| | Value |
|---|---|
| QA-valid rows | 137,117 |
| Rows with an explanation | 120,588 (**87.9%**) |
| Answer-only targets | 16,529 (12.1%) |
| Explanation length | median 64, mean 94, p90 203, p99 520 tokens |
| Explanations cut at 128 tokens | 22.2% (6.2% would be cut at 256) |
| Very short explanations (< 5 tokens) | 4,407 (3.2%) |
| Questions over 96 tokens | 0.5% (median 10 tokens) |
| Options over 24 tokens | 0.03% (median 2 tokens) |

**Decision:** keep the defaults (question 96, option 24, explanation 128).
The question and option limits almost never truncate. 128 keeps the full
explanation for ~78% of rows while keeping decoding cost manageable;
192–256 is an optional later experiment if GPU time allows.

## Commands

```bash
# local smoke run (CPU)
uv run python scripts/train_seq2seq.py --max-train-questions 2000 --max-dev-questions 300 \
    --max-epochs 3 --run-id smoke-2k
uv run python scripts/evaluate_seq2seq.py \
    --checkpoint-dir checkpoints/seq2seq/smoke-2k/best \
    --output-dir outputs/seq2seq/smoke-2k/eval

# full run (GPU cluster)
uv run python scripts/train_seq2seq.py --device cuda --batch-size 128 --run-id full-seed42

# resume an interrupted run (same args + --resume)
uv run python scripts/train_seq2seq.py --device cuda --batch-size 128 --run-id full-seed42 --resume

# evaluate the selected checkpoint on internal_dev
uv run python scripts/evaluate_seq2seq.py \
    --checkpoint-dir checkpoints/seq2seq/full-seed42/best \
    --output-dir outputs/seq2seq/full-seed42/eval-internal-dev
```

## Artifacts

```text
checkpoints/seq2seq/<run_id>/best/   model.pt, checkpoint_config.json (selected epoch)
checkpoints/seq2seq/<run_id>/last/   + training_state.pt (optimizer, history) for --resume
outputs/seq2seq/<run_id>/            config.json, training_history.json, training_summary.json
outputs/seq2seq/<run_id>/eval-*/     metrics.json, predictions.jsonl, errors.jsonl, subject_accuracy.csv
```

`predictions.jsonl` contains `gold_label` and `reference_explanation`, attached
**after** generation, for evaluation export only.

## Model selection

Best checkpoint = highest `internal_dev` accuracy, with lowest dev token loss as
the tie-break. Early stopping uses patience 3 on that criterion.

## Smoke-test results (pipeline checks, not research results)

### Real data: `smoke-2k` (CPU)

Trained on 2,000 `part1_train` MCQs for 3 epochs (vocab 10,810, 15.8M
parameters), selecting on 300 `internal_dev` MCQs, then evaluated on the full
`internal_dev` partition.

| Epoch | Train loss | Answer-token loss | Dev loss | Dev acc (300) |
|---|---|---|---|---|
| 1 | 7.07 | 3.25 | 5.97 | 22.7% |
| 2 | 5.96 | 1.45 | 5.72 | 25.3% |
| 3 | 5.55 | 1.42 | 5.58 | 27.0% |

Full `internal_dev` evaluation of the epoch-3 checkpoint:

| Metric | Value |
|---|---|
| MCQ accuracy | 26.97% (2,465 / 9,141) |
| Malformed outputs | 0% |
| ROUGE-L F1 (rows with a reference) | 0.028 |
| Empty generated explanations | ~11% |

Interpretation:

- The pipeline works end to end on real data: manifest verification → training →
  checkpointing → evaluation → exported predictions.
- Train and dev loss fall together, so the model is **underfitting** (too little
  training), not overfitting. Accuracy is at chance level (25%).
- The answer-token loss (1.42) is still at the guessing level (ln 4 ≈ 1.39): early
  learning goes to the far more numerous explanation tokens. Watch this in the
  full run, and try `--answer-loss-weight 3` if it stays near 1.39.
- Generated explanations are degenerate, e.g. `ans . is ' i . e . e . e ...`. The
  model has only learned that many references begin "Ans. is …" (see the 4,407
  very short explanations above).

These numbers are **not** comparable to TF-IDF's 39.84%. That comparison
requires the full GPU run.

### Synthetic data (development check)

On a 400-train / 80-dev synthetic set trained for 40 epochs, the model fitted
the training answers (answer loss ~0.3) but stayed at chance on dev, i.e. it
memorised rather than generalised. The full run's early stopping on dev accuracy
guards against this; a near-chance final accuracy would still be a legitimate
RQ1 finding, not a bug.

## Recommended first GPU configuration

`--batch-size 128 --embedding-dim 256 --hidden-dim 256 --num-layers 1 --dropout 0.3
--max-epochs 20 --patience 3`. Then try `--answer-loss-weight 3–5` as a single
ablation if dev accuracy lags. Per-epoch dev evaluation does greedy generation on
all 9,141 dev MCQs. If that is slow, evaluate with `--max-dev-questions` during
tuning and run the final evaluation on the full partition.

## Known limitations

- Word-level vocabulary: rare drug and disease names become `<unk>` (OOV rate is reported).
- Greedy decoding only (no beam search).
- ROUGE-L is reported as supplementary only. BERTScore needs a model download and is
  left to the shared RQ2 evaluation alongside blind human rating.
- Official validation/test are not touched.