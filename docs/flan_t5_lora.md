# FLAN-T5 + LoRA (Part I)

This is ClearMed's pretrained text-to-text Part I model for RQ1 (MCQ accuracy)
and RQ2 (explanation quality).  Unlike the scratch GRU Seq2Seq baseline, it
starts from a pretrained FLAN-T5 encoder-decoder and updates only LoRA adapter
weights.

## Data contract and safeguards

Inputs use the canonical question plus labelled A--D options and the fixed
instruction to answer and explain. They never read `cop`, `exp`, or any
target-derived data. Training targets are `Answer: X` and append
`Explanation: ...` only where an expert explanation exists. Source formatting
uses the canonical helper when it fits. For a long source it allocates tokenizer
space across the question and every option; it rejects limits that cannot retain
all four options instead of silently dropping one.

## Parsing and evaluation

Both generative Part I models share one conservative answer parser. It accepts
the requested `Answer: C` form, including `Answer: C` followed by an explicit
`Explanation: ...`, and also accepts an otherwise standalone `C` or `C.` as an
unambiguous MCQ answer. T5's tokenizer decodes the target newline as a space,
so `Answer: C Explanation: ...` is treated as the same structured form. It rejects strings such as `A or B`, `B12.`, `ASD`,
`DPT`, `CLL`, and `A beta C. III`; invalid outputs remain `MALFORMED` and are
never repaired from a gold label.

Prediction records retain the raw generation and separately record whether it
was compliant with the requested answer-and-explanation layout. Thus a valid
standalone answer can count toward MCQ accuracy without being mistaken for an
explanation. Explanation overlap and future factual-quality review use only
explicitly generated explanation text; no explanation is inferred or filled in.

The scripts load only `part1_train` and `internal_dev`, reject row-ID and
normalised-question overlap, and verify the full canonical partitions against
`data/splits/manifest.json` before any deterministic seed-42 subsetting.

## Model and artifacts

The local default is `google/flan-t5-small`; use `google/flan-t5-base` for the
planned GPU study. PEFT uses Seq2Seq LoRA with `r=8`, `alpha=16`, dropout
`0.05`, no bias, and verified `q`/`v` target modules. Base weights are frozen;
only adapter weights train. Each run saves a reloadable adapter and tokenizer to
`checkpoints/flan_t5_lora/<run_id>/`, while configuration, metrics, history,
and JSONL predictions live in `outputs/flan_t5_lora/<run_id>/`.

## Commands

Install the locked environment first:

```cmd
uv sync
```

After generating the canonical partitions, run a tiny local smoke experiment:

```cmd
uv run python scripts\train_flan_t5_lora.py --model-name google/flan-t5-small --max-train-questions 8 --max-dev-questions 4 --batch-size 2 --max-steps 1 --device cpu --run-id flan-smoke-v1
```

Evaluate the saved adapter without retraining:

```cmd
uv run python scripts\evaluate_flan_t5_lora.py --checkpoint-dir checkpoints\flan_t5_lora\flan-smoke-v1 --output-dir outputs\flan_t5_lora\flan-smoke-v1\eval-internal-dev --device cpu
```

These are implementation smoke checks, not final research results. A real run
requires a locally cached or downloadable checkpoint; the project still needs a
documented full GPU fine-tune, held-out protocol, BERTScore, and blind factual
quality assessment before making RQ1/RQ2 claims.
