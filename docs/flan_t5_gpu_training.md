# FLAN-T5 + LoRA GPU training on SMU Origami

This is the executable workflow for ClearMed Part I. It prepares a full
`google/flan-t5-base` LoRA experiment but does **not** claim that it will fit a
particular GPU. Confirm your own allocation with `myinfo`, run the smoke job,
then run the short feasibility test before committing to full training.

The completed SMU Origami feasibility run used an RTX 3090 (24 GB), BF16,
batch size 2, gradient accumulation 4, and 20 optimizer steps. Its active
training-loop time was 16.179 seconds; teacher-forced validation loss improved
from 3.3301 to 3.3235, and both `best` and `latest` checkpoints were saved.
This establishes feasibility for that exact bounded configuration only, not a
full-training result.

The checked-in Slurm template is [scripts/slurm/flan_t5_lora.sbatch](../scripts/slurm/flan_t5_lora.sbatch).
It does not contain account, partition, QOS, memory, or wall-time guesses.

## 1. Connect and inspect your allocation

On Windows Command Prompt (not PowerShell), connect through the SMU-authorised
VPN, then SSH using the host supplied by SMU:

```cmd
ssh <SMU_USER>@<ORIGAMI_HOST>
```

On the Origami login node, run:

```bash
myinfo
```

Record the permitted partition, account, QOS, CPU count, memory, wall-time and
GPU type/memory. The exact values in `myinfo` are the source of truth. Do not
start model training on the login node.

## 2. Place the repository and data on persistent storage

From Windows Command Prompt, copy a normal working tree (not checkpoints or
outputs) if the repository is not already available on the cluster:

```cmd
scp -r C:\Users\Regan\Desktop\ClearMed <SMU_USER>@<ORIGAMI_HOST>:~/ClearMed
```

On Linux, replace the path below with persistent project storage if your home
directory is not suitable:

```bash
cd ~/ClearMed
mkdir -p outputs/slurm checkpoints/flan_t5_lora outputs/flan_t5_lora
uv sync --frozen
uv run python scripts/generate_splits.py --seed 42 --output-dir data/splits
```

Do not regenerate `data/splits` if the verified canonical partitions are
already present. The training and evaluation CLIs verify their manifest,
fingerprints, row IDs, and normalised-question isolation before sampling.

Check what the environment can see. Load only modules actually available on
your cluster; the template deliberately does not guess their names.

```bash
module avail 2>&1 | less
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If this reports a CPU-only PyTorch build, create or install an environment with
a CUDA-compatible PyTorch wheel **using the cluster's supported guidance**,
then rerun `uv sync --frozen`/the check. Keep `uv.lock` unchanged unless the
team intentionally updates dependencies. Model downloads and Hugging Face cache
are retained under `model-cache/` by the job script, not `/tmp`.

## 3. Submit the GPU smoke test

Set the two module names only if `module avail` showed that they are necessary;
otherwise omit those two exported variables. Replace every angle-bracket value
with the corresponding `myinfo` value. The memory and time values are Slurm
formats accepted for your allocation.

```bash
cd ~/ClearMed
mkdir -p outputs/slurm
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=smoke,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_PYTHON_MODULE=<PYTHON_MODULE>,CLEARMED_CUDA_MODULE=<CUDA_MODULE> scripts/slurm/flan_t5_lora.sbatch
```

If no modules are needed, use the shorter export list:

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=smoke,CLEARMED_PROJECT_DIR=$PWD scripts/slurm/flan_t5_lora.sbatch
```

The smoke test uses `google/flan-t5-small`, 16/16 deterministic train/dev
caps, two optimizer updates, CUDA-required mode, a validation event, and both
best/latest checkpoint creation. It is not a research result.

Inspect it without submitting another job:

```bash
myqueue
squeue --me
tail -n 100 outputs/slurm/clearmed-flan-<JOB_ID>.out
tail -n 100 outputs/slurm/clearmed-flan-<JOB_ID>.err
sacct --job=<JOB_ID>
```

The log must show `cuda_available=True`, a GPU name, finite losses and the
resulting run directory. You can inspect allocated memory/GPU usage while the
job runs by looking at the log's `nvidia-smi` output; run `nvidia-smi` yourself
only inside an allocated interactive/job context, not as a model-training
substitute on the login node.

## 4. Feasibility test for FLAN-T5-base

Before the full run, use the same template with a unique run ID and a bounded
base-model budget. This measures memory and throughput on your actual GPU:

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=train,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_RUN_ID=base-feasibility-seed42,CLEARMED_MODEL_NAME=google/flan-t5-base,CLEARMED_BATCH_SIZE=2,CLEARMED_GRADIENT_ACCUMULATION_STEPS=4,CLEARMED_MAX_STEPS=20,CLEARMED_EPOCHS=1,CLEARMED_VALIDATION_QUESTIONS=32,CLEARMED_VALIDATION_INTERVAL_STEPS=10,CLEARMED_CHECKPOINT_INTERVAL_STEPS=10 scripts/slurm/flan_t5_lora.sbatch
```

Use `outputs/flan_t5_lora/<run-id>/metrics.json` to choose a batch size,
accumulation count, and wall-time. It records CUDA peak allocation/reservation
and synchronized optimizer-step throughput. Do not infer feasibility from a
different GPU. If base OOMs, lower per-device batch size first while preserving
effective batch size through gradient accumulation; report the measured limit
rather than silently switching the research model.

## 5. Interpreting performance measurements and running a 200-step benchmark

`metrics.json` preserves the existing experiment metrics and adds
`training.performance` for the current invocation. Its timing fields are all
seconds:

- `total_elapsed_seconds` is the synchronized wall-clock time from just before
  optimizer updates through the final checkpoint save.
- `optimizer_step_elapsed_seconds` and
  `optimizer_step_throughput_steps_per_second` cover active training time only;
  validation and checkpoint work are excluded and reported separately as
  `validation_elapsed_seconds` and `checkpoint_elapsed_seconds`.
- `optimizer_steps_this_invocation` is especially useful after resume: it is
  not the run's cumulative global step count.

`training.performance.gpu_memory` uses **bytes**. Its
`peak_gpu_memory_allocated_bytes` is memory held by live PyTorch tensors, while
`peak_gpu_memory_reserved_bytes` includes PyTorch's CUDA caching allocator.
`total_device_vram_bytes` is the GPU's physical VRAM capacity; neither peak is
system RAM, and `system_ram_measured` is explicitly `false`. Peaks are reset
immediately before optimizer updates, after model/optimizer setup, and include
the persistent CUDA allocation present at that boundary.

The training log also emits a flushed `[flan_t5_lora]` progress line every 25
optimizer steps and at the bounded run's end. It reports cumulative optimizer
steps, active elapsed seconds, and synchronized optimizer-step throughput.

After the 20-step feasibility run, use this unique, non-overwriting 200-step
benchmark to obtain a steadier throughput figure. It remains a resource
benchmark, not a research result or a replacement for final evaluation:

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=train,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_RUN_ID=flan-base-benchmark-200-seed42,CLEARMED_MODEL_NAME=google/flan-t5-base,CLEARMED_BATCH_SIZE=2,CLEARMED_GRADIENT_ACCUMULATION_STEPS=4,CLEARMED_MAX_STEPS=200,CLEARMED_EPOCHS=1,CLEARMED_VALIDATION_QUESTIONS=32,CLEARMED_VALIDATION_INTERVAL_STEPS=100,CLEARMED_CHECKPOINT_INTERVAL_STEPS=100 scripts/slurm/flan_t5_lora.sbatch
```

## 6. Full fine-tuning

After the feasibility result, submit the unique full run. Omit
`CLEARMED_MAX_STEPS` to use the three-epoch budget, or set it explicitly if
your agreed protocol uses a step budget.

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=train,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_RUN_ID=flan-base-full-seed42,CLEARMED_MODEL_NAME=google/flan-t5-base,CLEARMED_BATCH_SIZE=<MEASURED_BATCH_SIZE>,CLEARMED_GRADIENT_ACCUMULATION_STEPS=<MEASURED_ACCUMULATION>,CLEARMED_EPOCHS=3,CLEARMED_PRECISION=auto,CLEARMED_VALIDATION_QUESTIONS=512,CLEARMED_VALIDATION_INTERVAL_STEPS=500,CLEARMED_CHECKPOINT_INTERVAL_STEPS=250 scripts/slurm/flan_t5_lora.sbatch
```

Training uses every eligible `part1_train` row and selects the adapter with the
lowest teacher-forced loss on a deterministic seed-42 subset of `internal_dev`
(512 rows by default). No generation occurs at every validation event. Artifacts
are intentionally split:

```text
checkpoints/flan_t5_lora/flan-base-full-seed42/best/     selected reloadable adapter
checkpoints/flan_t5_lora/flan-base-full-seed42/latest/   adapter + optimizer/scheduler/RNG state
outputs/flan_t5_lora/flan-base-full-seed42/              config, history, selection metrics
```

## 7. Resume a pre-empted or time-limited run

First ensure the last job wrote a complete `latest/training_state.pt`,
`latest/training_metadata.json`, and `latest/adapter/`. Resubmit the same run
ID and **identical model, LoRA, data, batch, accumulation, optimizer, and
precision settings**, adding `CLEARMED_RESUME=1`:

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=train,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_RUN_ID=flan-base-full-seed42,CLEARMED_MODEL_NAME=google/flan-t5-base,CLEARMED_BATCH_SIZE=<MEASURED_BATCH_SIZE>,CLEARMED_GRADIENT_ACCUMULATION_STEPS=<MEASURED_ACCUMULATION>,CLEARMED_EPOCHS=3,CLEARMED_PRECISION=auto,CLEARMED_VALIDATION_QUESTIONS=512,CLEARMED_VALIDATION_INTERVAL_STEPS=500,CLEARMED_CHECKPOINT_INTERVAL_STEPS=250,CLEARMED_RESUME=1 scripts/slurm/flan_t5_lora.sbatch
```

The resume path validates model/LoRA/data identity and training settings, then
continues the recorded optimizer step; it does not silently start a fresh run.
It restores Python, NumPy, PyTorch and CUDA RNG state where available. Exact
bitwise equivalence after a different job/GPU is not guaranteed, but batch order
and optimizer/scheduler progression are recorded and deterministic for the
same environment.

## 8. Final full internal-development evaluation

After full training, evaluate `best` independently. This is the only stage
that autoregressively generates the complete `internal_dev` predictions. Add
`CLEARMED_BERTSCORE=1` only after the normal evaluation completes, because it
adds a costly model-based metric.

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> --qos=<QOS> --cpus-per-task=<CPUS> --mem=<MEMORY> --time=<WALLTIME> --export=ALL,CLEARMED_MODE=evaluate,CLEARMED_PROJECT_DIR=$PWD,CLEARMED_CHECKPOINT_DIR=$PWD/checkpoints/flan_t5_lora/flan-base-full-seed42/best,CLEARMED_EVAL_OUTPUT_DIR=$PWD/outputs/flan_t5_lora/flan-base-full-seed42/eval-internal-dev,CLEARMED_EVAL_BATCH_SIZE=<MEASURED_EVAL_BATCH_SIZE>,CLEARMED_MAX_NEW_TOKENS=128 scripts/slurm/flan_t5_lora.sbatch
```

For BERTScore, submit an otherwise identical **new** output directory with
`CLEARMED_BERTSCORE=1`; output directories are intentionally non-overwriting.

The final directory contains `final_metrics.json`, `explanation_metrics.json`,
`predictions.jsonl`, and `errors.jsonl`. Share the selected `best/` adapter,
run `config.json`, training history, final metrics, explanation metrics, and
predictions with teammates. `final_metrics.json` contains accuracy, correct/
incorrect counts, malformed rate, subject accuracy, answer distribution,
confusion matrix, and strict-format parse distribution. Explanation outputs
record the references, generated text, ROUGE-L, optional BERTScore aggregate,
non-empty/format rates, EOS and length-limit rates, and repetition diagnostics.

## Troubleshooting

- **CUDA missing:** check the job log, `module list`, and the required CUDA
  module. The job exits rather than falling back to CPU.
- **Pending job / invalid allocation:** inspect `squeue --me` and `sacct`; use
  the partition/account/QOS reported by `myinfo`, not values copied from this
  guide.
- **CUDA OOM:** reduce `CLEARMED_BATCH_SIZE`, increase accumulation to preserve
  effective batch size, then repeat the bounded feasibility run.
- **Time limit or pre-emption:** inspect `sacct --job=<JOB_ID>`, confirm
  `latest/` is complete, then resubmit with `CLEARMED_RESUME=1`.
- **Dependency/model download failure:** verify `uv sync --frozen`, persistent
  write access to `.uv-cache` and `model-cache`, and cluster network/cache
  policy. Do not put Hugging Face tokens in the Slurm command or logs.
- **Missing partitions or manifest mismatch:** regenerate canonical partitions
  only in a new/approved location, or restore the verified `data/splits` files.
- **Malformed or empty outputs:** retain the reported results. Inspect
  `predictions.jsonl`, EOS/truncation diagnostics and `errors.jsonl`; do not
  alter the strict parser to repair predictions.
