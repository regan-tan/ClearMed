#!/usr/bin/env bash
# ClearMed FLAN-T5 + LoRA Slurm template.  Create outputs/slurm before sbatch.
#SBATCH --job-name=clearmed-flan
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
# Allocation-specific settings are intentionally supplied to sbatch after myinfo:
##SBATCH --partition=REPLACE_WITH_MYINFO_PARTITION
##SBATCH --account=REPLACE_WITH_MYINFO_ACCOUNT
##SBATCH --qos=REPLACE_WITH_MYINFO_QOS
##SBATCH --cpus-per-task=REPLACE_WITH_MYINFO_CPUS
##SBATCH --mem=REPLACE_WITH_MYINFO_MEMORY
##SBATCH --time=REPLACE_WITH_MYINFO_WALLTIME

set -Eeuo pipefail

: "${CLEARMED_MODE:?Set CLEARMED_MODE to smoke, train, or evaluate.}"
: "${CLEARMED_PROJECT_DIR:?Set CLEARMED_PROJECT_DIR to the Linux ClearMed checkout.}"

case "$CLEARMED_MODE" in smoke|train|evaluate) ;; *) echo "Invalid CLEARMED_MODE: $CLEARMED_MODE" >&2; exit 2;; esac
[[ -d "$CLEARMED_PROJECT_DIR" ]] || { echo "Missing project directory: $CLEARMED_PROJECT_DIR" >&2; exit 2; }
cd "$CLEARMED_PROJECT_DIR"
mkdir -p outputs/slurm .uv-cache model-cache
export UV_CACHE_DIR="$PWD/.uv-cache"
export HF_HOME="$PWD/model-cache"
export TOKENIZERS_PARALLELISM=false

# Module names differ by allocation.  Leave these unset only when Python/CUDA
# are already available in the job environment; never assume login-node state.
if [[ -n "${CLEARMED_PYTHON_MODULE:-}" ]]; then module load "$CLEARMED_PYTHON_MODULE"; fi
if [[ -n "${CLEARMED_CUDA_MODULE:-}" ]]; then module load "$CLEARMED_CUDA_MODULE"; fi
command -v uv >/dev/null || { echo "uv is unavailable; load the supported Python/uv module first." >&2; exit 2; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-unknown} mode=$CLEARMED_MODE"
module list 2>&1 || true
uv run python -c 'import torch; print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda={torch.version.cuda}"); assert torch.cuda.is_available(), "CUDA is not visible inside this allocated job."; print(f"gpu={torch.cuda.get_device_name(0)}")'
nvidia-smi

run_id="${CLEARMED_RUN_ID:-}"
case "$CLEARMED_MODE" in
  smoke)
    run_id="${run_id:-smoke-${SLURM_JOB_ID}}"
    uv run python scripts/train_flan_t5_lora.py \
      --model-name google/flan-t5-small --max-train-questions 16 --max-dev-questions 16 \
      --validation-questions 8 --batch-size 2 --gradient-accumulation-steps 1 \
      --max-steps 2 --epochs 1 --validation-interval-steps 1 --checkpoint-interval-steps 1 \
      --device cuda --require-cuda --precision auto --run-id "$run_id"
    ;;
  train)
    : "${run_id:?Set CLEARMED_RUN_ID to a unique full-experiment identifier.}"
    command=(uv run python scripts/train_flan_t5_lora.py
      --model-name "${CLEARMED_MODEL_NAME:-google/flan-t5-base}"
      --batch-size "${CLEARMED_BATCH_SIZE:-4}"
      --gradient-accumulation-steps "${CLEARMED_GRADIENT_ACCUMULATION_STEPS:-4}"
      --epochs "${CLEARMED_EPOCHS:-3}"
      --validation-questions "${CLEARMED_VALIDATION_QUESTIONS:-512}"
      --validation-interval-steps "${CLEARMED_VALIDATION_INTERVAL_STEPS:-500}"
      --checkpoint-interval-steps "${CLEARMED_CHECKPOINT_INTERVAL_STEPS:-250}"
      --device cuda --require-cuda --precision "${CLEARMED_PRECISION:-auto}" --run-id "$run_id")
    if [[ -n "${CLEARMED_MAX_STEPS:-}" ]]; then command+=(--max-steps "$CLEARMED_MAX_STEPS"); fi
    if [[ "${CLEARMED_RESUME:-0}" == "1" ]]; then command+=(--resume); fi
    "${command[@]}"
    ;;
  evaluate)
    : "${CLEARMED_CHECKPOINT_DIR:?Set CLEARMED_CHECKPOINT_DIR to the selected best adapter directory.}"
    : "${CLEARMED_EVAL_OUTPUT_DIR:?Set CLEARMED_EVAL_OUTPUT_DIR to a new persistent output directory.}"
    command=(uv run python scripts/evaluate_flan_t5_lora.py
      --checkpoint-dir "$CLEARMED_CHECKPOINT_DIR" --output-dir "$CLEARMED_EVAL_OUTPUT_DIR"
      --batch-size "${CLEARMED_EVAL_BATCH_SIZE:-8}" --max-new-tokens "${CLEARMED_MAX_NEW_TOKENS:-128}"
      --device cuda --require-cuda)
    if [[ "${CLEARMED_BERTSCORE:-0}" == "1" ]]; then command+=(--bertscore); fi
    "${command[@]}"
    ;;
esac
