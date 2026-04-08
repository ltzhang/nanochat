#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

TOKENS_PER_RUN=100139008
NUM_ITERATIONS=191
DEVICE_BATCH_SIZE=16
MAX_SEQ_LEN=2048
WINDOW_PATTERN=L

run_case() {
    local name="$1"
    shift
    local log_path="result/${name}.log"
    echo "[$(date '+%F %T')] starting ${name}" | tee -a "$log_path"
    python -m scripts.base_train \
        --depth=12 \
        --run=dummy \
        --model-tag="${name}" \
        --window-pattern="${WINDOW_PATTERN}" \
        --device-batch-size="${DEVICE_BATCH_SIZE}" \
        --max-seq-len="${MAX_SEQ_LEN}" \
        --num-iterations="${NUM_ITERATIONS}" \
        --target-param-data-ratio=-1 \
        --eval-every=-1 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        "$@" >> "$log_path" 2>&1
    echo "[$(date '+%F %T')] finished ${name}" | tee -a "$log_path"
}

mkdir -p result

TOKENIZER_DIR="$HOME/.cache/nanochat/tokenizer"
if [[ ! -f "${TOKENIZER_DIR}/tokenizer.pkl" || ! -f "${TOKENIZER_DIR}/token_bytes.pt" ]]; then
    echo "[$(date '+%F %T')] tokenizer artifacts missing; training tokenizer" | tee result/tokenizer_train.log
    python -m scripts.tok_train >> result/tokenizer_train.log 2>&1
fi

run_case "d12_untied_100m"
run_case "d12_tie_2_10_k1_100m" --tie-layers-start=2 --tie-layers-end=10 --tie-layers-stride=1
run_case "d12_tie_2_10_k2_100m" --tie-layers-start=2 --tie-layers-end=10 --tie-layers-stride=2
run_case "d12_tie_2_10_k4_100m" --tie-layers-start=2 --tie-layers-end=10 --tie-layers-stride=4
run_case "d12_tie_2_10_k8_100m" --tie-layers-start=2 --tie-layers-end=10 --tie-layers-stride=8

{
    echo "d12 layer tying sweep summary"
    echo "date: $(date '+%F %T')"
    echo "tokens_per_run: ${TOKENS_PER_RUN}"
    echo "num_iterations: ${NUM_ITERATIONS}"
    echo "device_batch_size: ${DEVICE_BATCH_SIZE}"
    echo "max_seq_len: ${MAX_SEQ_LEN}"
    echo "window_pattern: ${WINDOW_PATTERN}"
    echo "total_batch_size: auto"
    echo
    printf "%-24s %-14s %-14s %-14s\n" "run" "first_loss" "last_loss" "best_loss"
    for log_path in result/d12_*_100m.log; do
        run_name="$(basename "$log_path" .log)"
        first_loss="$(grep -m1 -oP 'loss: \K[0-9.]+(?= \|)' "$log_path" || true)"
        last_loss="$(grep -oP 'loss: \K[0-9.]+(?= \|)' "$log_path" | tail -n1 || true)"
        best_loss="$(grep -oP 'loss: \K[0-9.]+(?= \|)' "$log_path" | sort -g | head -n1 || true)"
        printf "%-24s %-14s %-14s %-14s\n" "${run_name}" "${first_loss:-NA}" "${last_loss:-NA}" "${best_loss:-NA}"
    done
} | tee result/d12_tie_sweep_summary.txt
