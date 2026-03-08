#!/usr/bin/env bash
# Tokenize all parquets in base_data into process/output/*.bin
# Runs up to 3 processes in parallel; skips existing .bin files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$SCRIPT_DIR/output"
PARQUET_DIR="$HOME/.cache/nanochat/base_data"
PARALLELISM=4

mkdir -p "$OUTPUT_DIR"

pids=()

run_one() {
    local parquet="$1"
    local stem
    stem="$(basename "$parquet" .parquet)"
    local bin="$OUTPUT_DIR/${stem}.bin"
    local log="$OUTPUT_DIR/${stem}.log"

    if [[ -f "$bin" ]]; then
        echo "[skip] $stem (already exists)"
        return
    fi

    echo "[start] $stem -> $bin"
    (
        cd "$REPO_ROOT"
        source .venv/bin/activate
        python process/parquet_to_bin.py --parquet "$parquet" --out "$bin" 2>&1 | tee "$log"
    ) &
    pids+=($!)
}

wait_for_slot() {
    while [[ ${#pids[@]} -ge $PARALLELISM ]]; do
        # Wait for any one to finish
        local new_pids=()
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        pids=("${new_pids[@]}")
        if [[ ${#pids[@]} -ge $PARALLELISM ]]; then
            sleep 1
        fi
    done
}

for parquet in $(ls "$PARQUET_DIR"/*.parquet | sort); do
    wait_for_slot
    run_one "$parquet"
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "All done."
ls -lh "$OUTPUT_DIR"/*.bin 2>/dev/null | awk '{print $5, $9}'
