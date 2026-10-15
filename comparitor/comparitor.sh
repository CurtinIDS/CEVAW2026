#!/bin/bash

set -x
set -e

NPROCESSES=4
COMPARITOR_PIDS=()

cleanup() {
    echo "Shutting down zombie comparitors..."
    for pid in "${COMPARITOR_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# --- Launch comparitor jobs: N per device, all running concurrently ---
for n in $(seq 0 $(($NPROCESSES - 1))); do
    CUDA_VISIBLE_DEVICES="${n}" PYTHONUNBUFFERED=1 uv run comparitor.py compare --rater llm --prompt prompts/emotionality.md \
        /mnt/data/cevaw/corpus_1998_to_2025_v8.parquet 2>&1 \
        | tee >( { head -n 1000 > "comparitor_${n}.log"; cat > /dev/null; } ) \
        | tail -n 1000 >> "comparitor_${n}.log" &
    COMPARITOR_PIDS+=($!)
done

echo "Launched ${n} comparitor jobs, waiting for completion..."

# Wait for every comparitor job individually so a mid-list failure
# under `set -e` isn't masked (plain `wait pid1 pid2 ...` only
# reports the LAST pid's exit status).
FAILED=0
for pid in "${COMPARITOR_PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "Comparitor job pid ${pid} failed" >&2
        FAILED=1
    fi
done

echo "All comparitor jobs finished."
exit "$FAILED"
