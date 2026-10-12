#!/usr/bin/env bash
# Fast preflight on 8 GPUs before a multi-day run: 2 rollout phases (16 optimizer steps)
# with the real model and the full per-rank batch. Check metrics.jsonl for max_mem_gb,
# time_generate and time_rollout.
#   scripts/smoke_8xh100.sh [config=sudoku_llada] [extra overrides...]
set -euo pipefail
CONFIG=${1:-sudoku_llada}; shift || true
cd "$(dirname "$0")/.."
rm -rf "outputs/smoke_${CONFIG}"
exec scripts/train_8xh100.sh "$CONFIG" train.max_steps=16 train.save_steps=16 \
    train.n_synthetic=2000 train.output_dir="outputs/smoke_${CONFIG}" "$@"
