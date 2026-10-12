#!/usr/bin/env bash
# Evaluate a checkpoint on one node with 8 GPUs at generation lengths 128/256/512.
#   scripts/eval_8xh100.sh sudoku outputs/sudoku_llada/checkpoint-10000 [--test_file data.csv]
set -euo pipefail
TASK=$1; ADAPTER=${2:-}; shift 2 || shift $#
cd "$(dirname "$0")/.."
exec torchrun --standalone --nproc_per_node=8 -m espo.evaluate --task "$TASK" \
    ${ADAPTER:+--adapter "$ADAPTER"} "$@"
