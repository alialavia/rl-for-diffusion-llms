#!/usr/bin/env bash
# Train ESPO on one node with 8 H100s.
#   scripts/train_8xh100.sh sudoku_llada                      # configs/sudoku_llada.yaml
#   scripts/train_8xh100.sh countdown_llada train.max_steps=2000 espo.beta=0.005
set -euo pipefail
CONFIG=${1:-sudoku_llada}; shift || true
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
EXTRA=()
if [ $# -gt 0 ]; then EXTRA=(--set "$@"); fi
exec torchrun --standalone --nproc_per_node=8 -m espo.train \
    --config "configs/${CONFIG}.yaml" "${EXTRA[@]+"${EXTRA[@]}"}"
