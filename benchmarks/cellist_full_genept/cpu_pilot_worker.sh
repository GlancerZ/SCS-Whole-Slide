#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
test -z "${SLURM_JOB_GPUS:-}"
export CUDA_VISIBLE_DEVICES=""
source scripts/cellist_env.sh
export CELLIST_THREADS=4
for dataset in 2104 2105; do
    python -u benchmarks/cellist_full_genept/cellist.py --dataset "$dataset" --workers 1 \
      > "runs/Cellist_full_genept_v1/logs/cellist_paper_${dataset}.log" 2>&1
done
