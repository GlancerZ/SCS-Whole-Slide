#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
test -z "${SLURM_JOB_GPUS:-}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source .venv-scs/bin/activate
for dataset in seqfish_rep1_fov0 seqfish_rep1_fov1 2104 2105 2106 2107 stereo_mouse_brain; do
    python -u benchmarks/cellist_full_genept/prepare.py --dataset "$dataset" \
      > "runs/Cellist_full_genept_v1/logs/prepare_${dataset}.log" 2>&1
done
source scripts/cellist_env.sh
export CELLIST_THREADS=4
for dataset in 2104 2105 2106 2107 seqfish_rep1_fov0 seqfish_rep1_fov1 stereo_mouse_brain; do
    python -u benchmarks/cellist_full_genept/cellist.py --dataset "$dataset" --workers 2 \
      > "runs/Cellist_full_genept_v1/logs/cellist_${dataset}.log" 2>&1
done
