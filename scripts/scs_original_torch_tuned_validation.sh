#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
test "${SLURM_JOB_ID}" = "20305630"
export CUDA_VISIBLE_DEVICES=GPU-3d51d05e-eef7-c209-2519-7bd67ed19d82
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=4

python -B -u scripts/scs_original_torch_benchmark.py \
    --output runs/ST19_original_torch_tuned_fp32_bs256_20305630 \
    --gpu "$CUDA_VISIBLE_DEVICES" --epochs 100 --batch-size 256 \
    --inference-batch-size 256 --optimizer-mode foreach --input-mode gpu_cache

python -B -u scripts/scs_original_torch_benchmark.py \
    --output runs/ST19_original_torch_tuned_fp32_bs10_20305630 \
    --gpu "$CUDA_VISIBLE_DEVICES" --epochs 3 --batch-size 10 \
    --inference-batch-size 256 --optimizer-mode foreach --input-mode gpu_cache

python -B -u scripts/scs_original_torch_benchmark.py \
    --output runs/ST19_original_torch_tuned_fp32_bs64_20305630 \
    --gpu "$CUDA_VISIBLE_DEVICES" --tiles x11_y12 --epochs 100 --batch-size 64 \
    --inference-batch-size 256 --optimizer-mode foreach --input-mode gpu_cache
