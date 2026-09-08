#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source .venv-scs-torch/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
torchrun --standalone --nproc_per_node=1 benchmarks/scs_paper/stereo_whole_train.py \
  --root runs/Cellist_full_genept_v1/datasets/stereo_mouse_brain --epochs 100 --resume
