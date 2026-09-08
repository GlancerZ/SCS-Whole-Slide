#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
source .venv-scs-torch/bin/activate
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for dataset in seqfish_rep1_fov0 seqfish_rep1_fov1; do
    data="runs/Cellist_full_genept_v1/datasets/${dataset}"
    while ! test -f "$data/prepared.json" || ! test -f "$data/molecule_ground_truth.npz"; do sleep 10; done
    if ! test -f "$data/genept_all_scale4/inference_completed.json"; then
        resume=()
        if test -f "$data/genept_all_scale4/latest.pt"; then resume=(--resume); fi
        torchrun --standalone --nproc_per_node=1 benchmarks/scs_paper/stereo_whole_train.py \
          --root "$data" --epochs 100 --batch-per-gpu 512 "${resume[@]}" \
          > "runs/Cellist_full_genept_v1/logs/genept_${dataset}.log" 2>&1
    fi
done
