#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
case "$1" in
    brain) export CUDA_VISIBLE_DEVICES=0; roots=(runs/SCS_stereo_whole_v1);;
    liver) export CUDA_VISIBLE_DEVICES=1; roots=(runs/SCS_stereo_whole_v1/liver/2104 runs/SCS_stereo_whole_v1/liver/2105 runs/SCS_stereo_whole_v1/liver/2106 runs/SCS_stereo_whole_v1/liver/2107);;
    *) exit 2;;
esac
for dataset_root in "${roots[@]}"; do
    while test ! -f "$dataset_root/prepared.json"; do
        if ! tmux -L scs-two-datasets has-session -t new-method 2>/dev/null; then
            echo "Preparation stopped without $dataset_root/prepared.json" >&2; exit 3
        fi
        sleep 10
    done
    if test "$1" = brain; then
        # This gate is written only after actual source/registration QA review.
        while test ! -f runs/SCS_stereo_whole_v1/alignment_reviewed.json; do sleep 10; done
    fi
    source .venv-scs-torch/bin/activate
    if test ! -f "$dataset_root/genept_all_scale4/inference_completed.json"; then
        resume=()
        if test -f "$dataset_root/genept_all_scale4/latest.pt"; then resume=(--resume); fi
        torchrun --standalone --nproc_per_node=1 benchmarks/scs_paper/stereo_whole_train.py \
            --root "$dataset_root" --epochs 100 "${resume[@]}" >> "$dataset_root/train.log" 2>&1
    fi
    source .venv-scs/bin/activate
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -u benchmarks/scs_paper/stereo_whole_segment.py --root "$dataset_root" --workers 4 \
        > "$dataset_root/postprocess.log" 2>&1
done
