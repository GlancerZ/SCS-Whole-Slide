#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
while [[ ! -f runs/SCS_paper_benchmark_v1/seqfish_rep1_fov1/benchmark.json ]]; do
    if ! tmux -L scs-paper-seqfish has-session -t benchmark 2>/dev/null; then
        echo 'Primary benchmark stopped before completion; coverage queue aborted.' >&2
        exit 1
    fi
    sleep 20
done
source .venv-scs-torch/bin/activate
python -u benchmarks/scs_paper/train.py --root runs/SCS_paper_benchmark_v1 \
    --tile 2104 --method raw_matched_scale4 --epochs 100 --batch-size 512 \
    > runs/SCS_paper_benchmark_v1/2104/raw_matched_scale4.log 2>&1
source .venv-scs/bin/activate
python -u benchmarks/scs_paper/coverage_evaluate.py
