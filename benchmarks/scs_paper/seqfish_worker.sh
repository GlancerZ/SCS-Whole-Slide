#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
# Serialize GPU work behind the four-section liver comparison.
while [[ ! -f runs/SCS_paper_benchmark_v1/2107/genept_scale4/completed.json ]]; do
    if ! tmux -L scs-paper-train has-session -t benchmark 2>/dev/null; then
        echo 'Liver training stopped before completion; second-stage queue aborted.' >&2
        exit 1
    fi
    sleep 20
done
for tile in seqfish_rep1_fov0 seqfish_rep1_fov1; do
    test -f "runs/SCS_paper_benchmark_v1/$tile/prepared.json"
    source .venv-scs-torch/bin/activate
    for method in scs_reference raw_scale4 genept_scale4; do
        python -u benchmarks/scs_paper/train.py --root runs/SCS_paper_benchmark_v1 \
            --tile "$tile" --method "$method" --epochs 100 --batch-size 512 \
            > "runs/SCS_paper_benchmark_v1/$tile/$method.log" 2>&1
    done
    source .venv-scs/bin/activate
    python -u benchmarks/scs_paper/evaluate.py --root runs/SCS_paper_benchmark_v1 --tiles "$tile"
done
