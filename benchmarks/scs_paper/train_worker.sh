#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
source .venv-scs-torch/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
python -m unittest benchmarks.scs_paper.test_train -v
for tile in 2104 2105 2106 2107; do
    test -f "runs/SCS_paper_benchmark_v1/$tile/prepared.json"
    for method in scs_reference raw_scale4 genept_scale4; do
        python -u benchmarks/scs_paper/train.py --root runs/SCS_paper_benchmark_v1 \
            --tile "$tile" --method "$method" --epochs 100 --batch-size 512 \
            > "runs/SCS_paper_benchmark_v1/$tile/$method.log" 2>&1
    done
done
