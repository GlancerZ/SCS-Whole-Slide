#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
source .venv-scs/bin/activate
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MPLBACKEND=Agg
python -u benchmarks/scs_paper/prepare.py --root runs/SCS_paper_benchmark_v1
