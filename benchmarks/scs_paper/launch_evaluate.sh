#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
source .venv-scs/bin/activate
tmux -L scs-paper-eval new-session -d -s benchmark 'source /home/glancerz/labwork/codex/TLS/.venv-scs/bin/activate && OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg python -u benchmarks/scs_paper/evaluate.py --root runs/SCS_paper_benchmark_v1 --watch > runs/SCS_paper_benchmark_v1/evaluate.log 2>&1'
while tmux -L scs-paper-eval has-session -t benchmark 2>/dev/null; do sleep 20; done
test -f runs/SCS_paper_benchmark_v1/2107/benchmark.json
