#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
test "${SLURM_JOB_ID}" = "20305630"
scs_bench_socket="scs-original-bench-20305630"
if tmux -L "$scs_bench_socket" has-session -t benchmark 2>/dev/null; then
    echo "Benchmark session already exists" >&2
    exit 1
fi
tmux -L "$scs_bench_socket" new-session -d -s benchmark \
    'cd /lustre09/project/6102157/glancerz/codex/TLS && source .venv-scs-torch/bin/activate && exec python -B -u scripts/scs_original_torch_benchmark.py --output runs/ST19_original_torch_benchmark_20305630_v2 --gpu GPU-3d51d05e-eef7-c209-2519-7bd67ed19d82 >> runs/ST19_original_torch_benchmark_20305630.log 2>&1'
while tmux -L "$scs_bench_socket" has-session -t benchmark 2>/dev/null; do
    sleep 15
done
tail -15 runs/ST19_original_torch_benchmark_20305630.log
python -c 'import json; from pathlib import Path; s=json.loads(Path("runs/ST19_original_torch_benchmark_20305630_v2/status.json").read_text()); assert s["state"] == "completed", s'
