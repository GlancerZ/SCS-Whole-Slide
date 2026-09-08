#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
tmux -L scs-paper-train new-session -d -s benchmark 'bash benchmarks/scs_paper/train_worker.sh > runs/SCS_paper_benchmark_v1/train_worker.log 2>&1'
while tmux -L scs-paper-train has-session -t benchmark 2>/dev/null; do sleep 20; done
test -f runs/SCS_paper_benchmark_v1/2107/genept_scale4/completed.json
