#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
tmux -L scs-paper-seqfish new-session -d -s benchmark 'bash benchmarks/scs_paper/seqfish_worker.sh > runs/SCS_paper_benchmark_v1/seqfish_worker.log 2>&1'
while tmux -L scs-paper-seqfish has-session -t benchmark 2>/dev/null; do sleep 20; done
test -f runs/SCS_paper_benchmark_v1/seqfish_rep1_fov1/benchmark.json
