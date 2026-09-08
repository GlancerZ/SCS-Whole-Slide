#!/usr/bin/env bash
set -euo pipefail
cd /home/glancerz/labwork/codex/TLS
tmux -L scs-paper new-session -d -s prepare 'bash benchmarks/scs_paper/prepare_worker.sh > runs/SCS_paper_benchmark_v1/prepare.log 2>&1'
# Keep the srun step alive: detached children are killed when its cgroup exits.
while tmux -L scs-paper has-session -t prepare 2>/dev/null; do sleep 20; done
test -f runs/SCS_paper_benchmark_v1/2107/prepared.json
