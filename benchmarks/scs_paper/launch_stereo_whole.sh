#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
tmux -L scs-two-datasets new-session -d -s new-method 'bash benchmarks/scs_paper/stereo_whole_worker.sh >> runs/SCS_stereo_whole_v1/worker.log 2>&1'
while tmux -L scs-two-datasets list-sessions >/dev/null 2>&1; do sleep 20; done
test -f runs/SCS_stereo_whole_v1/prepared.json
test -f runs/SCS_stereo_whole_v1/liver/2107/prepared.json
