#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
mkdir -p runs/SCS_stereo_whole_v1/speed_explore_v1
# A separate tmux server owns this step: never attach benchmark lifetime to
# the running brain training server or vice versa.
tmux -L scs-speed-explore new-session -d -s speed 'bash benchmarks/scs_paper/speed_worker.sh > runs/SCS_stereo_whole_v1/speed_explore_v1/worker.log 2>&1'
while tmux -L scs-speed-explore has-session -t speed 2>/dev/null; do sleep 10; done
test -f runs/SCS_stereo_whole_v1/speed_explore_v1/compile_trunk_b4096.json
