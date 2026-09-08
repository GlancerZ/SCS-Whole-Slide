#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
tmux -L scs-speed-fp32 new-session -d -s speed 'bash benchmarks/scs_paper/speed_fp32_worker.sh > runs/SCS_stereo_whole_v1/speed_explore_v1/fp32_worker.log 2>&1'
while tmux -L scs-speed-fp32 has-session -t speed 2>/dev/null; do sleep 10; done
test -f runs/SCS_stereo_whole_v1/speed_explore_v1/validation_compile_trunk_fp32.json
