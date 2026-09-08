#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
tmux -L scs-two-datasets new-session -d -s brain-train 'bash benchmarks/scs_paper/two_dataset_train_worker.sh brain >> runs/SCS_stereo_whole_v1/brain_train_worker.log 2>&1'
tmux -L scs-two-datasets new-session -d -s liver-train 'bash benchmarks/scs_paper/two_dataset_train_worker.sh liver >> runs/SCS_stereo_whole_v1/liver_train_worker.log 2>&1'
while tmux -L scs-two-datasets has-session -t brain-train 2>/dev/null || tmux -L scs-two-datasets has-session -t liver-train 2>/dev/null; do sleep 20; done
test -f runs/SCS_stereo_whole_v1/genept_all_scale4/whole_slide/completed.json
test -f runs/SCS_stereo_whole_v1/liver/2107/genept_all_scale4/whole_slide/completed.json
