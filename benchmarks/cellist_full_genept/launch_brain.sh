#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
: "${SLURM_JOB_ID:?Requires compute allocation}"
socket="cellist-full-genept-${SLURM_JOB_ID}"
if ! tmux -L "$socket" has-session -t brain 2>/dev/null; then
    tmux -L "$socket" new-session -d -s brain \
      "bash benchmarks/cellist_full_genept/brain_worker.sh >> runs/Cellist_full_genept_v1/logs/brain.log 2>&1"
fi
while tmux -L "$socket" has-session -t brain 2>/dev/null; do sleep 10; done
test -f runs/Cellist_full_genept_v1/datasets/stereo_mouse_brain/genept_all_scale4/inference_completed.json
