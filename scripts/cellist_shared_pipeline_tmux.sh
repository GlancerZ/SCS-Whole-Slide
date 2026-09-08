#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/cellist_env.sh
: "${SLURM_JOB_ID:?Run in the CPU allocation}"
cellist_socket="cellist-pipeline-tissue-${SLURM_JOB_ID}"
if ! tmux -L "$cellist_socket" has-session -t pipeline 2>/dev/null; then
    tmux -L "$cellist_socket" new-session -d -s pipeline \
        "cd '$cellist_workspace' && bash scripts/cellist_shared_pipeline_worker.sh > runs/ST19_cellist/pipeline_tissue.log 2>&1"
fi
while tmux -L "$cellist_socket" has-session -t pipeline 2>/dev/null; do
    sleep 10
done
test -f runs/ST19_cellist/whole_shared_nuclei_tissue/merged/completed.json
