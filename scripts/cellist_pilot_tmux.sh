#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/cellist_env.sh
: "${SLURM_JOB_ID:?Run in the CPU allocation}"
cellist_socket="cellist-pilot-${SLURM_JOB_ID}"
if ! tmux -L "$cellist_socket" has-session -t pilot 2>/dev/null; then
    tmux -L "$cellist_socket" new-session -d -s pilot \
        "cd '$cellist_workspace' && bash scripts/cellist_pilot_worker.sh > runs/ST19_cellist/pilot.log 2>&1"
fi
# Keep the containing Slurm step alive while tmux is doing the work.
while tmux -L "$cellist_socket" has-session -t pilot 2>/dev/null; do
    sleep 10
done
test -f runs/ST19_cellist/pilot/completed.json
