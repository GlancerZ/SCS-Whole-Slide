#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
source scripts/cellist_env.sh
: "${SLURM_JOB_ID:?Run inside the CPU allocation}"
qc_socket="cellist-qc-${SLURM_JOB_ID}"
if ! tmux -L "$qc_socket" has-session -t qc 2>/dev/null; then
    tmux -L "$qc_socket" new-session -d -s qc \
        "cd '$cellist_workspace' && source scripts/cellist_env.sh && python -u scripts/cellist_paper_qc.py > runs/ST19_cellist/qc_paper/qc.log 2>&1"
fi
while tmux -L "$qc_socket" has-session -t qc 2>/dev/null; do
    sleep 5
done
test -f runs/ST19_cellist/qc_paper/completed.json
