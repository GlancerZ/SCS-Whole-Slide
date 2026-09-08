#!/usr/bin/env bash
set -euo pipefail
workspace=/home/glancerz/labwork/codex/TLS
cd "$workspace"
: "${SLURM_JOB_ID:?Run this launcher inside the selected Slurm allocation}"
source "$workspace/.venv-scs/bin/activate"
if tmux -L scs-whole has-session -t queue 2>/dev/null; then
    echo "Whole-slide queue is already running on this node."
    exit 0
fi
remaining=$(squeue -h -j "$SLURM_JOB_ID" -o '%L')
scs_hours=$(awk -v value="$remaining" 'BEGIN {n=split(value,a,":"); if(n==3) seconds=a[1]*3600+a[2]*60+a[3]; else if(n==2) seconds=a[1]*60+a[2]; else exit 2; if(seconds<1200) exit 3; print (seconds-600)/3600}')
tmux -L scs-whole new-session -d -s queue \
    "source '$workspace/.venv-scs/bin/activate'; python -u '$workspace/scripts/scs_whole_worker.py' --hours '$scs_hours' --workers 2 >> '$workspace/runs/ST19_whole_scs/queue.log' 2>&1"
while tmux -L scs-whole has-session -t queue 2>/dev/null; do
    sleep 15
done
