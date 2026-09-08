#!/usr/bin/env bash
set -euo pipefail
source scripts/cellist_env.sh
while tmux -L "cellist-nuclei-tissue-${SLURM_JOB_ID}" has-session -t nuclei 2>/dev/null; do
    sleep 10
done
test -f runs/ST19_cellist/shared_nuclei_tissue/completed.json
python -u scripts/cellist_whole.py --workers 2 --tiles x10_y11,x11_y11
python -u scripts/cellist_seam_qa.py
python -u scripts/cellist_whole.py --workers 6
