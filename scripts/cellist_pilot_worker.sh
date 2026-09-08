#!/usr/bin/env bash
set -euo pipefail
source scripts/cellist_env.sh
python -u scripts/cellist_cpu_stage.py inspect
for cellist_stage in watershed seg; do
    if [[ ! -f "runs/ST19_cellist/pilot/${cellist_stage}.done.json" ]]; then
        time python -u scripts/cellist_cpu_stage.py "$cellist_stage"
    fi
done
python -u scripts/cellist_cpu_stage.py qa
