#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs-torch/bin/activate
socket=scs-genept-diagnosis
session=repair-pilot2000
if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "repair pilot already running" >&2
    exit 1
fi
tmux -L "$socket" new-session -d -s "$session" \
  "bash scripts/scs_2000_repair_pilot_worker.sh > runs/ST19_shared_6000.repair_pilot2000.log 2>&1"
echo "started $socket/$session"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
    sleep 30
done
test -f runs/ST19_shared_6000/pilot2000_raw_v1/completed.json
test -f runs/ST19_shared_6000/pilot2000_genept_v1/completed.json
