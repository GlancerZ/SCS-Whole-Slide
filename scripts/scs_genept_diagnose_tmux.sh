#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs-torch/bin/activate
socket=scs-genept-diagnosis
session=collapse-probe
if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "diagnosis already running" >&2
    exit 1
fi
tmux -L "$socket" new-session -d -s "$session" \
  "exec python -B -u -m optimizations.scs_streaming.diagnose_genept --output runs/ST19_shared_6000/diagnosis_genept_v1 > runs/ST19_shared_6000.diagnosis_genept_v1.log 2>&1"
echo "started $socket/$session"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
    sleep 30
done
test -f runs/ST19_shared_6000/diagnosis_genept_v1/completed.json
