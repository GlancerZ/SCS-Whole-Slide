#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs-torch/bin/activate
socket=scs-genept-diagnosis
session=genes2000-probe
if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "2000-gene diagnosis already running" >&2
    exit 1
fi
tmux -L "$socket" new-session -d -s "$session" \
  "while tmux -L scs-genept-diagnosis has-session -t collapse-probe 2>/dev/null; do sleep 5; done; exec python -B -u -m optimizations.scs_streaming.diagnose_genept --output runs/ST19_shared_6000/diagnosis_genept_2000_v1 --cases genept2000_muon,raw2000_muon,genept2000_scaled_muon,raw2000_low_muon,position_only_muon,expression_only_muon,baseline_low_muon > runs/ST19_shared_6000.diagnosis_genept_2000_v1.log 2>&1"
echo "started $socket/$session"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
    sleep 30
done
test -f runs/ST19_shared_6000/diagnosis_genept_2000_v1/completed.json
