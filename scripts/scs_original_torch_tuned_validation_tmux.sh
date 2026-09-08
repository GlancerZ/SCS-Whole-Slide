#!/usr/bin/env bash
set -euo pipefail
cd /lustre09/project/6102157/glancerz/codex/TLS
source .venv-scs-torch/bin/activate
test "${SLURM_JOB_ID}" = "20305630"
scs_validation_socket=scs-tuned-validation-20305630
if tmux -L "$scs_validation_socket" has-session -t validation 2>/dev/null; then
    echo 'Validation already running' >&2
    exit 1
fi
tmux -L "$scs_validation_socket" new-session -d -s validation \
    'bash /lustre09/project/6102157/glancerz/codex/TLS/scripts/scs_original_torch_tuned_validation.sh >> /lustre09/project/6102157/glancerz/codex/TLS/runs/ST19_original_torch_tuned_validation_20305630.log 2>&1'
while tmux -L "$scs_validation_socket" has-session -t validation 2>/dev/null; do
    sleep 15
done
tail -5 runs/ST19_original_torch_tuned_validation_20305630.log
python -c 'import json; from pathlib import Path; paths=[Path("runs") / ("ST19_original_torch_tuned_fp32_bs"+str(b)+"_20305630/status.json") for b in (256,10,64)]; assert all(json.loads(p.read_text())["state"]=="completed" for p in paths)'
