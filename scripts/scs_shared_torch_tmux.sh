#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs-torch/bin/activate

socket=scs-shared-torch
session=train-scale4
data_root=${SCS_DATA_ROOT:-runs/ST19_shared_6000}
batch_size=${SCS_BATCH_SIZE:-5760}
epochs=${SCS_EPOCHS:-1}
run_name=${SCS_RUN_NAME:-torch_scale4_bs5760}
log=${SCS_LOG:-runs/ST19_shared_6000.torch_scale4.log}

if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "tmux session already exists: $socket/$session" >&2
    exit 1
fi

tmux -L "$socket" new-session -d -s "$session" \
    "exec python -B -u -m optimizations.scs_streaming.shared_train_torch train \
      --root '$data_root' \
      --scale 4 --amp bf16 --batch-size '$batch_size' --epochs '$epochs' \
      --workers 0 --prefetch 1 --run-name '$run_name' \
      >> '$log' 2>&1"

echo "started tmux session $socket/$session; log=$log"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
    sleep 30
done
echo "tmux session finished: $socket/$session"
