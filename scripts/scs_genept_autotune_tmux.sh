#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs-torch/bin/activate

socket=scs-genept-gpu
session=autotune-scale4
data_root=${SCS_DATA_ROOT:-runs/ST19_shared_6000}
dataset_name=${GENEPT_DATASET_NAME:-genept_allgenes_linear_random90}
output=${GENEPT_AUTOTUNE_OUTPUT:-$data_root/autotune_genept_allgenes_scale4.json}
log=${GENEPT_AUTOTUNE_LOG:-$data_root/autotune_genept_allgenes_scale4.log}

if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $socket/$session" >&2
  exit 1
fi

tmux -L "$socket" new-session -d -s "$session" \
  "exec python -B -u -m optimizations.scs_streaming.shared_train_torch autotune \
    --root '$data_root' --scale 4 --amp bf16 \
    --start-batch 4096 --max-batch 8192 --batch-granularity 128 --repeats 2 \
    --input-pipeline genept --dataset-name '$dataset_name' --residency mmap \
    --output '$output' >> '$log' 2>&1"

echo "started tmux session $socket/$session; log=$log"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
  sleep 30
done
echo "tmux session finished: $socket/$session"
