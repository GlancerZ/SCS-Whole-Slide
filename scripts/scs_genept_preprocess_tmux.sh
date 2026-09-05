#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
source .venv-scs/bin/activate

socket=scs-genept
session=preprocess-allgenes-linear
data_root=${SCS_DATA_ROOT:-runs/ST19_shared_6000}
dataset_name=${GENEPT_DATASET_NAME:-genept_allgenes_linear_random90}
asset_root=${GENEPT_ASSET_ROOT:-$data_root/genept_assets}
gene_embeddings=${GENEPT_EMBEDDINGS:-}
log=${GENEPT_LOG:-$data_root/genept_allgenes_preprocess.log}

if [[ -z "$gene_embeddings" ]]; then
  gene_embeddings=$(find "$asset_root" -name GenePT_gene_embedding_ada_text.pickle -print -quit)
fi
if [[ -z "$gene_embeddings" || ! -f "$gene_embeddings" ]]; then
  echo "GenePT ada embedding pickle was not found below $asset_root" >&2
  exit 1
fi
if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $socket/$session" >&2
  exit 1
fi

tmux -L "$socket" new-session -d -s "$session" \
  "set -euo pipefail
   python -B -u -m optimizations.scs_streaming.genept_dataset probe \
     --root '$data_root' --gene-embeddings '$gene_embeddings' \
     > '$data_root/genept_allgenes_probe.json'
   python -B -u -m optimizations.scs_streaming.genept_dataset build \
     --root '$data_root' --gene-embeddings '$gene_embeddings' \
     --output-name '$dataset_name' --chunk-size 8192 \
     >> '$log' 2>&1"

echo "started tmux session $socket/$session; log=$log"
while tmux -L "$socket" has-session -t "$session" 2>/dev/null; do
  sleep 30
done
echo "tmux session finished: $socket/$session"
