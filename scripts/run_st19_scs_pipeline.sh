#!/usr/bin/env bash
set -euo pipefail

workspace=/home/glancerz/labwork/codex/TLS
run_dir="$workspace/runs/ST19_dense_1200_scs"
prepared_dir="$workspace/prepared/ST19_dense_1200"

mkdir -p "$run_dir"
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/pipeline.exit"' EXIT

source "$workspace/.venv-scs/bin/activate"

python "$workspace/scripts/run_scs_patch.py" \
  --scs-dir "$workspace/SCS" \
  --expression "$prepared_dir/expression.tsv" \
  --image "$prepared_dir/hematoxylin_registered.tif" \
  --run-dir "$run_dir" \
  --stage preprocess

python "$workspace/scripts/run_scs_patch.py" \
  --scs-dir "$workspace/SCS" \
  --expression "$prepared_dir/expression.tsv" \
  --image "$prepared_dir/hematoxylin_registered.tif" \
  --run-dir "$run_dir" \
  --stage train \
  --epochs 100

python "$workspace/scripts/run_scs_patch.py" \
  --scs-dir "$workspace/SCS" \
  --expression "$prepared_dir/expression.tsv" \
  --image "$prepared_dir/hematoxylin_registered.tif" \
  --run-dir "$run_dir" \
  --stage postprocess
